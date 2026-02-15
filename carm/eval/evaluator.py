from __future__ import annotations

import json
from pathlib import Path

import torch

from carm.data.io import write_jsonl
from carm.data.labeling import derive_reliability_target
from carm.data.schema import Action, ConflictExample
from carm.eval.baselines import BaseBaseline
from carm.eval.metrics import summarize_metrics
from carm.models.backbone import MockFrozenBackbone
from carm.models.carm_model import CARMHeads, select_action
from carm.models.policy import apply_action_and_generate


def _normalize(s: str) -> str:
    return " ".join(s.lower().split())


class CARMPredictor:
    name = "carm"

    def __init__(self, model: CARMHeads, backbone: MockFrozenBackbone, device: str = "cpu") -> None:
        self.model = model.to(torch.device(device))
        self.backbone = backbone
        self.device = torch.device(device)

    def predict(self, ex: ConflictExample) -> dict:
        self.model.eval()
        with torch.no_grad():
            mm = self.backbone.run_backbone_multimodal(ex.image_path, ex.text_input, ex.question)
            pv = self.backbone.run_probe_vision_only(ex.image_path, ex.question)
            pt = self.backbone.run_probe_text_only(ex.text_input, ex.question)

            conflict_logits, reliability, action_logits = self.model.carm_forward(
                anchor_states=mm.hidden_states.to(self.device),
                phi_v=pv.features.to(self.device),
                phi_t=pt.features.to(self.device),
            )
            action = select_action(action_logits)
            answer, abstained, audit = apply_action_and_generate(action, pv, pt)

            pred_conf_idx = int(torch.argmax(conflict_logits, dim=-1).item())
            conflict_map = ["none", "object", "attribute", "relation", "count"]
            confidence = float(torch.softmax(action_logits, dim=-1).max().item())

            return {
                "pred_conflict_type": conflict_map[pred_conf_idx],
                "pred_action": action.value,
                "final_answer": answer,
                "abstained": bool(abstained),
                "confidence": confidence,
                "r_v": float(reliability.squeeze(0)[0].item()),
                "r_t": float(reliability.squeeze(0)[1].item()),
                "audit": audit,
            }


def evaluate_predictor(
    predictor: CARMPredictor | BaseBaseline,
    examples: list[ConflictExample],
    output_dir: str | Path,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for ex in examples:
        pred = predictor.predict(ex)

        # Baselines return dataclass; CARM returns dict.
        if not isinstance(pred, dict):
            pred = pred.__dict__

        rel_t = derive_reliability_target(
            evidence_modality=ex.evidence_modality,
            corrupted_modality=ex.corrupted_modality,
            severity=ex.severity,
        )

        final_answer = str(pred["final_answer"])
        correct = _normalize(final_answer) == _normalize(ex.gold_answer)

        row = {
            "example_id": ex.example_id,
            "image_path": ex.image_path,
            "text_input": ex.text_input,
            "question": ex.question,
            "gold_answer": ex.gold_answer,
            "split": ex.split.value,
            "conflict_type": ex.conflict_type.value,
            "corrupted_modality": ex.corrupted_modality.value,
            "corruption_family": ex.corruption_family,
            "severity": ex.severity,
            "evidence_modality": ex.evidence_modality.value,
            "oracle_action": ex.oracle_action.value,
            "pred_conflict_type": pred["pred_conflict_type"],
            "pred_action": pred["pred_action"],
            "r_v": float(pred["r_v"]),
            "r_t": float(pred["r_t"]),
            "abstained": bool(pred["abstained"]),
            "final_answer": final_answer,
            "correct": bool(correct),
            "confidence": float(pred.get("confidence", 0.0)),
            "target_r_v": rel_t.r_v,
            "target_r_t": rel_t.r_t,
        }

        if "audit" in pred:
            row["audit"] = pred["audit"]
        records.append(row)

    write_jsonl(out_dir / "per_example_predictions.jsonl", records)
    metrics = summarize_metrics(records)

    # Acceptance checks aligned with proposal targets.
    acceptance = {
        "max_consistent_drop": 1.0,
        "max_monotonicity_violation_rate": 0.15,
        "max_ece": 0.12,
        "max_brier": 0.25,
    }
    metrics["acceptance_thresholds"] = acceptance

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
