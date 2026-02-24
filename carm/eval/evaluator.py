from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from carm.data.io import write_jsonl
from carm.data.labeling import derive_reliability_target
from carm.data.schema import ConflictExample, CorruptModality, Family
from carm.eval.baselines import BaseBaseline
from carm.eval.metrics import summarize_metrics
from carm.models.backbone import MockFrozenBackbone
from carm.models.carm_model import CARMHeads, select_action
from carm.models.policy import apply_action_and_generate, normalize_answer


class CARMPredictor:
    name = "carm"

    def __init__(self, model: CARMHeads, backbone: MockFrozenBackbone, device: str = "cpu") -> None:
        self.model = model.to(torch.device(device))
        self.backbone = backbone
        self.device = torch.device(device)

    @staticmethod
    def _vision_payload(ex: ConflictExample) -> str:
        recipe = ex.metadata.get("vision_recipe") if isinstance(ex.metadata, dict) else None
        if isinstance(recipe, dict) and "payload" in recipe:
            return str(recipe["payload"])
        return ex.image_path

    def predict(self, ex: ConflictExample) -> dict[str, Any]:
        self.model.eval()
        with torch.no_grad():
            mm = self.backbone.run_backbone_multimodal(self._vision_payload(ex), ex.text_input, ex.question)
            pv = self.backbone.run_probe_vision_only(self._vision_payload(ex), ex.question)
            pt = self.backbone.run_probe_text_only(ex.text_input, ex.question)

            conflict_logits, reliability, action_logits = self.model.carm_forward(
                anchor_states=mm.hidden_states.to(self.device),
                phi_v=pv.features.to(self.device),
                phi_t=pt.features.to(self.device),
            )
            action = select_action(action_logits)
            answer, abstained, audit = apply_action_and_generate(action, pv, pt)

            pred_conf_idx = int(torch.argmax(conflict_logits, dim=-1).item())
            conflict_map = [
                Family.NONE.value,
                Family.EXISTENCE.value,
                Family.COUNT.value,
                Family.ATTRIBUTE_COLOR.value,
            ]
            pred_conf = conflict_map[min(pred_conf_idx, len(conflict_map) - 1)]

            return {
                "pred_conflict_type": pred_conf,
                "pred_action": action.value,
                "final_answer": answer,
                "abstained": bool(abstained),
                "confidence": float(torch.softmax(action_logits, dim=-1).max().item()),
                "r_v": float(reliability.squeeze(0)[0].item()),
                "r_t": float(reliability.squeeze(0)[1].item()),
                "audit": audit,
            }


def evaluate_predictor(
    predictor: CARMPredictor | BaseBaseline,
    examples: list[ConflictExample],
    output_dir: str | Path,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for ex in examples:
        pred = predictor.predict(ex)
        if not isinstance(pred, dict):
            pred = pred.__dict__

        rel_t = derive_reliability_target(
            evidence_modality=ex.evidence_modality,
            corrupt_modality=ex.corrupt_modality,
            severity=ex.severity,
        )

        final_answer = str(pred["final_answer"])
        correct = normalize_answer(final_answer) == normalize_answer(ex.gold_answer)

        row: dict[str, Any] = {
            "example_id": ex.example_id,
            "base_id": ex.base_id,
            "variant_id": ex.variant_id,
            "image_path": ex.image_path,
            "text_input": ex.text_input,
            "question": ex.question,
            "gold_answer": ex.gold_answer,
            "split": ex.split.value,
            "family": ex.family.value,
            "operator": ex.operator.value,
            "corrupt_modality": ex.corrupt_modality.value,
            "severity": ex.severity,
            "answer_type": ex.answer_type.value,
            "oracle_action": ex.oracle_action.value,
            "heldout_family_flag": ex.heldout_family_flag,
            "heldout_severity_flag": ex.heldout_severity_flag,
            "hard_swap_flag": ex.hard_swap_flag,
            "pred_conflict_type": str(pred["pred_conflict_type"]),
            "pred_action": str(pred["pred_action"]),
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

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
