from __future__ import annotations

import re
from dataclasses import dataclass

from carm.data.schema import Action
from carm.models.interfaces import ProbeResult


ABSTAIN_TEMPLATE = "I cannot answer reliably because the modalities conflict."


@dataclass
class PolicyConfig:
    semantic_threshold: float = 0.82
    min_overlap_tokens: int = 2
    abstain_message: str = ABSTAIN_TEMPLATE


YES_NO_MAP = {
    "y": "yes",
    "yeah": "yes",
    "yep": "yes",
    "true": "yes",
    "n": "no",
    "nope": "no",
    "false": "no",
}


def normalize_answer(text: str) -> str:
    stripped = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    return YES_NO_MAP.get(stripped, stripped)


def _is_structured(text: str) -> bool:
    return bool(re.fullmatch(r"(yes|no|\d+|left|right|red|blue|black|white)", text))


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def semantic_similarity(a: str, b: str) -> float:
    ta, tb = _token_set(a), _token_set(b)
    if not ta and not tb:
        return 1.0
    inter = len(ta.intersection(tb))
    union = len(ta.union(tb))
    return inter / max(1, union)


def answers_agree(a: str, b: str, cfg: PolicyConfig | None = None) -> bool:
    cfg = cfg or PolicyConfig()
    na = normalize_answer(a)
    nb = normalize_answer(b)

    if _is_structured(na) or _is_structured(nb):
        return na == nb

    ta = _token_set(na)
    tb = _token_set(nb)
    overlap = len(ta.intersection(tb))
    if overlap < cfg.min_overlap_tokens:
        return False
    return semantic_similarity(na, nb) >= cfg.semantic_threshold


def apply_action_and_generate(
    action: Action,
    vision_probe: ProbeResult,
    text_probe: ProbeResult,
    cfg: PolicyConfig | None = None,
) -> tuple[str, bool, dict[str, str]]:
    """
    Returns (final_answer, abstained, audit_info).
    audit_info documents effective modality suppression path.
    """
    cfg = cfg or PolicyConfig()

    if action == Action.TRUST_VISION:
        return vision_probe.answer_text, False, {"path": "vision_only", "suppressed": "text"}

    if action == Action.TRUST_TEXT:
        return text_probe.answer_text, False, {"path": "text_only", "suppressed": "vision"}

    if action == Action.ABSTAIN:
        return cfg.abstain_message, True, {"path": "abstain", "suppressed": "both"}

    if answers_agree(vision_probe.answer_text, text_probe.answer_text, cfg=cfg):
        # Canonicalize to deterministic normalized answer.
        final = normalize_answer(vision_probe.answer_text)
        return final, False, {"path": "require_agreement", "suppressed": "none"}

    return cfg.abstain_message, True, {"path": "require_agreement_abstain", "suppressed": "both"}
