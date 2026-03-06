from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from carm.data.answer_vocab import COLOR_ALIASES, COUNT_WORDS, DEFAULT_COLOR_VOCAB, NO_ALIASES, YES_ALIASES
from carm.data.schema import AnswerType


YES_NO_MAP = {
    **{value: "yes" for value in YES_ALIASES},
    "yeah": "yes",
    "yep": "yes",
    **{value: "no" for value in NO_ALIASES},
    "nope": "no",
}


@dataclass
class CanonicalizationConfig:
    bool_map: dict[str, str] = field(default_factory=lambda: dict(YES_NO_MAP))
    count_min: int = 0
    count_max: int = 20
    color_vocab: set[str] = field(
        default_factory=lambda: set(DEFAULT_COLOR_VOCAB)
    )
    color_synonyms: dict[str, str] = field(
        default_factory=lambda: {
            **COLOR_ALIASES,
            "gold": "yellow",
            "silver": "gray",
        }
    )
    family_vocab_overrides: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "CanonicalizationConfig":
        if not raw:
            return cls()

        cfg = cls()
        bool_map = raw.get("boolean_map")
        if isinstance(bool_map, dict):
            cfg.bool_map.update({str(k).lower(): str(v).lower() for k, v in bool_map.items()})

        count_range = raw.get("count_range")
        if isinstance(count_range, dict):
            if "min" in count_range:
                cfg.count_min = int(count_range["min"])
            if "max" in count_range:
                cfg.count_max = int(count_range["max"])

        color_vocab = raw.get("color_vocab")
        if isinstance(color_vocab, list):
            cfg.color_vocab = {str(x).lower() for x in color_vocab}

        color_synonyms = raw.get("color_synonyms")
        if isinstance(color_synonyms, dict):
            cfg.color_synonyms.update({str(k).lower(): str(v).lower() for k, v in color_synonyms.items()})

        family_vocab_overrides = raw.get("family_vocab_overrides")
        if isinstance(family_vocab_overrides, dict):
            cfg.family_vocab_overrides = {
                str(family).lower(): {str(value).lower() for value in values if str(value).strip() and str(value).lower() != "unknown"}
                for family, values in family_vocab_overrides.items()
                if isinstance(values, list)
            }

        return cfg


@dataclass
class CanonicalizedAnswer:
    normalized_text: str
    canonical_label: str | None
    canonical_status: str  # mapped | unmapped | invalid


def normalize_text(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        return ""
    normalized = " ".join(tokens)
    return YES_NO_MAP.get(normalized, normalized)


def _canonicalize_boolean(norm: str, cfg: CanonicalizationConfig) -> str | None:
    if not norm:
        return None
    if norm in {"yes", "no"}:
        return norm
    if norm in cfg.bool_map:
        mapped = cfg.bool_map[norm]
        return mapped if mapped in {"yes", "no"} else None
    parts = norm.split()
    if len(parts) == 1:
        return None
    for part in parts:
        if part in {"yes", "no"}:
            return part
        mapped = cfg.bool_map.get(part)
        if mapped in {"yes", "no"}:
            return mapped
    return None


def _canonicalize_count(norm: str, cfg: CanonicalizationConfig) -> str | None:
    if not norm:
        return None

    if norm in COUNT_WORDS:
        val = COUNT_WORDS[norm]
        canonical = str(val)
        if "count" in cfg.family_vocab_overrides:
            return canonical if canonical in cfg.family_vocab_overrides["count"] else None
        if cfg.count_min <= val <= cfg.count_max:
            return str(val)
    for part in norm.split():
        if part in COUNT_WORDS:
            val = COUNT_WORDS[part]
            canonical = str(val)
            if "count" in cfg.family_vocab_overrides:
                return canonical if canonical in cfg.family_vocab_overrides["count"] else None
            if cfg.count_min <= val <= cfg.count_max:
                return str(val)

    m = re.search(r"\d+", norm)
    if m is not None:
        val = int(m.group(0))
        canonical = str(val)
        if "count" in cfg.family_vocab_overrides:
            return canonical if canonical in cfg.family_vocab_overrides["count"] else None
        if cfg.count_min <= val <= cfg.count_max:
            return canonical
    return None


def _canonicalize_color(norm: str, cfg: CanonicalizationConfig) -> str | None:
    if not norm:
        return None
    active_color_vocab = cfg.family_vocab_overrides.get("attribute_color", cfg.color_vocab)
    candidate = cfg.color_synonyms.get(norm, norm)
    if candidate in active_color_vocab:
        return candidate

    parts = norm.split()
    if len(parts) > 1:
        for part in parts:
            part_c = cfg.color_synonyms.get(part, part)
            if part_c in active_color_vocab:
                return part_c
    return None


def canonicalize_answer(
    text: str,
    answer_type: AnswerType | str,
    cfg: CanonicalizationConfig | None = None,
) -> CanonicalizedAnswer:
    cfg = cfg or CanonicalizationConfig()
    raw = (text or "").strip()
    if not raw:
        return CanonicalizedAnswer(normalized_text="", canonical_label=None, canonical_status="invalid")

    # Handle explicit abstentions as invalid answer outputs for answer-track metrics.
    if raw.strip().upper() in {"<ABSTAIN>", "ABSTAIN"}:
        norm = normalize_text(raw)
        return CanonicalizedAnswer(normalized_text=norm, canonical_label=None, canonical_status="invalid")

    norm = normalize_text(raw)
    kind = answer_type.value if isinstance(answer_type, AnswerType) else str(answer_type)

    canonical_label: str | None = None
    if kind == AnswerType.BOOLEAN.value:
        canonical_label = _canonicalize_boolean(norm, cfg)
    elif kind == AnswerType.INTEGER.value:
        canonical_label = _canonicalize_count(norm, cfg)
    elif kind == AnswerType.COLOR.value:
        canonical_label = _canonicalize_color(norm, cfg)
    else:
        # Fallback: try all canonicalizers in a predictable order.
        canonical_label = _canonicalize_boolean(norm, cfg)
        if canonical_label is None:
            canonical_label = _canonicalize_count(norm, cfg)
        if canonical_label is None:
            canonical_label = _canonicalize_color(norm, cfg)

    status = "mapped" if canonical_label is not None else "unmapped"
    return CanonicalizedAnswer(normalized_text=norm, canonical_label=canonical_label, canonical_status=status)


def semantic_similarity(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9]+", a.lower()))
    tb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return float(len(ta.intersection(tb)) / max(1, len(ta.union(tb))))


def semantic_match(a: str, b: str, threshold: float = 0.82) -> bool:
    return semantic_similarity(a, b) >= threshold
