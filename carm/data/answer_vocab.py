from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from carm.data.schema import ConflictExample, Family


DEFAULT_COLOR_VOCAB = (
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "brown",
    "gray",
    "orange",
    "pink",
    "purple",
)

YES_ALIASES = {"yes", "y", "true", "1"}
NO_ALIASES = {"no", "n", "false", "0"}

COUNT_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

COLOR_ALIASES = {
    "grey": "gray",
    "violet": "purple",
}


@dataclass(frozen=True)
class ParsedAnswer:
    """Candidate extracted from raw generation plus its ontology mapping."""

    candidate_text: str | None
    canonicalized_candidate: str | None


def normalize_text(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    return " ".join(tokens)


def canonicalize_candidate_answer(
    candidate: str,
    family: Family,
    *,
    recognized_color_labels: Iterable[str] | None = None,
) -> str | None:
    """Map a candidate answer string into the family's canonical ontology form.

    Family-valid means:
    - existence: canonicalizes to ``yes`` or ``no``
    - count: canonicalizes to an integer string
    - color: canonicalizes to a recognized canonical color label
    """

    norm = normalize_text(candidate)
    if not norm:
        return None

    if family == Family.EXISTENCE:
        if norm in YES_ALIASES:
            return "yes"
        if norm in NO_ALIASES:
            return "no"
        for token in norm.split():
            if token in YES_ALIASES:
                return "yes"
            if token in NO_ALIASES:
                return "no"
        return None

    if family == Family.COUNT:
        if re.fullmatch(r"\d+", norm):
            return str(int(norm))
        for token in norm.split():
            if token in COUNT_WORDS:
                return str(COUNT_WORDS[token])
        match = re.search(r"\d+", norm)
        if match is not None:
            return str(int(match.group(0)))
        return None

    if family == Family.ATTRIBUTE_COLOR:
        recognized = {str(v).strip().lower() for v in (recognized_color_labels or DEFAULT_COLOR_VOCAB) if str(v).strip()}
        tokens = norm.split()
        if norm in COLOR_ALIASES:
            mapped = COLOR_ALIASES[norm]
            return mapped if mapped in recognized else None
        if norm in recognized:
            return norm
        for token in tokens:
            mapped = COLOR_ALIASES.get(token, token)
            if mapped in recognized:
                return mapped
        return None

    return None


def canonicalize_family_answer_for_agreement(
    answer: str,
    family: Family,
    *,
    recognized_color_labels: Iterable[str] | None = None,
) -> str | None:
    """Canonicalize an answer string for family-aware agreement checks.

    This reuses the same family ontologies used by vocab building/parsing.
    For colors, fall back to the more permissive gold normalization path so
    identical custom labels such as ``beige`` can still compare canonically.
    """

    canonical = canonicalize_candidate_answer(
        answer,
        family,
        recognized_color_labels=recognized_color_labels,
    )
    if canonical is not None:
        return canonical

    if family == Family.ATTRIBUTE_COLOR:
        norm = normalize_text(answer)
        if not norm:
            return None
        tokens = norm.split()
        if len(tokens) != 1:
            return None
        return COLOR_ALIASES.get(tokens[0], tokens[0])
    return None


def normalize_gold_answer(answer: str, family: Family) -> str | None:
    """Normalize structured gold answers into canonical family labels."""

    norm = normalize_text(answer)
    if not norm:
        return None

    if family == Family.EXISTENCE:
        return canonicalize_candidate_answer(norm, family)

    if family == Family.COUNT:
        return canonicalize_candidate_answer(norm, family)

    if family == Family.ATTRIBUTE_COLOR:
        tokens = norm.split()
        if not tokens:
            return None
        if len(tokens) == 1:
            return COLOR_ALIASES.get(tokens[0], tokens[0])
        for token in tokens:
            mapped = COLOR_ALIASES.get(token, token)
            if mapped in DEFAULT_COLOR_VOCAB or token in COLOR_ALIASES:
                return mapped
        return COLOR_ALIASES.get(tokens[0], tokens[0])

    return None


def parse_generated_answer(
    generated_text: str,
    family: Family,
    *,
    recognized_color_labels: Iterable[str] | None = None,
) -> ParsedAnswer:
    """Extract a candidate answer from free-form generation, then canonicalize it."""

    norm = normalize_text(generated_text)
    if not norm:
        return ParsedAnswer(candidate_text=None, canonicalized_candidate=None)

    if family == Family.EXISTENCE:
        for token in norm.split():
            if token in YES_ALIASES | NO_ALIASES:
                return ParsedAnswer(
                    candidate_text=token,
                    canonicalized_candidate=canonicalize_candidate_answer(token, family),
                )
        return ParsedAnswer(candidate_text=None, canonicalized_candidate=None)

    if family == Family.COUNT:
        match = re.search(r"\d+", norm)
        if match is not None:
            token = match.group(0)
            return ParsedAnswer(
                candidate_text=token,
                canonicalized_candidate=canonicalize_candidate_answer(token, family),
            )
        for token in norm.split():
            if token in COUNT_WORDS:
                return ParsedAnswer(
                    candidate_text=token,
                    canonicalized_candidate=canonicalize_candidate_answer(token, family),
                )
        return ParsedAnswer(candidate_text=None, canonicalized_candidate=None)

    if family == Family.ATTRIBUTE_COLOR:
        recognized = {str(v).strip().lower() for v in (recognized_color_labels or DEFAULT_COLOR_VOCAB) if str(v).strip()}
        if norm in recognized or norm in COLOR_ALIASES:
            return ParsedAnswer(
                candidate_text=norm,
                canonicalized_candidate=canonicalize_candidate_answer(
                    norm,
                    family,
                    recognized_color_labels=recognized,
                ),
            )
        for token in norm.split():
            mapped = COLOR_ALIASES.get(token, token)
            if mapped in recognized:
                return ParsedAnswer(
                    candidate_text=token,
                    canonicalized_candidate=canonicalize_candidate_answer(
                        token,
                        family,
                        recognized_color_labels=recognized,
                    ),
                )
        return ParsedAnswer(candidate_text=None, canonicalized_candidate=None)

    return ParsedAnswer(candidate_text=norm, canonicalized_candidate=None)


def normalize_family_vocab(values: Iterable[str], family: Family) -> tuple[str, ...]:
    """Normalize a family vocab, de-duplicate it, and force ``unknown`` last."""

    normalized: list[str] = []
    seen: set[str] = set()

    if family == Family.EXISTENCE:
        return ("yes", "no", "unknown")

    for raw in values:
        token = str(raw).strip().lower()
        if not token or token == "unknown":
            continue
        if family == Family.COUNT:
            token = canonicalize_candidate_answer(token, family) or token
        elif family == Family.ATTRIBUTE_COLOR:
            token = COLOR_ALIASES.get(token, token)
        if token not in seen:
            seen.add(token)
            normalized.append(token)

    if family == Family.COUNT:
        normalized.sort(key=lambda item: int(item) if re.fullmatch(r"\d+", item) else 10**9)
    else:
        normalized.sort()
    normalized.append("unknown")
    return tuple(normalized)


def build_family_vocabs(examples: list[ConflictExample]) -> dict[Family, tuple[str, ...]]:
    """Build family vocabs from the examples provided."""

    observed: dict[Family, set[str]] = {
        Family.EXISTENCE: set(),
        Family.COUNT: set(),
        Family.ATTRIBUTE_COLOR: set(),
    }
    for ex in examples:
        if ex.family not in observed:
            continue
        normalized = normalize_gold_answer(ex.gold_answer, ex.family)
        if normalized is not None:
            observed[ex.family].add(normalized)

    return {
        Family.EXISTENCE: ("yes", "no", "unknown"),
        Family.COUNT: normalize_family_vocab(observed[Family.COUNT], Family.COUNT),
        Family.ATTRIBUTE_COLOR: normalize_family_vocab(observed[Family.ATTRIBUTE_COLOR], Family.ATTRIBUTE_COLOR),
    }


def save_family_vocabs(vocabs: dict[Family, tuple[str, ...]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {family.value: list(values) for family, values in vocabs.items()}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_family_vocabs(path: str | Path) -> dict[Family, tuple[str, ...]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at root for family vocab: {path}")

    loaded: dict[Family, tuple[str, ...]] = {}
    for family in (Family.EXISTENCE, Family.COUNT, Family.ATTRIBUTE_COLOR):
        raw_values = data.get(family.value, [])
        if not isinstance(raw_values, list):
            raise ValueError(f"Expected list vocab for family {family.value}: {path}")
        loaded[family] = normalize_family_vocab(raw_values, family)
    return loaded


def family_vocab_jsonable(vocabs: dict[Family, tuple[str, ...]]) -> dict[str, list[str]]:
    return {family.value: list(values) for family, values in vocabs.items()}


def canonicalization_mapping_from_family_vocabs(vocabs: dict[Family, tuple[str, ...]]) -> dict[str, object]:
    count_values = [int(v) for v in vocabs.get(Family.COUNT, ()) if re.fullmatch(r"\d+", v)]
    color_values = [v for v in vocabs.get(Family.ATTRIBUTE_COLOR, ()) if v != "unknown"]
    count_range = {"min": min(count_values), "max": max(count_values)} if count_values else {"min": 0, "max": 0}
    return {
        "family_vocab_overrides": family_vocab_jsonable(vocabs),
        "count_range": count_range,
        "color_vocab": color_values,
    }
