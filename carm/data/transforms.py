from __future__ import annotations

import random
import re
from dataclasses import replace

from carm.data.labeling import derive_oracle_action
from carm.data.schema import ConflictExample, CorruptModality, Family, Operator


STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "on",
    "in",
    "at",
    "of",
    "to",
    "for",
    "there",
    "this",
    "that",
    "these",
    "those",
}

NUMBER_WORDS = {
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
}


def _derived_id(base_id: str, suffix: str) -> str:
    return f"{base_id}::{suffix}"


def _noun_like_tokens(text: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in toks if t not in STOPWORDS and len(t) > 2}


def noun_jaccard(a: str, b: str) -> float:
    ta = _noun_like_tokens(a)
    tb = _noun_like_tokens(b)
    if not ta and not tb:
        return 1.0
    return len(ta.intersection(tb)) / max(1, len(ta.union(tb)))


def _infer_answer_type_value(example: ConflictExample) -> str:
    return example.answer_type.value


def caption_swap(
    example: ConflictExample,
    donor_text: str,
    operator: Operator,
    hard_swap_flag: bool,
    seed: int,
) -> ConflictExample:
    suffix = "swap_hard" if operator == Operator.SWAP_HARD else "swap_easy"
    swapped = replace(
        example,
        example_id=_derived_id(example.base_id, suffix),
        variant_id=suffix,
        text_input=donor_text,
        operator=operator,
        corrupt_modality=CorruptModality.TEXT,
        severity=1,
        hard_swap_flag=hard_swap_flag,
    )
    swapped.oracle_action = derive_oracle_action(swapped.corrupt_modality)
    swapped.metadata = {
        **example.metadata,
        "swap_seed": seed,
        "swap_operator": operator.value,
        "answer_type_bucket": _infer_answer_type_value(example),
    }
    return swapped


def _replace_first_number_token(text: str) -> tuple[str, bool]:
    m = re.search(r"\b\d+\b", text)
    if m:
        n = int(m.group(0))
        repl = str((n + 1) if n < 99 else n - 1)
        return text[: m.start()] + repl + text[m.end() :], True

    for word, n in NUMBER_WORDS.items():
        pat = rf"\b{re.escape(word)}\b"
        if re.search(pat, text, flags=re.IGNORECASE):
            repl_n = (n + 1) if n < 10 else n - 1
            reverse = {v: k for k, v in NUMBER_WORDS.items()}
            repl_w = reverse.get(repl_n, "one")
            out = re.sub(pat, repl_w, text, count=1, flags=re.IGNORECASE)
            return out, True

    return text, False


def _flip_negation(text: str) -> str:
    if re.search(r"\bnot\b", text, flags=re.IGNORECASE):
        return re.sub(r"\bnot\b", "", text, count=1, flags=re.IGNORECASE).replace("  ", " ").strip()

    aux = re.search(r"\b(is|are|was|were|has|have|do|does|did|can)\b", text, flags=re.IGNORECASE)
    if aux:
        insert_at = aux.end()
        return text[:insert_at] + " not" + text[insert_at:]
    return "not " + text


def _edit_color(text: str, color_vocab: list[str], seed: int) -> str:
    rng = random.Random(seed)
    for color in color_vocab:
        pat = rf"\b{re.escape(color)}\b"
        if re.search(pat, text, flags=re.IGNORECASE):
            alternatives = [c for c in color_vocab if c != color]
            if not alternatives:
                return text
            replacement = rng.choice(alternatives)
            return re.sub(pat, replacement, text, count=1, flags=re.IGNORECASE)

    replacement = rng.choice(color_vocab) if color_vocab else "red"
    return f"{text.rstrip('.')} {replacement}."


def text_edit(
    example: ConflictExample,
    color_vocab: list[str],
    seed: int,
) -> ConflictExample:
    edited_text = example.text_input

    if example.family == Family.COUNT:
        edited_text, changed = _replace_first_number_token(edited_text)
        if not changed:
            edited_text = f"{edited_text.rstrip('.')} There are 3 objects."
    elif example.family == Family.EXISTENCE:
        edited_text = _flip_negation(edited_text)
    elif example.family == Family.ATTRIBUTE_COLOR:
        edited_text = _edit_color(edited_text, color_vocab=color_vocab, seed=seed)

    edited = replace(
        example,
        example_id=_derived_id(example.base_id, "text_edit"),
        variant_id="text_edit",
        text_input=edited_text,
        operator=Operator.TEXT_EDIT,
        corrupt_modality=CorruptModality.TEXT,
        severity=1,
    )
    edited.oracle_action = derive_oracle_action(edited.corrupt_modality)
    edited.metadata = {
        **example.metadata,
        "edit_family": example.family.value,
    }
    return edited


def vision_corrupt(
    example: ConflictExample,
    corruption_type: str,
    severity: int,
) -> ConflictExample:
    payload = f"{example.image_path}|{corruption_type}|s{severity}"
    variant_id = f"vision_corrupt_s{severity}"
    out = replace(
        example,
        example_id=_derived_id(example.base_id, variant_id),
        variant_id=variant_id,
        image_path=example.image_path,
        operator=Operator.VISION_CORRUPT,
        corrupt_modality=CorruptModality.VISION,
        severity=severity,
    )
    out.oracle_action = derive_oracle_action(out.corrupt_modality)
    out.metadata = {
        **example.metadata,
        "vision_recipe": {
            "type": corruption_type,
            "severity": severity,
            "payload": payload,
        },
    }
    return out
