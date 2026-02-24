from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from carm.data.labeling import derive_oracle_action
from carm.data.schema import (
    Action,
    AnswerType,
    ConflictExample,
    CorruptModality,
    EvidenceModality,
    Family,
    Operator,
    Split,
)


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
    "eleven": 11,
    "twelve": 12,
}


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at root: {path}")
    return data


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.lower().split())


def infer_family(question: str) -> Family | None:
    q = _normalize_whitespace(question)
    if re.match(r"^(is|are)\b", q):
        return Family.EXISTENCE
    if q.startswith("how many "):
        return Family.COUNT
    if q.startswith("what color "):
        return Family.ATTRIBUTE_COLOR
    return None


def _normalize_yes_no(answer: str) -> str | None:
    a = _normalize_whitespace(answer)
    if a in {"yes", "y", "true"}:
        return "yes"
    if a in {"no", "n", "false"}:
        return "no"
    return None


def _normalize_count(answer: str) -> str | None:
    a = _normalize_whitespace(answer)
    if re.fullmatch(r"\d+", a):
        return str(int(a))
    if a in NUMBER_WORDS:
        return str(NUMBER_WORDS[a])
    return None


def _normalize_color(answer: str, color_vocab: set[str]) -> str | None:
    a = _normalize_whitespace(answer)
    return a if a in color_vocab else None


def normalize_answer(answer: str, family: Family, color_vocab: set[str]) -> str | None:
    if family == Family.EXISTENCE:
        return _normalize_yes_no(answer)
    if family == Family.COUNT:
        return _normalize_count(answer)
    if family == Family.ATTRIBUTE_COLOR:
        return _normalize_color(answer, color_vocab)
    return None


def _question_subject_tokens(question: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", _normalize_whitespace(question))
    drop = {
        "is",
        "are",
        "there",
        "a",
        "an",
        "the",
        "how",
        "many",
        "what",
        "color",
        "of",
        "in",
        "on",
        "at",
    }
    return {t for t in tokens if t not in drop and len(t) > 2}


def _caption_numbers(caption: str) -> set[int]:
    out: set[int] = set()
    for tok in re.findall(r"[a-z0-9]+", caption.lower()):
        if tok.isdigit():
            out.add(int(tok))
        elif tok in NUMBER_WORDS:
            out.add(NUMBER_WORDS[tok])
    return out


def _caption_supports_existence(question: str, answer: str, caption: str) -> bool:
    subj = _question_subject_tokens(question)
    if not subj:
        return False
    cap_tokens = set(re.findall(r"[a-z0-9]+", caption.lower()))
    overlap = bool(subj.intersection(cap_tokens))
    return overlap if answer == "yes" else not overlap


def _caption_supports_count(answer: str, caption: str) -> bool:
    nums = _caption_numbers(caption)
    return int(answer) in nums


def _caption_supports_color(answer: str, caption: str) -> bool:
    return bool(re.search(rf"\b{re.escape(answer)}\b", caption, flags=re.IGNORECASE))


def caption_supports_answer(question: str, family: Family, answer: str, caption: str) -> bool:
    if family == Family.EXISTENCE:
        return _caption_supports_existence(question, answer, caption)
    if family == Family.COUNT:
        return _caption_supports_count(answer, caption)
    if family == Family.ATTRIBUTE_COLOR:
        return _caption_supports_color(answer, caption)
    return False


def _answer_type_for_family(family: Family) -> AnswerType:
    if family == Family.EXISTENCE:
        return AnswerType.BOOLEAN
    if family == Family.COUNT:
        return AnswerType.INTEGER
    if family == Family.ATTRIBUTE_COLOR:
        return AnswerType.COLOR
    return AnswerType.UNKNOWN


def _evidence_for_family(family: Family) -> EvidenceModality:
    if family == Family.COUNT:
        return EvidenceModality.BOTH
    if family == Family.ATTRIBUTE_COLOR:
        return EvidenceModality.VISION_REQUIRED
    if family == Family.EXISTENCE:
        return EvidenceModality.EITHER
    return EvidenceModality.EITHER


def _load_questions(path: str | Path, source: str) -> list[dict[str, Any]]:
    root = _load_json(path)
    out: list[dict[str, Any]] = []
    for q in root.get("questions", []):
        out.append(
            {
                "question_id": int(q["question_id"]),
                "image_id": int(q["image_id"]),
                "question": str(q["question"]),
                "source": source,
            }
        )
    return out


def _load_annotations(path: str | Path) -> dict[int, str]:
    root = _load_json(path)
    out: dict[int, str] = {}
    for ann in root.get("annotations", []):
        qid = int(ann["question_id"])
        out[qid] = str(ann.get("multiple_choice_answer", ""))
    return out


def _load_captions(*paths: str | Path) -> dict[int, list[str]]:
    out: dict[int, list[str]] = defaultdict(list)
    for path in paths:
        root = _load_json(path)
        for ann in root.get("annotations", []):
            out[int(ann["image_id"])].append(str(ann["caption"]))
    return out


def _image_path(image_id: int, source: str, train_dir: str, val_dir: str) -> str:
    file_name = f"COCO_{source}2014_{image_id:012d}.jpg"
    root = train_dir if source == "train" else val_dir
    return str(Path(root) / file_name)


def build_base_examples(
    *,
    vqa_questions_train: str,
    vqa_questions_val: str,
    vqa_annotations_train: str,
    vqa_annotations_val: str,
    coco_captions_train: str,
    coco_captions_val: str,
    image_train_dir: str,
    image_val_dir: str,
    families: list[Family],
    color_vocab: list[str],
    consistency_filter: bool,
    seed: int,
    max_per_family: int | None = None,
) -> tuple[list[ConflictExample], dict[str, Any]]:
    rng = random.Random(seed)

    questions = _load_questions(vqa_questions_train, "train") + _load_questions(vqa_questions_val, "val")
    annotations = _load_annotations(vqa_annotations_train)
    annotations.update(_load_annotations(vqa_annotations_val))
    captions_by_image = _load_captions(coco_captions_train, coco_captions_val)

    family_set = set(families)
    color_vocab_set = {c.lower() for c in color_vocab}

    grouped: dict[Family, list[ConflictExample]] = defaultdict(list)
    dropped = CounterDict()

    for row in questions:
        qid = int(row["question_id"])
        answer_raw = annotations.get(qid)
        if answer_raw is None:
            dropped.inc("missing_annotation")
            continue

        family = infer_family(row["question"])
        if family is None or family not in family_set:
            dropped.inc("family_filtered")
            continue

        answer = normalize_answer(answer_raw, family, color_vocab_set)
        if answer is None:
            dropped.inc("answer_normalization_failed")
            continue

        image_id = int(row["image_id"])
        captions = captions_by_image.get(image_id, [])
        if not captions:
            dropped.inc("missing_caption")
            continue

        caption = None
        if consistency_filter:
            for c in captions:
                if caption_supports_answer(row["question"], family, answer, c):
                    caption = c
                    break
            if caption is None:
                dropped.inc("consistency_filter_failed")
                continue
        else:
            caption = captions[0]

        base_id = f"vqa-{qid}"
        image_path = _image_path(
            image_id=image_id,
            source=str(row["source"]),
            train_dir=image_train_dir,
            val_dir=image_val_dir,
        )

        ex = ConflictExample(
            example_id=f"{base_id}::clean",
            base_id=base_id,
            variant_id="clean",
            image_path=image_path,
            text_input=caption,
            question=str(row["question"]),
            gold_answer=answer,
            split=Split.TRAIN,
            family=family,
            operator=Operator.CLEAN,
            corrupt_modality=CorruptModality.NONE,
            severity=0,
            answer_type=_answer_type_for_family(family),
            oracle_action=Action.REQUIRE_AGREEMENT,
            source_image_id=f"{row['source']}::{image_id}",
            template_id=None,
            evidence_modality=_evidence_for_family(family),
            heldout_family_flag=False,
            heldout_severity_flag=False,
            hard_swap_flag=False,
            metadata={
                "question_id": qid,
                "image_id": image_id,
                "source": row["source"],
            },
        )
        ex.oracle_action = derive_oracle_action(ex.corrupt_modality)
        grouped[family].append(ex)

    for family in list(grouped.keys()):
        rng.shuffle(grouped[family])
        if max_per_family is not None:
            grouped[family] = grouped[family][: max(0, int(max_per_family))]

    examples: list[ConflictExample] = []
    for family in sorted(grouped.keys(), key=lambda f: f.value):
        examples.extend(grouped[family])

    examples = sorted(examples, key=lambda ex: ex.base_id)

    stats = {
        "total_examples": len(examples),
        "family_counts": {
            family.value: len(grouped.get(family, []))
            for family in sorted(family_set, key=lambda f: f.value)
        },
        "dropped": dropped.to_dict(),
        "consistency_filter": consistency_filter,
        "seed": seed,
    }
    return examples, stats


class CounterDict:
    def __init__(self) -> None:
        self._data: dict[str, int] = defaultdict(int)

    def inc(self, key: str) -> None:
        self._data[key] += 1

    def to_dict(self) -> dict[str, int]:
        return dict(sorted(self._data.items(), key=lambda kv: kv[0]))
