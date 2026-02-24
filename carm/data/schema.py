from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


DATASET_RECORD_VERSION = "v1"


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST_ID = "test_id"
    TEST_OOD_FAMILY = "test_ood_family"
    TEST_OOD_SEVERITY = "test_ood_severity"
    TEST_OOD_HARD_SWAP = "test_ood_hard_swap"


class Family(str, Enum):
    NONE = "none"
    EXISTENCE = "existence"
    COUNT = "count"
    ATTRIBUTE_COLOR = "attribute_color"


class Operator(str, Enum):
    CLEAN = "clean"
    SWAP_EASY = "swap_easy"
    SWAP_HARD = "swap_hard"
    TEXT_EDIT = "text_edit"
    VISION_CORRUPT = "vision_corrupt"
    BOTH = "both"


class CorruptModality(str, Enum):
    NONE = "none"
    VISION = "vision"
    TEXT = "text"
    BOTH = "both"


class EvidenceModality(str, Enum):
    VISION_REQUIRED = "vision_required"
    TEXT_REQUIRED = "text_required"
    BOTH = "both"
    EITHER = "either"


class AnswerType(str, Enum):
    BOOLEAN = "boolean"
    INTEGER = "integer"
    COLOR = "color"
    UNKNOWN = "unknown"


class Action(str, Enum):
    TRUST_VISION = "trust_vision"
    TRUST_TEXT = "trust_text"
    REQUIRE_AGREEMENT = "require_agreement"
    ABSTAIN = "abstain"


_FAMILY_ALIASES = {
    "none": Family.NONE,
    "object": Family.EXISTENCE,
    "attribute": Family.ATTRIBUTE_COLOR,
    "attribute_color": Family.ATTRIBUTE_COLOR,
    "relation": Family.EXISTENCE,
    "count": Family.COUNT,
    "existence": Family.EXISTENCE,
}

_OPERATOR_ALIASES = {
    "clean": Operator.CLEAN,
    "caption_swap": Operator.SWAP_EASY,
    "caption_swap_same_topic": Operator.SWAP_HARD,
    "swap_easy": Operator.SWAP_EASY,
    "swap_hard": Operator.SWAP_HARD,
    "text_edit": Operator.TEXT_EDIT,
    "text_edit_count": Operator.TEXT_EDIT,
    "text_edit_relation": Operator.TEXT_EDIT,
    "text_edit_attribute": Operator.TEXT_EDIT,
    "corrupt": Operator.VISION_CORRUPT,
    "vision_corrupt": Operator.VISION_CORRUPT,
    "vision_blur": Operator.VISION_CORRUPT,
    "vision_occlusion": Operator.VISION_CORRUPT,
    "vision_distractor": Operator.VISION_CORRUPT,
    "both": Operator.BOTH,
}

_SPLIT_ALIASES = {
    "train": Split.TRAIN,
    "val": Split.VAL,
    "test": Split.TEST_ID,
    "test_id": Split.TEST_ID,
    "test_ood_family": Split.TEST_OOD_FAMILY,
    "test_ood_severity": Split.TEST_OOD_SEVERITY,
    "test_ood_hard_swap": Split.TEST_OOD_HARD_SWAP,
}

_CORRUPT_MODALITY_ALIASES = {
    "none": CorruptModality.NONE,
    "vision": CorruptModality.VISION,
    "text": CorruptModality.TEXT,
    "both": CorruptModality.BOTH,
}


@dataclass
class ConflictExample:
    example_id: str
    base_id: str
    variant_id: str
    image_path: str
    text_input: str
    question: str
    gold_answer: str
    split: Split
    family: Family
    operator: Operator
    corrupt_modality: CorruptModality
    severity: int
    answer_type: AnswerType
    oracle_action: Action
    source_image_id: str | None = None
    template_id: str | None = None
    evidence_modality: EvidenceModality = EvidenceModality.EITHER
    heldout_family_flag: bool = False
    heldout_severity_flag: bool = False
    hard_swap_flag: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    record_version: str = DATASET_RECORD_VERSION

    @property
    def conflict_type(self) -> Family:
        return self.family

    @property
    def corruption_family(self) -> str:
        return self.operator.value

    @property
    def corrupted_modality(self) -> CorruptModality:
        return self.corrupt_modality

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "ConflictExample":
        raw_family = str(item.get("family", item.get("conflict_type", "none"))).lower()
        raw_operator = str(item.get("operator", item.get("corruption_family", "clean"))).lower()
        raw_corrupt_modality = str(item.get("corrupt_modality", item.get("corrupted_modality", "none"))).lower()
        raw_split = str(item.get("split", "train")).lower()

        family = _FAMILY_ALIASES.get(raw_family)
        if family is None:
            raise ValueError(f"Unknown family/conflict_type value: {raw_family}")

        operator = _OPERATOR_ALIASES.get(raw_operator)
        if operator is None:
            raise ValueError(f"Unknown operator/corruption_family value: {raw_operator}")

        corrupt_modality = _CORRUPT_MODALITY_ALIASES.get(raw_corrupt_modality)
        if corrupt_modality is None:
            raise ValueError(f"Unknown corrupt_modality value: {raw_corrupt_modality}")

        split = _SPLIT_ALIASES.get(raw_split)
        if split is None:
            raise ValueError(f"Unknown split value: {raw_split}")

        base_id = str(item.get("base_id", str(item.get("example_id", "unknown")).split("::")[0]))
        variant_id = str(item.get("variant_id", str(item.get("example_id", "unknown")).split("::")[-1]))

        raw_answer_type = str(item.get("answer_type", "")).lower().strip()
        if not raw_answer_type:
            if family == Family.EXISTENCE:
                answer_type = AnswerType.BOOLEAN
            elif family == Family.COUNT:
                answer_type = AnswerType.INTEGER
            elif family == Family.ATTRIBUTE_COLOR:
                answer_type = AnswerType.COLOR
            else:
                answer_type = AnswerType.UNKNOWN
        else:
            answer_type = AnswerType(raw_answer_type)

        raw_evidence = str(item.get("evidence_modality", EvidenceModality.EITHER.value)).lower()
        evidence_modality = EvidenceModality(raw_evidence)

        return cls(
            example_id=str(item.get("example_id", f"{base_id}::{variant_id}")),
            base_id=base_id,
            variant_id=variant_id,
            image_path=str(item["image_path"]),
            text_input=str(item["text_input"]),
            question=str(item["question"]),
            gold_answer=str(item["gold_answer"]),
            split=split,
            family=family,
            operator=operator,
            corrupt_modality=corrupt_modality,
            severity=int(item.get("severity", 0)),
            answer_type=answer_type,
            oracle_action=Action(str(item.get("oracle_action", Action.REQUIRE_AGREEMENT.value))),
            source_image_id=item.get("source_image_id"),
            template_id=item.get("template_id"),
            evidence_modality=evidence_modality,
            heldout_family_flag=bool(item.get("heldout_family_flag", False)),
            heldout_severity_flag=bool(item.get("heldout_severity_flag", False)),
            hard_swap_flag=bool(item.get("hard_swap_flag", False)),
            metadata=item.get("metadata", {}),
            record_version=str(item.get("record_version", DATASET_RECORD_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        item = asdict(self)
        item["split"] = self.split.value
        item["family"] = self.family.value
        item["operator"] = self.operator.value
        item["corrupt_modality"] = self.corrupt_modality.value
        item["answer_type"] = self.answer_type.value
        item["oracle_action"] = self.oracle_action.value
        item["evidence_modality"] = self.evidence_modality.value
        # Backward-compatible aliases
        item["conflict_type"] = self.family.value
        item["corrupted_modality"] = self.corrupt_modality.value
        item["corruption_family"] = self.operator.value
        return item
