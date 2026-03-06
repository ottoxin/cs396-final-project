from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from carm.data.schema import ConflictExample


@dataclass
class PredictionOutput:
    final_answer: str
    abstained: bool
    confidence: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class AnswerOutput:
    raw_text: str
    answer_confidence: float | None = None
    confidence_source: str = "model"
    metadata: dict[str, Any] | None = None


@dataclass
class PolicyOutput:
    pred_conflict_type: str
    pred_action: str
    abstained: bool
    r_v: float | None = None
    r_t: float | None = None
    policy_confidence: float | None = None
    confidence_source: str = "model"
    audit: dict[str, Any] | None = None


class Predictor(Protocol):
    name: str

    def predict(self, ex: ConflictExample) -> PredictionOutput:
        ...


class AnswerPredictor(Protocol):
    name: str

    def predict_answer(self, ex: ConflictExample) -> AnswerOutput:
        ...

    def predict_policy(self, ex: ConflictExample) -> PolicyOutput | None:
        ...
