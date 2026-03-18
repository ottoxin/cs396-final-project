from __future__ import annotations

import re

from carm.data.schema import Action, ConflictExample
from carm.eval.baselines import ABSTAIN_ANSWER, BaseBaseline, BaselinePrediction
from carm.models.policy import apply_action_and_generate


PROMPT_ONLY_ABSTAIN_ACTION_TEMPLATE = (
    "You are a multimodal arbitration system.\n"
    "Given the image, caption, and question, return exactly one label from this closed set:\n"
    "TRUST_VISION\n"
    "TRUST_TEXT\n"
    "REQUIRE_AGREEMENT\n"
    "ABSTAIN\n"
    "Do not explain.\n"
    "Caption: {caption}\n"
    "Question: {question}\n"
    "Decision:"
)

_PROMPT_ACTION_MAP = {
    "TRUST_VISION": Action.TRUST_VISION,
    "TRUST_TEXT": Action.TRUST_TEXT,
    "REQUIRE_AGREEMENT": Action.REQUIRE_AGREEMENT,
    "ABSTAIN": Action.ABSTAIN,
}


def parse_prompt_action(text: str) -> Action | None:
    for token in _PROMPT_ACTION_MAP:
        if re.search(rf"\b{token}\b", text.upper()):
            return _PROMPT_ACTION_MAP[token]
    return None


class PromptOnlyAbstainBaseline(BaseBaseline):
    name = "prompt_only_abstain"

    def __init__(self, backbone, prompt_template: str = PROMPT_ONLY_ABSTAIN_ACTION_TEMPLATE) -> None:
        super().__init__(backbone)
        self.prompt_template = str(prompt_template)

    def predict(self, ex: ConflictExample) -> BaselinePrediction:
        generator = getattr(self.backbone, "generate_freeform", None)
        if not callable(generator):
            raise RuntimeError(
                "prompt_only_abstain requires a backbone.generate_freeform(prompt, image=...) implementation."
            )

        prompt = self.prompt_template.format(caption=ex.text_input, question=ex.question)
        generation = generator(prompt, image=self._vision_payload(ex))
        raw_text = str(generation.text or "").strip()
        pred_action = parse_prompt_action(raw_text)

        pv = self.backbone.run_probe_vision_only(self._vision_payload(ex), ex.question)
        pt = self.backbone.run_probe_text_only(ex.text_input, ex.question)
        metadata = self._merge_probe_metadata(pv, pt)
        metadata["prompt_template"] = self.prompt_template
        metadata["prompt_raw_output"] = raw_text

        if pred_action is None:
            metadata["prompt_parse_status"] = "invalid"
            return BaselinePrediction(
                final_answer=ABSTAIN_ANSWER,
                abstained=True,
                confidence=float(generation.confidence or 0.0),
                raw_text=raw_text,
                metadata=metadata,
            )

        metadata["prompt_parse_status"] = "parsed"
        metadata["pred_action"] = pred_action.value
        final_answer, abstained, audit = apply_action_and_generate(pred_action, pv, pt, family=ex.family)
        metadata["audit"] = audit
        return BaselinePrediction(
            final_answer=final_answer,
            abstained=bool(abstained),
            confidence=float(generation.confidence or 0.0),
            raw_text=raw_text,
            metadata=metadata,
        )
