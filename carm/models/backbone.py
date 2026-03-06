from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from carm.data.answer_vocab import (
    DEFAULT_COLOR_VOCAB,
    ParsedAnswer,
    normalize_family_vocab,
    parse_generated_answer,
)
from carm.data.schema import Family
from carm.data.vqa_coco import infer_family
from carm.models.features import extract_probe_features
from carm.models.interfaces import BackboneResult, ProbeResult


def _normalize_color_vocab(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = str(raw).strip().lower()
        if not token:
            continue
        token = normalize_family_vocab((token,), Family.ATTRIBUTE_COLOR)[0]
        if token not in seen:
            seen.add(token)
            normalized.append(token)
    return tuple(normalized) or DEFAULT_COLOR_VOCAB


@dataclass
class BackboneConfig:
    hidden_size: int = 128
    seq_len: int = 32
    max_new_tokens: int = 10
    count_min: int = 0
    count_max: int = 20
    color_vocab: tuple[str, ...] | list[str] = DEFAULT_COLOR_VOCAB
    vocab: tuple[str, ...] | list[str] = field(default_factory=tuple)
    family_vocab_overrides: dict[str, tuple[str, ...] | list[str]] | None = None
    force_fallback_distribution: bool = False

    def __post_init__(self) -> None:
        count_min = int(self.count_min)
        count_max = int(self.count_max)
        if count_max < count_min:
            count_min, count_max = count_max, count_min
        self.count_min = count_min
        self.count_max = count_max

        self.color_vocab = _normalize_color_vocab(tuple(self.color_vocab))

        raw_overrides = self.family_vocab_overrides or {}
        normalized_overrides: dict[str, tuple[str, ...]] = {}
        for raw_family, raw_values in raw_overrides.items():
            family_key = str(raw_family).strip().lower()
            try:
                family = Family(family_key)
            except ValueError:
                continue
            normalized_overrides[family.value] = normalize_family_vocab(tuple(raw_values), family)
        if Family.EXISTENCE.value in normalized_overrides:
            normalized_overrides[Family.EXISTENCE.value] = ("yes", "no", "unknown")
        self.family_vocab_overrides = normalized_overrides or None
        self.force_fallback_distribution = bool(self.force_fallback_distribution)

        raw_vocab = tuple(self.vocab)
        if raw_vocab:
            self.vocab = tuple(str(item).strip().lower() for item in raw_vocab if str(item).strip())
            return

        union: list[str] = ["yes", "no"]
        if self.family_vocab_overrides:
            for values in self.family_vocab_overrides.values():
                union.extend(v for v in values if v != "unknown")
        else:
            union.extend(self.color_vocab)
            union.extend(str(i) for i in range(self.count_min, self.count_max + 1))
        union.append("unknown")

        seen: set[str] = set()
        deduped: list[str] = []
        for item in union:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        self.vocab = tuple(deduped)


class Qwen25VLAdapter:
    name = "qwen2_5_vl_7b"

    def __init__(
        self,
        model_name: str | None = None,
        config: BackboneConfig | None = None,
        *,
        device: str = "auto",
        torch_dtype: str = "auto",
        cache_results: bool = True,
    ) -> None:
        self.model_name = model_name or "Qwen/Qwen2.5-VL-7B-Instruct"
        self.config = config or BackboneConfig()
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_torch_dtype(torch_dtype, self.device.type)
        self.cache_results = bool(cache_results)
        self._project_root = Path(__file__).resolve().parents[2]

        self._model: Any | None = None
        self._processor: Any | None = None
        self._tokenizer: Any | None = None
        self._family_vocab_token_ids: dict[tuple[str, ...], list[int]] = {}

        self._cache_mm: dict[str, BackboneResult] = {}
        self._cache_v: dict[str, ProbeResult] = {}
        self._cache_t: dict[str, ProbeResult] = {}

    @staticmethod
    def _resolve_device(raw: str) -> torch.device:
        if raw == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if raw.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(raw)

    @staticmethod
    def _resolve_torch_dtype(raw: str, device_type: str) -> torch.dtype:
        if raw == "auto":
            if device_type == "cuda":
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
            return torch.float32

        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        chosen = mapping.get(raw.lower())
        if chosen is None:
            raise ValueError(f"Unsupported torch dtype: {raw}")
        if device_type == "cpu" and chosen in {torch.float16, torch.bfloat16}:
            return torch.float32
        return chosen

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except Exception as exc:  # pragma: no cover - optional runtime deps
            raise RuntimeError(
                "Qwen adapter requires transformers with Qwen2.5-VL support. "
                "Install/update with: pip install 'transformers>=4.57.0'"
            ) from exc

        try:
            self._processor = AutoProcessor.from_pretrained(self.model_name)
        except ImportError as exc:  # pragma: no cover - optional runtime deps
            raise RuntimeError(
                "Qwen adapter processor load failed. Install vision deps with: pip install torchvision"
            ) from exc

        self._tokenizer = getattr(self._processor, "tokenizer", None)
        if self._tokenizer is None:
            raise RuntimeError("Loaded processor does not expose a tokenizer.")

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        self._model.to(self.device)
        self._model.eval()

    def _resolve_image_path(self, image_path: str) -> Path:
        candidate = Path(image_path)
        if candidate.exists():
            return candidate
        rooted = self._project_root / image_path
        if rooted.exists():
            return rooted
        raise FileNotFoundError(f"Image path not found for Qwen adapter: {image_path}")

    def _prompt_instruction(self, family: Family | None) -> str:
        if family == Family.EXISTENCE:
            return "Answer yes or no only."
        if family == Family.COUNT:
            return "Answer with a single integer only."
        if family == Family.ATTRIBUTE_COLOR:
            return "Answer with a single color word only."
        return "Answer with a short answer only."

    def _qa_prompt(self, context: str, family: Family | None) -> str:
        return (
            "You are answering a VQA question.\n"
            f"{self._prompt_instruction(family)}\n"
            f"{context}\n"
            "Answer:"
        )

    def _family_vocab(self, family: Family | None) -> tuple[str, ...]:
        if family is not None and self.config.family_vocab_overrides:
            override = self.config.family_vocab_overrides.get(family.value)
            if override:
                return tuple(override)
        if family == Family.EXISTENCE:
            return ("yes", "no", "unknown")
        if family == Family.COUNT:
            return tuple(str(i) for i in range(self.config.count_min, self.config.count_max + 1)) + ("unknown",)
        if family == Family.ATTRIBUTE_COLOR:
            return tuple(self.config.color_vocab) + ("unknown",)
        return tuple(self.config.vocab)

    def _prepare_inputs(self, prompt: str, image_path: Path | None) -> dict[str, Any]:
        self._ensure_loaded()
        assert self._processor is not None

        if image_path is None:
            chat_text = prompt
            if hasattr(self._processor, "apply_chat_template"):
                messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                chat_text = self._processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            batch = self._processor(text=[chat_text], return_tensors="pt")
        else:
            with Image.open(image_path) as img:
                rgb = img.convert("RGB")
                chat_text = prompt
                if hasattr(self._processor, "apply_chat_template"):
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    chat_text = self._processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                batch = self._processor(text=[chat_text], images=[rgb], return_tensors="pt")

        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        return batch

    def _format_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        hs = hidden.detach().float().cpu()
        seq_len, dim = hs.shape

        target_seq = int(self.config.seq_len)
        target_dim = int(self.config.hidden_size)

        if dim > target_dim:
            hs = hs[:, :target_dim]
        elif dim < target_dim:
            pad = torch.zeros((seq_len, target_dim - dim), dtype=hs.dtype)
            hs = torch.cat([hs, pad], dim=-1)

        seq_len = hs.shape[0]
        if seq_len > target_seq:
            hs = hs[-target_seq:, :]
        elif seq_len < target_seq:
            pad = torch.zeros((target_seq - seq_len, hs.shape[1]), dtype=hs.dtype)
            hs = torch.cat([pad, hs], dim=0)
        return hs

    def _recognized_color_labels(self, family: Family | None) -> set[str]:
        labels = set(DEFAULT_COLOR_VOCAB)
        if family == Family.ATTRIBUTE_COLOR:
            labels.update(v for v in self._family_vocab(family) if v != "unknown")
        return labels

    def _parse_answer(self, generated_text: str, family: Family | None) -> ParsedAnswer:
        if family in {Family.EXISTENCE, Family.COUNT, Family.ATTRIBUTE_COLOR}:
            return parse_generated_answer(
                generated_text,
                family,
                recognized_color_labels=self._recognized_color_labels(family),
            )
        return ParsedAnswer(candidate_text=generated_text.strip() or None, canonicalized_candidate=None)

    def _token_ids_for_vocab(self, vocab: tuple[str, ...]) -> list[int]:
        self._ensure_loaded()
        assert self._tokenizer is not None

        if vocab in self._family_vocab_token_ids:
            return list(self._family_vocab_token_ids[vocab])

        templates = [
            "{token}",
            " {token}",
            "Answer: {token}",
            "The answer is {token}",
            "\n{token}",
        ]

        chosen: list[int] | None = None
        for template in templates:
            ids: list[int] = []
            failed = False
            for token in vocab:
                encoded = self._tokenizer.encode(template.format(token=token), add_special_tokens=False)
                if not encoded:
                    failed = True
                    break
                ids.append(int(encoded[-1]))
            if failed:
                continue
            if len(set(ids)) == len(ids):
                chosen = ids
                break

        if chosen is None:
            raise ValueError(
                "Unable to derive unique token ids for family vocab from tokenizer encodings. "
                f"vocab={vocab}"
            )

        self._family_vocab_token_ids[vocab] = list(chosen)
        return list(chosen)

    def _uniform_dist(self, vocab: tuple[str, ...]) -> torch.Tensor:
        if not vocab:
            return torch.ones(1, dtype=torch.float32)
        return torch.full((len(vocab),), 1.0 / len(vocab), dtype=torch.float32)

    def _sequence_confidence(self, generated_scores: Any, generated_tokens: torch.Tensor) -> float:
        if not generated_scores:
            return 0.5

        probs: list[float] = []
        token_ids = generated_tokens.detach().cpu().tolist()
        for step, token_id in enumerate(token_ids):
            if step >= len(generated_scores):
                break
            step_scores = generated_scores[step]
            if not isinstance(step_scores, torch.Tensor):
                continue
            if token_id < 0 or token_id >= step_scores.shape[-1]:
                continue
            step_probs = torch.softmax(step_scores[0].detach().float(), dim=-1)
            probs.append(float(step_probs[token_id].item()))

        if not probs:
            return 0.5
        mean_prob = sum(probs) / len(probs)
        return max(1e-6, min(0.999999, mean_prob))

    def _fallback_dist(self, answer_text: str, family: Family | None, confidence: float) -> torch.Tensor:
        vocab = self._family_vocab(family)
        if not vocab:
            return torch.ones(1, dtype=torch.float32)
        if len(vocab) == 1:
            return torch.ones(1, dtype=torch.float32)

        answer = answer_text if answer_text in vocab else "unknown"
        if answer not in vocab:
            answer = vocab[-1]
        answer_idx = vocab.index(answer)
        conf = max(1.0 / len(vocab), min(0.999999, float(confidence)))
        base = (1.0 - conf) / max(1, len(vocab) - 1)
        dist = torch.full((len(vocab),), base, dtype=torch.float32)
        dist[answer_idx] = conf
        dist = dist / dist.sum()
        return dist

    def _distribution_from_first_token_logits(self, logits: torch.Tensor | None, family: Family | None) -> torch.Tensor | None:
        if self.config.force_fallback_distribution:
            return None
        vocab = self._family_vocab(family)
        if logits is None:
            return None

        try:
            token_ids = self._token_ids_for_vocab(vocab)
        except ValueError:
            return None
        pieces: list[torch.Tensor] = []
        for token_id in token_ids:
            if 0 <= token_id < logits.shape[-1]:
                pieces.append(logits[token_id].detach().float().unsqueeze(0))
            else:
                pieces.append(torch.tensor([-1e9], dtype=torch.float32, device=logits.device))
        scores = torch.cat(pieces, dim=0)
        probs = torch.softmax(scores, dim=-1).detach().cpu().float()
        if torch.isnan(probs).any():
            return None
        return probs

    def _infer(
        self,
        prompt: str,
        image_path: Path | None,
        family: Family | None,
    ) -> tuple[torch.Tensor, torch.Tensor, str, str, dict[str, Any]]:
        self._ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None

        inputs = self._prepare_inputs(prompt, image_path)
        with torch.inference_mode():
            prompt_outputs = self._model(**inputs, output_hidden_states=True, return_dict=True)
            generated = self._model.generate(
                **inputs,
                max_new_tokens=int(self.config.max_new_tokens),
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        prompt_len = int(inputs["input_ids"].shape[1])
        sequences = generated.sequences
        generated_tokens = sequences[:, prompt_len:]
        generated_text = self._tokenizer.decode(generated_tokens[0].detach().cpu().tolist(), skip_special_tokens=True)
        parsed = self._parse_answer(generated_text, family)
        vocab = self._family_vocab(family)
        if family in {Family.EXISTENCE, Family.COUNT, Family.ATTRIBUTE_COLOR}:
            canonicalized_candidate = parsed.canonicalized_candidate
            parsed_in_active_vocab = canonicalized_candidate is not None and canonicalized_candidate in vocab
            answer_text = canonicalized_candidate if parsed_in_active_vocab else "unknown"
        else:
            canonicalized_candidate = None
            answer_text = parsed.candidate_text or "unknown"
            parsed_in_active_vocab = answer_text in vocab
        seq_confidence = self._sequence_confidence(getattr(generated, "scores", None), generated_tokens[0])

        first_scores = None
        if getattr(generated, "scores", None):
            score0 = generated.scores[0]
            if isinstance(score0, torch.Tensor):
                first_scores = score0[0]
        dist = self._distribution_from_first_token_logits(first_scores, family)
        projection_succeeded = dist is not None
        used_fallback_dist = False
        if dist is None:
            dist = self._fallback_dist(answer_text, family, seq_confidence)
            used_fallback_dist = True

        hidden = prompt_outputs.hidden_states[-1][0]
        hidden_fmt = self._format_hidden(hidden)
        dist_argmax_label = vocab[int(torch.argmax(dist).item())] if vocab else answer_text
        result_meta = {
            "projection_succeeded": bool(projection_succeeded),
            "used_fallback_dist": bool(used_fallback_dist),
            "parsed_unknown": answer_text == "unknown",
            "parsed_in_active_vocab": bool(parsed_in_active_vocab),
            "canonicalized_candidate": canonicalized_candidate,
            "out_of_vocab_generation": canonicalized_candidate is not None and canonicalized_candidate not in vocab,
            "dist_argmax_label": dist_argmax_label,
            "parsed_argmax_agree": answer_text == dist_argmax_label,
        }
        return hidden_fmt, dist, answer_text, generated_text, result_meta

    @staticmethod
    def _clone_backbone_result(result: BackboneResult) -> BackboneResult:
        return BackboneResult(
            hidden_states=result.hidden_states.clone(),
            answer_dist=result.answer_dist.clone(),
            answer_text=result.answer_text,
            raw_text=result.raw_text,
            metadata=dict(result.metadata or {}),
        )

    @staticmethod
    def _clone_probe_result(result: ProbeResult) -> ProbeResult:
        return ProbeResult(
            answer_dist=result.answer_dist.clone(),
            answer_text=result.answer_text,
            features=result.features.clone(),
            raw_text=result.raw_text,
            metadata=dict(result.metadata or {}),
        )

    def run_backbone_multimodal(self, image: str, text: str, question: str) -> BackboneResult:
        key = f"{image}||{text}||{question}"
        if self.cache_results and key in self._cache_mm:
            return self._clone_backbone_result(self._cache_mm[key])

        family = infer_family(question)
        image_path = self._resolve_image_path(image)
        prompt = self._qa_prompt(f"Caption: {text}\nQuestion: {question}", family)
        hidden_states, dist, answer, raw_text, metadata = self._infer(prompt, image_path=image_path, family=family)
        result = BackboneResult(hidden_states=hidden_states, answer_dist=dist, answer_text=answer, raw_text=raw_text, metadata=metadata)
        if self.cache_results:
            self._cache_mm[key] = result
        return self._clone_backbone_result(result)

    def run_probe_vision_only(self, image: str, question: str) -> ProbeResult:
        key = f"{image}||{question}"
        if self.cache_results and key in self._cache_v:
            return self._clone_probe_result(self._cache_v[key])

        family = infer_family(question)
        image_path = self._resolve_image_path(image)
        prompt = self._qa_prompt(f"Question: {question}", family)
        _, dist, answer, raw_text, metadata = self._infer(prompt, image_path=image_path, family=family)
        result = ProbeResult(answer_dist=dist, answer_text=answer, features=extract_probe_features(dist), raw_text=raw_text, metadata=metadata)
        if self.cache_results:
            self._cache_v[key] = result
        return self._clone_probe_result(result)

    def run_probe_text_only(self, text: str, question: str) -> ProbeResult:
        key = f"{text}||{question}"
        if self.cache_results and key in self._cache_t:
            return self._clone_probe_result(self._cache_t[key])

        family = infer_family(question)
        prompt = self._qa_prompt(f"Caption: {text}\nQuestion: {question}", family)
        _, dist, answer, raw_text, metadata = self._infer(prompt, image_path=None, family=family)
        result = ProbeResult(answer_dist=dist, answer_text=answer, features=extract_probe_features(dist), raw_text=raw_text, metadata=metadata)
        if self.cache_results:
            self._cache_t[key] = result
        return self._clone_probe_result(result)


class LlavaNextAdapter:
    name = "llava_next_8b"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or "llava-hf/llava-v1.6-8b"

    def run_backbone_multimodal(self, image: str, text: str, question: str) -> BackboneResult:
        raise NotImplementedError("LlavaNextAdapter is a Phase A stub. Runnable inference is next-wave work.")

    def run_probe_vision_only(self, image: str, question: str) -> ProbeResult:
        raise NotImplementedError("LlavaNextAdapter is a Phase A stub. Runnable inference is next-wave work.")

    def run_probe_text_only(self, text: str, question: str) -> ProbeResult:
        raise NotImplementedError("LlavaNextAdapter is a Phase A stub. Runnable inference is next-wave work.")
