from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from carm.models.features import extract_probe_features
from carm.models.interfaces import BackboneResult, ProbeResult


@dataclass
class BackboneConfig:
    hidden_size: int = 128
    seq_len: int = 32
    vocab: tuple[str, ...] = (
        "yes",
        "no",
        "red",
        "blue",
        "green",
        "yellow",
        "black",
        "white",
        "1",
        "2",
        "3",
        "unknown",
    )


class MockFrozenBackbone:
    """Deterministic local backbone adapter for Phase A pipeline validation."""

    name = "mock_frozen_backbone"

    def __init__(self, config: BackboneConfig | None = None) -> None:
        self.config = config or BackboneConfig()

    def _seed_from_payload(self, payload: str) -> int:
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
        return int(digest, 16)

    def _sample_distribution(self, payload: str) -> torch.Tensor:
        seed = self._seed_from_payload(payload)
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        logits = torch.randn(len(self.config.vocab), generator=g)
        return torch.softmax(logits, dim=-1)

    def _hidden_states(self, payload: str) -> torch.Tensor:
        seed = self._seed_from_payload("hs::" + payload)
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        return torch.randn(self.config.seq_len, self.config.hidden_size, generator=g)

    def _decode(self, dist: torch.Tensor) -> str:
        idx = int(torch.argmax(dist).item())
        return self.config.vocab[idx]

    def run_backbone_multimodal(self, image: str, text: str, question: str) -> BackboneResult:
        payload = f"mm::{image}::{text}::{question}"
        dist = self._sample_distribution(payload)
        return BackboneResult(
            hidden_states=self._hidden_states(payload),
            answer_dist=dist,
            answer_text=self._decode(dist),
        )

    def run_probe_vision_only(self, image: str, question: str) -> ProbeResult:
        payload = f"v::{image}::{question}"
        dist = self._sample_distribution(payload)
        return ProbeResult(
            answer_dist=dist,
            answer_text=self._decode(dist),
            features=extract_probe_features(dist),
        )

    def run_probe_text_only(self, text: str, question: str) -> ProbeResult:
        payload = f"t::{text}::{question}"
        dist = self._sample_distribution(payload)
        return ProbeResult(
            answer_dist=dist,
            answer_text=self._decode(dist),
            features=extract_probe_features(dist),
        )


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
        self._vocab_token_ids: list[int] | None = None

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
        except Exception as exc:  # pragma: no cover - depends on optional runtime deps
            raise RuntimeError(
                "Qwen adapter requires transformers with Qwen2.5-VL support. "
                "Install/update with: pip install 'transformers>=4.57.0'"
            ) from exc

        try:
            self._processor = AutoProcessor.from_pretrained(self.model_name)
        except ImportError as exc:  # pragma: no cover - optional runtime deps
            raise RuntimeError(
                "Qwen adapter processor load failed. Install vision deps with: "
                "pip install torchvision"
            ) from exc
        self._tokenizer = getattr(self._processor, "tokenizer", None)
        if self._tokenizer is None:
            raise RuntimeError("Loaded processor does not expose a tokenizer.")

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=self.dtype,
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

    def _vocab_prompt(self, context: str) -> str:
        options = ", ".join(self.config.vocab)
        return (
            "You are answering a VQA question.\n"
            f"Choose exactly one answer from this closed set: {options}.\n"
            "Return only the answer token with no extra text.\n"
            f"{context}\n"
            "Answer:"
        )

    def _ensure_vocab_token_ids(self) -> list[int]:
        self._ensure_loaded()
        assert self._tokenizer is not None
        if self._vocab_token_ids is not None:
            return self._vocab_token_ids

        ids: list[int] = []
        unk_id = getattr(self._tokenizer, "unk_token_id", None)
        for token in self.config.vocab:
            tok_ids = self._tokenizer.encode(f" {token}", add_special_tokens=False)
            if not tok_ids:
                tok_ids = self._tokenizer.encode(token, add_special_tokens=False)
            if not tok_ids:
                ids.append(unk_id if unk_id is not None else 0)
            else:
                ids.append(int(tok_ids[0]))
        self._vocab_token_ids = ids
        return ids

    def _prepare_inputs(self, prompt: str, image_path: Path | None) -> dict[str, Any]:
        self._ensure_loaded()
        assert self._processor is not None

        if image_path is None:
            chat_text = prompt
            if hasattr(self._processor, "apply_chat_template"):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
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
                batch = self._processor(
                    text=[chat_text],
                    images=[rgb],
                    return_tensors="pt",
                )

        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        return batch

    def _dist_from_next_token_logits(self, logits: torch.Tensor) -> torch.Tensor:
        token_ids = self._ensure_vocab_token_ids()
        pieces: list[torch.Tensor] = []
        for tok_id in token_ids:
            if 0 <= tok_id < logits.shape[-1]:
                pieces.append(logits[tok_id].float().unsqueeze(0))
            else:
                pieces.append(torch.tensor([-1e9], dtype=torch.float32, device=logits.device))
        scores = torch.cat(pieces, dim=0)
        probs = torch.softmax(scores, dim=-1).detach().cpu().float()
        if torch.isnan(probs).any():
            probs = torch.full((len(self.config.vocab),), 1.0 / len(self.config.vocab), dtype=torch.float32)
        return probs

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

    def _infer(self, prompt: str, image_path: Path | None) -> tuple[torch.Tensor, torch.Tensor, str]:
        self._ensure_loaded()
        assert self._model is not None

        inputs = self._prepare_inputs(prompt, image_path)
        with torch.inference_mode():
            outputs = self._model(**inputs, output_hidden_states=True, return_dict=True)

        last_logits = outputs.logits[0, -1, :]
        dist = self._dist_from_next_token_logits(last_logits)
        pred_idx = int(torch.argmax(dist).item())
        answer = str(self.config.vocab[pred_idx])

        hidden = outputs.hidden_states[-1][0]
        hidden_fmt = self._format_hidden(hidden)
        return hidden_fmt, dist, answer

    @staticmethod
    def _clone_backbone_result(result: BackboneResult) -> BackboneResult:
        return BackboneResult(
            hidden_states=result.hidden_states.clone(),
            answer_dist=result.answer_dist.clone(),
            answer_text=result.answer_text,
        )

    @staticmethod
    def _clone_probe_result(result: ProbeResult) -> ProbeResult:
        return ProbeResult(
            answer_dist=result.answer_dist.clone(),
            answer_text=result.answer_text,
            features=result.features.clone(),
        )

    def run_backbone_multimodal(self, image: str, text: str, question: str) -> BackboneResult:
        key = f"{image}||{text}||{question}"
        if self.cache_results and key in self._cache_mm:
            return self._clone_backbone_result(self._cache_mm[key])

        image_path = self._resolve_image_path(image)
        prompt = self._vocab_prompt(f"Caption: {text}\nQuestion: {question}")
        hidden_states, dist, answer = self._infer(prompt, image_path=image_path)
        result = BackboneResult(
            hidden_states=hidden_states,
            answer_dist=dist,
            answer_text=answer,
        )
        if self.cache_results:
            self._cache_mm[key] = result
        return self._clone_backbone_result(result)

    def run_probe_vision_only(self, image: str, question: str) -> ProbeResult:
        key = f"{image}||{question}"
        if self.cache_results and key in self._cache_v:
            return self._clone_probe_result(self._cache_v[key])

        image_path = self._resolve_image_path(image)
        prompt = self._vocab_prompt(f"Question: {question}")
        _, dist, answer = self._infer(prompt, image_path=image_path)
        result = ProbeResult(
            answer_dist=dist,
            answer_text=answer,
            features=extract_probe_features(dist),
        )
        if self.cache_results:
            self._cache_v[key] = result
        return self._clone_probe_result(result)

    def run_probe_text_only(self, text: str, question: str) -> ProbeResult:
        key = f"{text}||{question}"
        if self.cache_results and key in self._cache_t:
            return self._clone_probe_result(self._cache_t[key])

        prompt = self._vocab_prompt(f"Caption: {text}\nQuestion: {question}")
        _, dist, answer = self._infer(prompt, image_path=None)
        result = ProbeResult(
            answer_dist=dist,
            answer_text=answer,
            features=extract_probe_features(dist),
        )
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
