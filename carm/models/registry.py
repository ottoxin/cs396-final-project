from __future__ import annotations

from pathlib import Path
from typing import Any

from carm.data.answer_vocab import family_vocab_jsonable, load_family_vocabs
from carm.models.backbone import BackboneConfig, LlavaNextAdapter, Qwen25VLAdapter


def _resolve_model_name(backbone_cfg: dict[str, Any], key: str, default_name: str) -> str:
    direct = backbone_cfg.get("model_name")
    if isinstance(direct, str) and direct.strip():
        return direct
    registry = backbone_cfg.get("registry")
    if isinstance(registry, dict):
        item = registry.get(key)
        if isinstance(item, dict):
            name = item.get("model_name")
            if isinstance(name, str) and name.strip():
                return name
    return default_name


def _resolve_family_vocab_overrides(backbone_cfg: dict[str, Any]) -> dict[str, list[str]] | None:
    resolved: dict[str, list[str]] = {}

    vocab_path = backbone_cfg.get("family_vocab_path")
    if isinstance(vocab_path, str) and vocab_path.strip():
        loaded = load_family_vocabs(Path(vocab_path))
        resolved.update(family_vocab_jsonable(loaded))

    inline = backbone_cfg.get("family_vocab_overrides")
    if isinstance(inline, dict):
        for family, values in inline.items():
            if isinstance(values, list):
                resolved[str(family).lower()] = [str(v) for v in values]

    return resolved or None


def create_backbone(backbone_cfg: dict[str, Any]):
    name = str(backbone_cfg.get("name", "qwen2_5_vl_7b"))

    config = BackboneConfig(
        hidden_size=int(backbone_cfg.get("hidden_size", 128)),
        seq_len=int(backbone_cfg.get("seq_len", 32)),
        max_new_tokens=int(backbone_cfg.get("max_new_tokens", 10)),
        count_min=int(backbone_cfg.get("count_min", 0)),
        count_max=int(backbone_cfg.get("count_max", 20)),
        color_vocab=tuple(backbone_cfg.get("color_vocab", ())),
        family_vocab_overrides=_resolve_family_vocab_overrides(backbone_cfg),
        force_fallback_distribution=bool(backbone_cfg.get("force_fallback_distribution", False)),
    )

    if name == "qwen2_5_vl_7b":
        return Qwen25VLAdapter(
            model_name=_resolve_model_name(backbone_cfg, "qwen2_5_vl_7b", "Qwen/Qwen2.5-VL-7B-Instruct"),
            config=config,
            device=str(backbone_cfg.get("device", "auto")),
            torch_dtype=str(backbone_cfg.get("torch_dtype", "auto")),
            cache_results=bool(backbone_cfg.get("cache_results", True)),
            cache_max_entries=(
                int(backbone_cfg["cache_max_entries"])
                if backbone_cfg.get("cache_max_entries") is not None
                else None
            ),
        )

    if name == "llava_next_8b":
        return LlavaNextAdapter(
            model_name=_resolve_model_name(backbone_cfg, "llava_next_8b", "llava-hf/llava-v1.6-8b")
        )

    raise ValueError(f"Unknown backbone name: {name}")
