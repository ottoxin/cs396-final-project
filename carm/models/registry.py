from __future__ import annotations

from typing import Any

from carm.models.backbone import BackboneConfig, LlavaNextAdapter, MockFrozenBackbone, Qwen25VLAdapter


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


def create_backbone(backbone_cfg: dict[str, Any]):
    name = str(backbone_cfg.get("name", "mock_frozen_backbone"))

    config = BackboneConfig(
        hidden_size=int(backbone_cfg.get("hidden_size", 128)),
        seq_len=int(backbone_cfg.get("seq_len", 32)),
    )

    if name == "mock_frozen_backbone":
        return MockFrozenBackbone(config)

    if name == "qwen2_5_vl_7b":
        return Qwen25VLAdapter(
            model_name=_resolve_model_name(backbone_cfg, "qwen2_5_vl_7b", "Qwen/Qwen2.5-VL-7B-Instruct"),
            config=config,
            device=str(backbone_cfg.get("device", "auto")),
            torch_dtype=str(backbone_cfg.get("torch_dtype", "auto")),
            cache_results=bool(backbone_cfg.get("cache_results", True)),
        )

    if name == "llava_next_8b":
        return LlavaNextAdapter(
            model_name=_resolve_model_name(backbone_cfg, "llava_next_8b", "llava-hf/llava-v1.6-8b")
        )

    raise ValueError(f"Unknown backbone name: {name}")
