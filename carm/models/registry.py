from __future__ import annotations

from typing import Any

from carm.models.backbone import BackboneConfig, LlavaNextAdapter, MockFrozenBackbone, Qwen25VLAdapter


def create_backbone(backbone_cfg: dict[str, Any]):
    name = str(backbone_cfg.get("name", "mock_frozen_backbone"))

    if name == "mock_frozen_backbone":
        return MockFrozenBackbone(
            BackboneConfig(
                hidden_size=int(backbone_cfg.get("hidden_size", 128)),
                seq_len=int(backbone_cfg.get("seq_len", 32)),
            )
        )

    if name == "qwen2_5_vl_7b":
        return Qwen25VLAdapter(model_name=backbone_cfg.get("model_name"))

    if name == "llava_next_8b":
        return LlavaNextAdapter(model_name=backbone_cfg.get("model_name"))

    raise ValueError(f"Unknown backbone name: {name}")
