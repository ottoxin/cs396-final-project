from __future__ import annotations

from typing import Any

import torch


def _normalize_explicit_device(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, torch.device):
        return str(raw)

    text = str(raw).strip().lower()
    if not text or text == "auto":
        return None
    return str(torch.device(text))


def _normalize_backbone_device(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, torch.device):
        return str(raw)

    text = str(raw).strip().lower()
    if not text or text == "auto":
        return None
    return str(torch.device(text))


def resolve_carm_device(training_device: Any, backbone: Any) -> str:
    explicit = _normalize_explicit_device(training_device)
    if explicit is not None:
        return explicit

    for candidate in (
        getattr(backbone, "device", None),
        getattr(getattr(backbone, "config", None), "device", None),
    ):
        resolved = _normalize_backbone_device(candidate)
        if resolved is not None:
            return resolved

    return "cpu"
