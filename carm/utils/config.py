from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")

    parent_name = data.pop("extends", None)
    if parent_name:
        parent = path.parent / f"{parent_name}.yaml"
        if not parent.exists():
            raise FileNotFoundError(f"Extended config not found: {parent}")
        base = load_yaml_config(parent)
        return _deep_merge(base, data)
    return data
