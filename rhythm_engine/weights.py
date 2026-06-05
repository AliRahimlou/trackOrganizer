from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


def _finite_weight(value: Any, default: float = 1.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out) or out <= 0.0:
        return float(default)
    return float(out)


@lru_cache(maxsize=16)
def load_provider_weights(path: str) -> dict[str, float]:
    p = Path(path).expanduser()
    with open(p, "r", encoding="utf-8") as fh:
        payload: Any = json.load(fh)
    if isinstance(payload, Mapping) and isinstance(payload.get("provider_weights"), Mapping):
        payload = payload.get("provider_weights")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Provider weight payload must be a mapping: {path}")
    return {str(key): _finite_weight(value) for key, value in payload.items()}


def provider_weight_for_name(provider: str, weights: Mapping[str, float] | None) -> float:
    if not weights:
        return 1.0
    name = str(provider)
    if name in weights:
        return _finite_weight(weights[name])
    parts = name.split(":")
    for end in range(len(parts), 0, -1):
        prefix = ":".join(parts[:end])
        if prefix in weights:
            return _finite_weight(weights[prefix])
    for part in reversed(parts):
        if part in weights:
            return _finite_weight(weights[part])
    return 1.0


def weights_from_config(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    return load_provider_weights(str(path))
