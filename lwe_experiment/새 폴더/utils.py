from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_int_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def seeded_rng(base_seed: int, label: str) -> np.random.Generator:
    return np.random.default_rng(stable_int_seed(base_seed, label))


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (int, np.integer)):
        return float(value)
    return value


def sanitize_run_name(run_name: str) -> str:
    return run_name.replace("*", "_")


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True)


def centered_mod(values: np.ndarray, q: int) -> np.ndarray:
    return ((values + (q // 2)) % q) - (q // 2)


def discrete_gaussian(
    rng: np.random.Generator,
    size: int | tuple[int, ...],
    sigma: float,
) -> np.ndarray:
    return np.rint(rng.normal(loc=0.0, scale=sigma, size=size)).astype(int)
