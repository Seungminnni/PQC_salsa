import json
import numpy as np


def json_dumps(value):
    return json.dumps(value, ensure_ascii=False)


def seeded_rng(base_seed: int, namespace: str) -> np.random.Generator:
    seed = abs(hash((int(base_seed), namespace))) % (2**32)
    return np.random.default_rng(seed)
