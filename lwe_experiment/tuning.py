from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ProposedFinalTuningResult:
    selected_params: dict[str, Any]
    notes: str


def tune_proposed_final_params(
    *,
    config: Any,
    seed: int,
    generator_params: dict[str, Any],
) -> ProposedFinalTuningResult:
    return ProposedFinalTuningResult(
        selected_params=dict(generator_params),
        notes="tuning=disabled_for_final_coordinatewise_model",
    )
