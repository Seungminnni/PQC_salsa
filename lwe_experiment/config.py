from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TUNABLE_PROPOSED_FACTORS: set[str] = {
    "beta_min",
    "beta_max",
    "boundary_base",
    "boundary_jitter",
    "clipping_bound",
    "setup_eval_samples",
    "family_count",
}


@dataclass(slots=True)
class ModelConfig:
    name: str
    generator_type: str
    overrides: dict[str, Any] = field(default_factory=dict)

    def resolved_params(
        self,
        baseline_params: dict[str, Any],
        proposed_base_params: dict[str, Any],
    ) -> dict[str, Any]:
        if self.generator_type == "baseline":
            params = dict(baseline_params)
            params.setdefault("num_functions", 0)
            params.setdefault("function_family", "baseline_discrete_gaussian")
            params.setdefault("region_count", 0)
            return params
        params = dict(proposed_base_params)
        params.update(self.overrides)
        return params


@dataclass(slots=True)
class ExperimentConfig:
    global_params: dict[str, Any]
    baseline_params: dict[str, Any]
    proposed_base_params: dict[str, Any]
    models: list[ModelConfig]

    def validate(self) -> None:
        baseline_names = [model.name for model in self.models if model.generator_type == "baseline"]
        if baseline_names != ["baseline_lwe"]:
            raise ValueError("Exactly one baseline model named 'baseline_lwe' is required.")
        proposed_names = [model.name for model in self.models if model.generator_type == "proposed"]
        if proposed_names != ["proposed_final"]:
            raise ValueError("Exactly one proposed model named 'proposed_final' is required.")

        seen_names: set[str] = set()
        for model in self.models:
            if model.name in seen_names:
                raise ValueError(f"Duplicate model name: {model.name}")
            seen_names.add(model.name)

            if model.generator_type not in {"baseline", "proposed"}:
                raise ValueError(f"Unsupported generator type for {model.name}: {model.generator_type}")

            if model.generator_type == "proposed":
                illegal_keys = set(model.overrides) - TUNABLE_PROPOSED_FACTORS
                if illegal_keys:
                    raise ValueError(f"{model.name} changes non-tunable factors: {sorted(illegal_keys)}")
                if model.name != "proposed_final":
                    raise ValueError("Only the final proposed model 'proposed_final' is supported.")

    @property
    def run_seeds(self) -> list[int]:
        seeds = self.global_params.get("run_seeds")
        if seeds:
            return [int(seed) for seed in seeds]

        runs_per_model = int(self.global_params.get("runs_per_model", 3))
        base_seed = int(self.global_params.get("base_seed", 1337))
        return [base_seed + offset * 97 for offset in range(runs_per_model)]


def _default_models() -> list[ModelConfig]:
    return [
        ModelConfig(name="baseline_lwe", generator_type="baseline"),
        ModelConfig(name="proposed_final", generator_type="proposed"),
    ]


def build_default_config(quick: bool = False) -> ExperimentConfig:
    global_params = {
        "n": 8,
        "m": 32,
        "q": 97,
        "noise_sigma": 1.0,
        "num_samples": 72 if quick else 160,
        "train_fraction": 0.7,
        "run_seeds": [2027, 2041] if quick else [2027, 2041, 2053],
        "results_dir": "independent_coordinate_results",
        "figures_dir": "independent_coordinate_figures",
        "tsne_max_points": 250,
        "pairwise_distance_max_points": 250,
        "num_clusters": 4,
        "probe_hidden_layers": [64, 32],
        "probe_max_iter": 350 if quick else 450,
        "recovery_refinement_passes": 4,
        "exhaustive_recovery_limit": 0,
    }
    baseline_params = {
        "secret_sigma": 1.0,
        "clipping_bound": 2,
        "quantization_step": 1,
    }
    proposed_base_params = {
        "num_functions": 1,
        "function_family": "coordinatewise_region_scalar",
        "region_count": 4,
        "clipping_bound": 4,
        "beta_min": 0.08,
        "beta_max": 0.16,
        "boundary_base": 0.92,
        "boundary_jitter": 0.22,
        "setup_eval_samples": 2048 if quick else 4096,
        "family_count": 5,
        "tuning_enabled": False,
    }
    config = ExperimentConfig(
        global_params=global_params,
        baseline_params=baseline_params,
        proposed_base_params=proposed_base_params,
        models=_default_models(),
    )
    config.validate()
    return config


def load_experiment_config(config_path: str | Path | None, quick: bool = False) -> ExperimentConfig:
    if config_path is None:
        return build_default_config(quick=quick)

    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    models = [
        ModelConfig(
            name=item["name"],
            generator_type=item["generator_type"],
            overrides=item.get("overrides", {}),
        )
        for item in payload["models"]
    ]
    config = ExperimentConfig(
        global_params=payload["global_params"],
        baseline_params=payload["baseline_params"],
        proposed_base_params=payload["proposed_base_params"],
        models=models,
    )
    config.validate()
    if quick:
        config.global_params["num_samples"] = min(int(config.global_params["num_samples"]), 72)
        config.global_params["run_seeds"] = config.run_seeds[:2]
    return config
