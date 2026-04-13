from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt
from typing import Any, Protocol

import numpy as np

from .utils import json_dumps, seeded_rng


@dataclass(slots=True)
class SecretSample:
    secret: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class SecretGenerator(Protocol):
    def sample_with_metadata(self, sample_index: int) -> SecretSample: ...

    def get_setup_diagnostics(self) -> dict[str, Any]: ...


def _entropy_base2(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    probabilities = counts[counts > 0.0] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float).ravel()
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _family_name(family_id: int) -> str:
    names = {
        0: "tanh",
        1: "cubic_polynomial",
        2: "sinusoidal",
        3: "clipped_quadratic",
        4: "random_piecewise_polynomial",
    }
    return names[int(family_id)]


def _positive_transform_input(magnitude: float, threshold: float, region_slot: int) -> float:
    if region_slot == 0:
        return min(magnitude / max(threshold, 1e-6), 1.75)
    tail = max(magnitude - threshold, 0.0)
    return min(1.0 + tail, 2.25)


@dataclass(slots=True)
class BaselineSecretGenerator:
    n: int
    base_seed: int
    secret_sigma: float
    clipping_bound: int

    def sample_with_metadata(self, sample_index: int) -> SecretSample:
        rng = seeded_rng(self.base_seed, f"secret:{sample_index}")
        raw = rng.normal(loc=0.0, scale=self.secret_sigma, size=self.n)
        secret = np.clip(np.rint(raw), -self.clipping_bound, self.clipping_bound).astype(int)
        return SecretSample(secret=secret, metadata={})

    def get_setup_diagnostics(self) -> dict[str, Any]:
        return {"generator_variant": "baseline_discrete_gaussian"}


@dataclass(slots=True)
class ProposedFinalSecretGenerator:
    n: int
    base_seed: int
    region_count: int
    clipping_bound: int
    beta_min: float
    beta_max: float
    boundary_base: float
    boundary_jitter: float
    setup_eval_samples: int
    family_count: int
    threshold_values: np.ndarray = field(init=False, repr=False)
    family_ids: np.ndarray = field(init=False, repr=False)
    coordinate_betas: np.ndarray = field(init=False, repr=False)
    coordinate_bounds: np.ndarray = field(init=False, repr=False)
    region_amplitudes: np.ndarray = field(init=False, repr=False)
    region_biases: np.ndarray = field(init=False, repr=False)
    region_linear_terms: np.ndarray = field(init=False, repr=False)
    region_frequencies: np.ndarray = field(init=False, repr=False)
    region_quadratic_terms: np.ndarray = field(init=False, repr=False)
    region_cubic_terms: np.ndarray = field(init=False, repr=False)
    region_caps: np.ndarray = field(init=False, repr=False)
    setup_diagnostics: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.region_count != 4:
            raise ValueError("proposed_final requires region_count = 4.")
        if self.clipping_bound != 4:
            raise ValueError("proposed_final requires clipping_bound = 4 for coordinate-wise bounds up to 4.")
        if self.beta_min < 0.0 or self.beta_max <= 0.0 or self.beta_min > self.beta_max:
            raise ValueError("proposed_final requires 0 <= beta_min <= beta_max.")
        if self.beta_max > 0.2:
            raise ValueError("proposed_final requires beta_max <= 0.2.")
        if self.boundary_base <= 0.0:
            raise ValueError("proposed_final requires boundary_base > 0.")
        if self.boundary_jitter < 0.0:
            raise ValueError("proposed_final requires boundary_jitter >= 0.")
        if self.setup_eval_samples <= 0:
            raise ValueError("proposed_final requires setup_eval_samples > 0.")
        if self.family_count != 5:
            raise ValueError("proposed_final requires family_count = 5.")

        param_rng = seeded_rng(self.base_seed, "proposed_final:coordinate_params")
        self.threshold_values = np.clip(
            self.boundary_base + param_rng.uniform(-self.boundary_jitter, self.boundary_jitter, size=self.n),
            0.42,
            1.45,
        )
        family_pattern = np.tile(np.arange(self.family_count, dtype=int), int(np.ceil(self.n / self.family_count)))[: self.n]
        self.family_ids = param_rng.permutation(family_pattern)
        self.coordinate_betas = param_rng.uniform(self.beta_min, self.beta_max, size=self.n)
        self.coordinate_bounds = param_rng.choice(np.array([2, 3, 4], dtype=int), size=self.n, p=[0.10, 0.55, 0.35])

        # Positive-region parameters: slot 0 is inner positive, slot 1 is outer positive.
        self.region_amplitudes = param_rng.uniform(0.82, 1.28, size=(self.n, 2))
        self.region_biases = param_rng.uniform(0.01, 0.18, size=(self.n, 2))
        self.region_linear_terms = param_rng.uniform(0.28, 0.78, size=(self.n, 2))
        self.region_frequencies = param_rng.uniform(0.90, 2.40, size=(self.n, 2))
        self.region_quadratic_terms = param_rng.uniform(0.08, 0.42, size=(self.n, 2))
        self.region_cubic_terms = param_rng.uniform(0.05, 0.22, size=(self.n, 2))
        self.region_caps = param_rng.uniform(1.00, 1.70, size=(self.n, 2))
        self.setup_diagnostics = self._estimate_setup_diagnostics()

    def _positive_response(self, coordinate_index: int, region_slot: int, magnitude: float) -> float:
        threshold = float(self.threshold_values[coordinate_index])
        transformed = _positive_transform_input(magnitude=magnitude, threshold=threshold, region_slot=region_slot)
        amplitude = float(self.region_amplitudes[coordinate_index, region_slot])
        bias = float(self.region_biases[coordinate_index, region_slot])
        linear = float(self.region_linear_terms[coordinate_index, region_slot])
        frequency = float(self.region_frequencies[coordinate_index, region_slot])
        quadratic = float(self.region_quadratic_terms[coordinate_index, region_slot])
        cubic = float(self.region_cubic_terms[coordinate_index, region_slot])
        cap = float(self.region_caps[coordinate_index, region_slot])
        family_id = int(self.family_ids[coordinate_index])

        if family_id == 0:
            base_value = linear * transformed + 0.85 * np.tanh(frequency * transformed)
        elif family_id == 1:
            clipped = min(transformed, 1.55)
            base_value = linear * clipped + cubic * (clipped**3)
        elif family_id == 2:
            base_value = linear * transformed + quadratic * np.sin(frequency * transformed)
        elif family_id == 3:
            base_value = linear * transformed + quadratic * min(transformed**2, cap)
        else:
            clipped = min(transformed, 1.45 + 0.20 * region_slot)
            base_value = linear * clipped + quadratic * (clipped**2) + cubic * (clipped**3)

        return float(amplitude * base_value + bias)

    def _coordinate_response(self, coordinate_index: int, value: float) -> tuple[float, int]:
        threshold = float(self.threshold_values[coordinate_index])
        magnitude = abs(value)
        if value <= -threshold:
            region_index = 0
            region_slot = 1
        elif value <= 0.0:
            region_index = 1
            region_slot = 0
        elif value <= threshold:
            region_index = 2
            region_slot = 0
        else:
            region_index = 3
            region_slot = 1
        response = self._positive_response(coordinate_index, region_slot, magnitude)
        return (-response if value < 0.0 else response), region_index

    def _sample_secret_vector(self, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
        z_values = rng.normal(loc=0.0, scale=1.0, size=self.n)
        dither = rng.integers(-1, 2, size=self.n)
        regions = np.zeros(self.n, dtype=int)
        raw_values = np.zeros(self.n, dtype=float)
        pre_quantized = np.zeros(self.n, dtype=float)
        secret = np.zeros(self.n, dtype=int)
        clipped_mask = np.zeros(self.n, dtype=bool)

        for coordinate_index in range(self.n):
            response, region_index = self._coordinate_response(coordinate_index, float(z_values[coordinate_index]))
            raw_values[coordinate_index] = response
            regions[coordinate_index] = region_index
            pre_quantized_value = response + self.coordinate_betas[coordinate_index] * float(dither[coordinate_index])
            pre_quantized[coordinate_index] = pre_quantized_value
            rounded = int(np.rint(pre_quantized_value))
            bound = int(self.coordinate_bounds[coordinate_index])
            clipped_mask[coordinate_index] = abs(rounded) > bound
            secret[coordinate_index] = int(np.clip(rounded, -bound, bound))

        metadata = {
            "region_counts_vector": np.bincount(regions, minlength=self.region_count),
            "avg_abs_before_quantization": float(np.mean(np.abs(pre_quantized))),
            "avg_abs_after_quantization": float(np.mean(np.abs(secret))),
            "fraction_clipped_coordinates": float(np.mean(clipped_mask.astype(float))),
            "occupied_quantization_levels": int(np.unique(secret).size),
        }
        return secret, metadata

    def _estimate_setup_diagnostics(self) -> dict[str, Any]:
        rng = seeded_rng(self.base_seed, "proposed_final:setup_eval")
        setup_secrets = np.zeros((self.setup_eval_samples, self.n), dtype=int)
        region_counts = np.zeros(self.region_count, dtype=int)
        clipped_fraction_total = 0.0
        occupied_levels_per_sample: list[int] = []
        for sample_index in range(self.setup_eval_samples):
            secret, metadata = self._sample_secret_vector(rng)
            setup_secrets[sample_index] = secret
            region_counts += np.asarray(metadata["region_counts_vector"], dtype=int)
            clipped_fraction_total += float(metadata["fraction_clipped_coordinates"])
            occupied_levels_per_sample.append(int(metadata["occupied_quantization_levels"]))

        per_coordinate_mean = np.mean(setup_secrets, axis=0)
        per_coordinate_variance = np.var(setup_secrets, axis=0)
        per_coordinate_entropy: list[float] = []
        occupied_levels_per_coordinate: list[int] = []
        symmetry_gap_by_level: dict[str, float] = {}
        for coordinate_index in range(self.n):
            bound = int(self.coordinate_bounds[coordinate_index])
            levels = np.arange(-bound, bound + 1)
            counts = np.array([np.sum(setup_secrets[:, coordinate_index] == level) for level in levels], dtype=float)
            per_coordinate_entropy.append(_entropy_base2(counts))
            occupied_levels_per_coordinate.append(int(np.sum(counts > 0.0)))
        for level in range(1, int(np.max(self.coordinate_bounds)) + 1):
            positive = np.mean(setup_secrets == level)
            negative = np.mean(setup_secrets == -level)
            symmetry_gap_by_level[str(level)] = float(abs(positive - negative))

        expected_region_counts = np.zeros(self.region_count, dtype=float)
        for threshold in self.threshold_values:
            outer_prob = 1.0 - _normal_cdf(float(threshold))
            inner_prob = _normal_cdf(float(threshold)) - 0.5
            expected_region_counts += np.array([outer_prob, inner_prob, inner_prob, outer_prob]) * self.setup_eval_samples

        coefficient_a = self.region_amplitudes
        coefficient_b = self.region_biases
        return {
            "generator_variant": "coordinatewise_independent_region_generator_heterogeneous",
            "coordinates_generated_independently": True,
            "setup_region_occupancy_counts_json": json_dumps(region_counts.tolist()),
            "setup_region_occupancy_entropy": float(_entropy_base2(region_counts)),
            "initial_region_balance_counts_json": json_dumps(np.rint(expected_region_counts).astype(int).tolist()),
            "initial_region_balance_entropy": float(_entropy_base2(expected_region_counts)),
            "region_lattice_counts_json": None,
            "region_lattice_entropy": None,
            "pre_normalization_cov_diag_stats": {"mean": None, "std": None, "min": None, "max": None},
            "post_normalization_cov_diag_stats": {"mean": None, "std": None, "min": None, "max": None},
            "pre_whitening_cov_diag_stats": {"mean": None, "std": None, "min": None, "max": None},
            "post_whitening_cov_diag_stats": {"mean": None, "std": None, "min": None, "max": None},
            "whitening_condition_number": None,
            "orthogonal_mixing_seed": None,
            "structure_seed_offset": None,
            "orthogonal_seed_offset": None,
            "signed_permutation_seed": None,
            "quantization_mode": "coordinatewise_round_clip_variable_bounds",
            "noise_injection_mode": "coordinatewise_ternary_dither_variable_beta",
            "alpha": None,
            "beta": float(np.mean(self.coordinate_betas)),
            "gamma": None,
            "map_scale": None,
            "offset_scale": None,
            "within_region_sigma": None,
            "uniform_region_mix": None,
            "boundary_focus_mix": None,
            "boundary_temperature": None,
            "shell_balance_mix": None,
            "region_sampling_mode": "independent_scalar_intervals",
            "implied_prior_effective_rank": None,
            "implied_prior_pairwise_corr_mean_abs": None,
            "implied_prior_abs_mean": None,
            "implied_prior_boundary_margin_mean": None,
            "coordinate_region_boundaries_json": json_dumps(np.round(self.threshold_values, 6).tolist()),
            "function_family_per_coordinate_json": json_dumps([_family_name(family_id) for family_id in self.family_ids]),
            "coefficient_a_summary_json": json_dumps(_summary_stats(coefficient_a)),
            "coefficient_b_summary_json": json_dumps(_summary_stats(coefficient_b)),
            "coordinate_beta_json": json_dumps(np.round(self.coordinate_betas, 6).tolist()),
            "coordinate_clipping_bounds_json": json_dumps(self.coordinate_bounds.astype(int).tolist()),
            "empirical_coordinate_mean_json": json_dumps(np.round(per_coordinate_mean, 6).tolist()),
            "empirical_coordinate_variance_json": json_dumps(np.round(per_coordinate_variance, 6).tolist()),
            "empirical_coordinate_entropy_json": json_dumps(np.round(np.asarray(per_coordinate_entropy), 6).tolist()),
            "setup_fraction_clipped_coordinates": float(clipped_fraction_total / self.setup_eval_samples),
            "average_occupied_quantization_levels": float(np.mean(occupied_levels_per_coordinate)),
            "samplewise_occupied_quantization_levels_mean": float(np.mean(occupied_levels_per_sample)),
            "symmetry_bin_gap_json": json_dumps(symmetry_gap_by_level),
            "setup_coordinate_variance_trace": float(np.trace(np.cov(setup_secrets.astype(float), rowvar=False))),
            "setup_total_coordinate_draws": int(self.setup_eval_samples * self.n),
        }

    def sample_with_metadata(self, sample_index: int) -> SecretSample:
        rng = seeded_rng(self.base_seed, f"secret:{sample_index}")
        secret, metadata = self._sample_secret_vector(rng)
        return SecretSample(secret=secret, metadata=metadata)

    def get_setup_diagnostics(self) -> dict[str, Any]:
        pre_stats = self.setup_diagnostics["pre_whitening_cov_diag_stats"]
        post_stats = self.setup_diagnostics["post_whitening_cov_diag_stats"]
        return {
            "generator_variant": self.setup_diagnostics["generator_variant"],
            "coordinates_generated_independently": self.setup_diagnostics["coordinates_generated_independently"],
            "setup_region_occupancy_counts_json": self.setup_diagnostics["setup_region_occupancy_counts_json"],
            "setup_region_occupancy_entropy": float(self.setup_diagnostics["setup_region_occupancy_entropy"]),
            "initial_region_balance_counts_json": self.setup_diagnostics["initial_region_balance_counts_json"],
            "initial_region_balance_entropy": float(self.setup_diagnostics["initial_region_balance_entropy"]),
            "region_lattice_counts_json": self.setup_diagnostics["region_lattice_counts_json"],
            "region_lattice_entropy": _optional_float(self.setup_diagnostics["region_lattice_entropy"]),
            "pre_normalization_cov_diag_mean": _optional_float(pre_stats["mean"]),
            "pre_normalization_cov_diag_std": _optional_float(pre_stats["std"]),
            "pre_normalization_cov_diag_min": _optional_float(pre_stats["min"]),
            "pre_normalization_cov_diag_max": _optional_float(pre_stats["max"]),
            "post_normalization_cov_diag_mean": _optional_float(post_stats["mean"]),
            "post_normalization_cov_diag_std": _optional_float(post_stats["std"]),
            "post_normalization_cov_diag_min": _optional_float(post_stats["min"]),
            "post_normalization_cov_diag_max": _optional_float(post_stats["max"]),
            "pre_whitening_cov_diag_mean": _optional_float(pre_stats["mean"]),
            "pre_whitening_cov_diag_std": _optional_float(pre_stats["std"]),
            "pre_whitening_cov_diag_min": _optional_float(pre_stats["min"]),
            "pre_whitening_cov_diag_max": _optional_float(pre_stats["max"]),
            "post_whitening_cov_diag_mean": _optional_float(post_stats["mean"]),
            "post_whitening_cov_diag_std": _optional_float(post_stats["std"]),
            "post_whitening_cov_diag_min": _optional_float(post_stats["min"]),
            "post_whitening_cov_diag_max": _optional_float(post_stats["max"]),
            "whitening_condition_number": _optional_float(self.setup_diagnostics["whitening_condition_number"]),
            "orthogonal_mixing_seed": self.setup_diagnostics["orthogonal_mixing_seed"],
            "structure_seed_offset": self.setup_diagnostics["structure_seed_offset"],
            "orthogonal_seed_offset": self.setup_diagnostics["orthogonal_seed_offset"],
            "signed_permutation_seed": self.setup_diagnostics["signed_permutation_seed"],
            "alpha": _optional_float(self.setup_diagnostics["alpha"]),
            "beta": _optional_float(self.setup_diagnostics["beta"]),
            "gamma": _optional_float(self.setup_diagnostics["gamma"]),
            "map_scale": _optional_float(self.setup_diagnostics["map_scale"]),
            "offset_scale": _optional_float(self.setup_diagnostics["offset_scale"]),
            "quantization_mode": self.setup_diagnostics["quantization_mode"],
            "noise_injection_mode": self.setup_diagnostics["noise_injection_mode"],
            "within_region_sigma": _optional_float(self.setup_diagnostics["within_region_sigma"]),
            "uniform_region_mix": _optional_float(self.setup_diagnostics["uniform_region_mix"]),
            "boundary_focus_mix": _optional_float(self.setup_diagnostics["boundary_focus_mix"]),
            "boundary_temperature": _optional_float(self.setup_diagnostics["boundary_temperature"]),
            "shell_balance_mix": _optional_float(self.setup_diagnostics["shell_balance_mix"]),
            "region_sampling_mode": self.setup_diagnostics["region_sampling_mode"],
            "implied_prior_effective_rank": _optional_float(self.setup_diagnostics["implied_prior_effective_rank"]),
            "implied_prior_pairwise_corr_mean_abs": _optional_float(
                self.setup_diagnostics["implied_prior_pairwise_corr_mean_abs"]
            ),
            "implied_prior_abs_mean": _optional_float(self.setup_diagnostics["implied_prior_abs_mean"]),
            "implied_prior_boundary_margin_mean": _optional_float(
                self.setup_diagnostics["implied_prior_boundary_margin_mean"]
            ),
            "coordinate_region_boundaries_json": self.setup_diagnostics["coordinate_region_boundaries_json"],
            "function_family_per_coordinate_json": self.setup_diagnostics["function_family_per_coordinate_json"],
            "coefficient_a_summary_json": self.setup_diagnostics["coefficient_a_summary_json"],
            "coefficient_b_summary_json": self.setup_diagnostics["coefficient_b_summary_json"],
            "coordinate_beta_json": self.setup_diagnostics["coordinate_beta_json"],
            "coordinate_clipping_bounds_json": self.setup_diagnostics["coordinate_clipping_bounds_json"],
            "empirical_coordinate_mean_json": self.setup_diagnostics["empirical_coordinate_mean_json"],
            "empirical_coordinate_variance_json": self.setup_diagnostics["empirical_coordinate_variance_json"],
            "empirical_coordinate_entropy_json": self.setup_diagnostics["empirical_coordinate_entropy_json"],
            "setup_fraction_clipped_coordinates": float(self.setup_diagnostics["setup_fraction_clipped_coordinates"]),
            "average_occupied_quantization_levels": float(self.setup_diagnostics["average_occupied_quantization_levels"]),
            "samplewise_occupied_quantization_levels_mean": float(
                self.setup_diagnostics["samplewise_occupied_quantization_levels_mean"]
            ),
            "symmetry_bin_gap_json": self.setup_diagnostics["symmetry_bin_gap_json"],
            "setup_coordinate_variance_trace": float(self.setup_diagnostics["setup_coordinate_variance_trace"]),
            "setup_total_coordinate_draws": int(self.setup_diagnostics["setup_total_coordinate_draws"]),
        }


def build_secret_generator(
    model_name: str,
    generator_type: str,
    n: int,
    run_seed: int,
    params: dict[str, int | float | bool | str],
) -> SecretGenerator:
    base_seed = int(run_seed)
    if generator_type == "baseline":
        return BaselineSecretGenerator(
            n=n,
            base_seed=base_seed,
            secret_sigma=float(params["secret_sigma"]),
            clipping_bound=int(params["clipping_bound"]),
        )
    if generator_type == "proposed":
        if model_name != "proposed_final":
            raise ValueError("Only the final proposed model 'proposed_final' is supported in this framework.")
        return ProposedFinalSecretGenerator(
            n=n,
            base_seed=base_seed,
            region_count=int(params.get("region_count", 4)),
            clipping_bound=int(params.get("clipping_bound", 4)),
            beta_min=float(params.get("beta_min", 0.05)),
            beta_max=float(params.get("beta_max", 0.15)),
            boundary_base=float(params.get("boundary_base", 0.95)),
            boundary_jitter=float(params.get("boundary_jitter", 0.20)),
            setup_eval_samples=int(params.get("setup_eval_samples", 4096)),
            family_count=int(params.get("family_count", 5)),
        )
    raise ValueError(f"Unsupported generator type for {model_name}: {generator_type}")
