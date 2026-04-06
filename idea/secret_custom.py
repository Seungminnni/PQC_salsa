from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, floor, sqrt
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


STYLE_NAMES = {
    0: "centered_smooth",
    1: "flat_like",
    2: "sparse_like",
    3: "bimodal_like",
    4: "heavy_tail_like",
    5: "region_piecewise_like",
}

FAMILY_NAMES = {
    0: "odd_cubic",
    1: "tanh_based",
    2: "odd_clipped_quadratic",
    3: "sinusoidal_perturbation",
    4: "piecewise_polynomial",
    5: "smooth_staircase_like",
}

SCORE_NAMES = {
    0: "gaussian_rbf",
    1: "laplace_rbf",
    2: "cauchy_log",
    3: "triangular_window",
    4: "polynomial_bump",
    5: "cosine_window",
}

QUANTIZER_NAMES = {
    0: "stochastic_rounding",
    1: "random_threshold",
    2: "dithered_floor",
    3: "random_deadzone_round",
}


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


def _style_name(style_id: int) -> str:
    return STYLE_NAMES[int(style_id)]


def _family_name(family_id: int) -> str:
    return FAMILY_NAMES[int(family_id)]


def _score_name(score_id: int) -> str:
    return SCORE_NAMES[int(score_id)]


def _quantizer_name(quantizer_id: int) -> str:
    return QUANTIZER_NAMES[int(quantizer_id)]


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    # generate.py sets np.seterr(all='raise'); guard numerics here.
    shifted = np.clip(shifted, -60.0, 0.0)
    with np.errstate(under='ignore', over='raise', invalid='raise'):
        weights = np.exp(shifted)
    denom = np.sum(weights)
    if denom <= 0.0 or not np.isfinite(denom):
        return np.full_like(weights, 1.0 / len(weights), dtype=float)
    return weights / denom


def _balanced_assignment(
    rng: np.random.Generator,
    size: int,
    num_categories: int,
) -> np.ndarray:
    repeats = int(np.ceil(size / max(num_categories, 1)))
    tiled = np.tile(np.arange(num_categories, dtype=int), repeats)[:size]
    return rng.permutation(tiled).astype(int)


def _priority_assignment(
    rng: np.random.Generator,
    size: int,
    base_categories: np.ndarray,
    priority_extras: np.ndarray,
) -> np.ndarray:
    if size <= base_categories.size:
        return rng.permutation(base_categories)[:size].astype(int)
    extra_count = size - base_categories.size
    repeats = int(np.ceil(extra_count / max(priority_extras.size, 1)))
    extras = np.tile(priority_extras, repeats)[:extra_count]
    combined = np.concatenate([base_categories, extras])
    return rng.permutation(combined).astype(int)


def _positive_transform_input(magnitude: float, threshold: float, region_slot: int) -> float:
    threshold = max(threshold, 1e-6)
    if region_slot == 0:
        normalized = magnitude / threshold
        return float(min(normalized, 1.85))
    tail = max(magnitude - threshold, 0.0)
    return float(min(1.0 + 1.25 * np.log1p(1.35 * tail), 2.75))


def _region_boundary_margin(value: float, threshold: float) -> float:
    abs_value = abs(value)
    return float(min(abs_value, abs(abs_value - threshold)))


def _slot_transform(magnitude: float, center: float, width: float, slot_index: int) -> float:
    normalized = magnitude / max(center + 0.35 * width, 1e-6)
    local_offset = abs(magnitude - center) / max(width, 1e-6)
    value = 0.72 * normalized + 0.45 * local_offset + 0.18 * slot_index
    return float(min(max(value, 0.0), 3.10))


def _build_positive_geometry(
    threshold: float,
    positive_region_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    start = 0.24 * threshold + 0.08 * (positive_region_count > 2)
    stop = threshold * (1.10 + 0.34 * max(positive_region_count - 1, 0))
    centers = np.linspace(start, stop, num=positive_region_count)
    centers += rng.uniform(-0.05, 0.05, size=positive_region_count)
    centers = np.maximum.accumulate(np.clip(centers, 0.10, None))
    widths = np.zeros(positive_region_count, dtype=float)
    for index in range(positive_region_count):
        if positive_region_count == 1:
            local_gap = threshold + 0.45
        elif index == 0:
            local_gap = centers[1] - centers[0]
        elif index == positive_region_count - 1:
            local_gap = centers[index] - centers[index - 1]
        else:
            local_gap = 0.5 * ((centers[index] - centers[index - 1]) + (centers[index + 1] - centers[index]))
        widths[index] = max(0.24, 0.56 * local_gap + 0.18)
    positive_boundaries = [0.0]
    if positive_region_count > 1:
        positive_boundaries.extend(float(0.5 * (centers[idx] + centers[idx + 1])) for idx in range(positive_region_count - 1))
    positive_boundaries.append(float(centers[-1] + 0.75 * widths[-1]))
    return centers.astype(float), widths.astype(float), positive_boundaries


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
    temperature_min: float
    temperature_max: float
    setup_eval_samples: int
    family_count: int
    style_count: int
    score_count: int
    quantizer_count: int
    threshold_values: np.ndarray = field(init=False, repr=False)
    region_temperatures: np.ndarray = field(init=False, repr=False)
    coordinate_betas: np.ndarray = field(init=False, repr=False)
    coordinate_bounds: np.ndarray = field(init=False, repr=False)
    coordinate_region_counts: np.ndarray = field(init=False, repr=False)
    style_ids: np.ndarray = field(init=False, repr=False)
    coordinate_score_ids: np.ndarray = field(init=False, repr=False)
    quantizer_ids: np.ndarray = field(init=False, repr=False)
    quantizer_biases: np.ndarray = field(init=False, repr=False)
    region_family_ids: list[np.ndarray] = field(init=False, repr=False)
    positive_centers: list[np.ndarray] = field(init=False, repr=False)
    positive_widths: list[np.ndarray] = field(init=False, repr=False)
    positive_boundaries: list[list[float]] = field(init=False, repr=False)
    region_linear_terms: list[np.ndarray] = field(init=False, repr=False)
    region_shape_terms: list[np.ndarray] = field(init=False, repr=False)
    region_aux_terms: list[np.ndarray] = field(init=False, repr=False)
    region_bias_terms: list[np.ndarray] = field(init=False, repr=False)
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
        if self.temperature_min <= 0.0 or self.temperature_max < self.temperature_min:
            raise ValueError("proposed_final requires 0 < temperature_min <= temperature_max.")
        if self.setup_eval_samples <= 0:
            raise ValueError("proposed_final requires setup_eval_samples > 0.")
        if self.family_count != len(FAMILY_NAMES):
            raise ValueError("proposed_final requires the full heterogeneous family set.")
        if self.style_count != len(STYLE_NAMES):
            raise ValueError("proposed_final requires the full heterogeneous style set.")
        if self.score_count != len(SCORE_NAMES):
            raise ValueError("proposed_final requires the full heterogeneous score-family set.")
        if self.quantizer_count != len(QUANTIZER_NAMES):
            raise ValueError("proposed_final requires the full heterogeneous quantizer set.")

        param_rng = seeded_rng(self.base_seed, "proposed_final:coordinate_params")
        self.threshold_values = np.clip(
            self.boundary_base + param_rng.uniform(-self.boundary_jitter, self.boundary_jitter, size=self.n),
            0.40,
            1.35,
        )
        self.region_temperatures = param_rng.uniform(self.temperature_min, self.temperature_max, size=self.n)
        self.coordinate_betas = param_rng.uniform(self.beta_min, self.beta_max, size=self.n)
        self.coordinate_bounds = param_rng.choice(np.array([2, 3, 4], dtype=int), size=self.n, p=[0.15, 0.40, 0.45])
        self.coordinate_region_counts = _priority_assignment(
            param_rng,
            self.n,
            np.array([4, 6, 8], dtype=int),
            np.array([6, 8, 4, 8, 6, 4], dtype=int),
        )
        self.style_ids = _priority_assignment(
            param_rng,
            self.n,
            np.arange(self.style_count, dtype=int),
            np.array([0, 4, 5, 3, 1, 2, 4, 5], dtype=int),
        )
        self.coordinate_score_ids = _priority_assignment(
            param_rng,
            self.n,
            np.arange(self.score_count, dtype=int),
            np.array([0, 2, 4, 1, 5, 3, 4, 2], dtype=int),
        )
        self.quantizer_ids = _priority_assignment(
            param_rng,
            self.n,
            np.arange(self.quantizer_count, dtype=int),
            np.array([0, 1, 2, 3, 0, 2, 1, 3], dtype=int),
        )
        self.quantizer_biases = param_rng.uniform(0.38, 0.62, size=self.n)
        self.region_family_ids = []
        self.positive_centers = []
        self.positive_widths = []
        self.positive_boundaries = []
        self.region_linear_terms = []
        self.region_shape_terms = []
        self.region_aux_terms = []
        self.region_bias_terms = []
        for coordinate_index in range(self.n):
            positive_region_count = int(self.coordinate_region_counts[coordinate_index] // 2)
            coordinate_rng = seeded_rng(self.base_seed, f"proposed_final:coord:{coordinate_index}")
            centers, widths, boundaries = _build_positive_geometry(
                threshold=float(self.threshold_values[coordinate_index]),
                positive_region_count=positive_region_count,
                rng=coordinate_rng,
            )
            slot_family_ids = _priority_assignment(
                coordinate_rng,
                positive_region_count,
                np.arange(self.family_count, dtype=int),
                np.array([1, 4, 5, 0, 3, 2], dtype=int),
            )
            self.region_family_ids.append(slot_family_ids.astype(int))
            self.positive_centers.append(centers)
            self.positive_widths.append(widths)
            self.positive_boundaries.append(boundaries)
            self.region_linear_terms.append(coordinate_rng.uniform(0.34, 1.02, size=positive_region_count))
            self.region_shape_terms.append(coordinate_rng.uniform(0.12, 0.72, size=positive_region_count))
            self.region_aux_terms.append(coordinate_rng.uniform(0.70, 2.60, size=positive_region_count))
            self.region_bias_terms.append(coordinate_rng.uniform(0.00, 0.09, size=positive_region_count))
        self.setup_diagnostics = self._estimate_setup_diagnostics()

    def _region_scores(self, coordinate_index: int, value: float) -> np.ndarray:
        temperature = float(self.region_temperatures[coordinate_index])
        score_id = int(self.coordinate_score_ids[coordinate_index])
        positive_centers = self.positive_centers[coordinate_index]
        positive_widths = self.positive_widths[coordinate_index]
        centers = np.concatenate([-positive_centers[::-1], positive_centers])
        widths = np.concatenate([positive_widths[::-1], positive_widths])
        normalized = (value - centers) / np.maximum(widths, 1e-6)

        if score_id == 0:
            psi = -0.5 * (normalized**2)
        elif score_id == 1:
            psi = -np.abs(normalized)
        elif score_id == 2:
            psi = -np.log1p(normalized**2)
        elif score_id == 3:
            psi = -np.maximum(np.abs(normalized) - 0.25, 0.0)
        elif score_id == 4:
            psi = -(normalized**2) - 0.12 * (normalized**4)
        else:
            clipped = np.clip(normalized, -1.0, 1.0)
            psi = np.cos(0.5 * np.pi * clipped) - 1.25 * np.abs(normalized)

        return temperature * np.asarray(psi, dtype=float)

    def _positive_response(self, coordinate_index: int, region_slot: int, magnitude: float) -> float:
        center = float(self.positive_centers[coordinate_index][region_slot])
        width = float(self.positive_widths[coordinate_index][region_slot])
        transformed = _slot_transform(magnitude=magnitude, center=center, width=width, slot_index=region_slot)
        linear = float(self.region_linear_terms[coordinate_index][region_slot])
        shape = float(self.region_shape_terms[coordinate_index][region_slot])
        aux = float(self.region_aux_terms[coordinate_index][region_slot])
        bias = float(self.region_bias_terms[coordinate_index][region_slot])
        style_id = int(self.style_ids[coordinate_index])
        family_id = int(self.region_family_ids[coordinate_index][region_slot])

        if style_id == 0:
            linear *= 1.02
            shape *= 0.46
            bias *= 0.24
        elif style_id == 1:
            linear *= 0.84
            shape *= 0.34
            bias *= 0.52
        elif style_id == 2:
            linear *= 0.68
            shape *= 0.40
            bias *= 0.16
        elif style_id == 3:
            linear *= 0.78
            shape *= 0.62
            bias *= 0.42
        elif style_id == 4:
            linear *= 0.82
            shape *= 0.92
            bias *= 0.34
        else:
            linear *= 1.12
            shape *= 0.76
            bias *= 0.30

        if family_id == 0:
            clipped = min(transformed, 1.85)
            base_value = linear * clipped + shape * (clipped**3)
        elif family_id == 1:
            base_value = linear * transformed + shape * np.tanh(aux * transformed)
        elif family_id == 2:
            base_value = linear * transformed + shape * min(transformed * transformed, aux)
        elif family_id == 3:
            base_value = linear * transformed + shape * np.sin(aux * transformed)
        elif family_id == 4:
            clipped = min(transformed, 1.95)
            base_value = linear * clipped + 0.55 * shape * (clipped**2) + 0.30 * shape * (clipped**3)
        else:
            base_value = linear * transformed
            base_value += 0.32 * shape * np.tanh(aux * (transformed - 0.55))
            base_value += 0.26 * shape * np.tanh(0.75 * aux * (transformed - 1.15))

        if style_id == 0:
            response = base_value
        elif style_id == 1:
            response = 0.82 * np.tanh(1.10 * base_value) + 0.34 * base_value
        elif style_id == 2:
            threshold = float(self.threshold_values[coordinate_index])
            sparsity_gate = 0.28 + 0.72 * (1.0 - np.exp(-((magnitude / max(threshold + 0.18, 1e-6)) ** 2)))
            response = base_value * sparsity_gate
        elif style_id == 3:
            mode_target = 1.20 + 0.24 * region_slot + 0.20 * (int(self.coordinate_bounds[coordinate_index]) - 2)
            response = mode_target * np.tanh(0.98 * base_value) + 0.18 * base_value
        elif style_id == 4:
            tail_power = 1.35 + 0.18 * region_slot
            tail_component = linear * transformed + shape * (max(transformed, 0.0) ** tail_power)
            response = 0.48 * base_value + 0.52 * tail_component
        else:
            response = base_value + 0.18 * shape * np.tanh(aux * (transformed - 0.90))

        return float(max(0.0, response + bias))

    def _soft_region_response(self, coordinate_index: int, value: float) -> tuple[float, np.ndarray]:
        magnitude = abs(value)
        positive_region_count = int(self.coordinate_region_counts[coordinate_index] // 2)
        positive_outputs = np.array(
            [
                self._positive_response(coordinate_index, region_slot=slot, magnitude=magnitude)
                for slot in range(positive_region_count)
            ],
            dtype=float,
        )
        region_outputs = np.concatenate([-positive_outputs[::-1], positive_outputs])
        weights = _softmax(self._region_scores(coordinate_index=coordinate_index, value=value))
        response = float(np.dot(weights, region_outputs))
        return response, weights

    def _quantize_coordinate(
        self,
        coordinate_index: int,
        value: float,
        rng: np.random.Generator,
    ) -> tuple[int, bool]:
        bound = int(self.coordinate_bounds[coordinate_index])
        quantizer_id = int(self.quantizer_ids[coordinate_index])
        lower = int(floor(value))
        upper = lower + 1
        fraction = float(value - lower)

        if quantizer_id == 0:
            quantized = upper if rng.random() < fraction else lower
        elif quantizer_id == 1:
            threshold = float(np.clip(self.quantizer_biases[coordinate_index] + rng.uniform(-0.14, 0.14), 0.08, 0.92))
            quantized = upper if fraction >= threshold else lower
        elif quantizer_id == 2:
            quantized = int(floor(value + rng.uniform(0.0, 1.0)))
        else:
            deadzone = 0.18 + 0.04 * int(self.coordinate_bounds[coordinate_index])
            if abs(value) < deadzone and rng.random() < 0.45:
                quantized = 0
            else:
                quantized = upper if rng.random() < fraction else lower

        clipped = abs(quantized) > bound
        return int(np.clip(quantized, -bound, bound)), bool(clipped)

    def _sample_secret_vector(self, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
        z_values = rng.normal(loc=0.0, scale=1.0, size=self.n)
        dither = rng.integers(-1, 2, size=self.n)
        dominant_regions = np.zeros(self.n, dtype=int)
        pre_quantized = np.zeros(self.n, dtype=float)
        secret = np.zeros(self.n, dtype=int)
        clipped_mask = np.zeros(self.n, dtype=bool)
        boundary_margins = np.zeros(self.n, dtype=float)

        for coordinate_index in range(self.n):
            response, weights = self._soft_region_response(coordinate_index, float(z_values[coordinate_index]))
            positive_region_count = int(self.coordinate_region_counts[coordinate_index] // 2)
            dominant_index = int(np.argmax(weights))
            if dominant_index < positive_region_count:
                coarse_region = 0 if dominant_index < max(1, positive_region_count // 2) else 1
            else:
                positive_slot = dominant_index - positive_region_count
                coarse_region = 2 if positive_slot < max(1, positive_region_count // 2) else 3
            dominant_regions[coordinate_index] = coarse_region
            boundary_margins[coordinate_index] = _region_boundary_margin(
                value=float(z_values[coordinate_index]),
                threshold=float(self.threshold_values[coordinate_index]),
            )
            pre_quantized_value = response + self.coordinate_betas[coordinate_index] * float(dither[coordinate_index])
            pre_quantized[coordinate_index] = pre_quantized_value
            secret[coordinate_index], clipped_mask[coordinate_index] = self._quantize_coordinate(
                coordinate_index=coordinate_index,
                value=pre_quantized_value,
                rng=rng,
            )

        metadata = {
            "region_counts_vector": np.bincount(dominant_regions, minlength=self.region_count),
            "avg_abs_before_quantization": float(np.mean(np.abs(pre_quantized))),
            "avg_abs_after_quantization": float(np.mean(np.abs(secret))),
            "fraction_clipped_coordinates": float(np.mean(clipped_mask.astype(float))),
            "occupied_quantization_levels": int(np.unique(secret).size),
            "region_boundary_margin": float(np.mean(boundary_margins)),
        }
        return secret, metadata

    def _expected_region_counts(self) -> np.ndarray:
        counts = np.zeros(self.region_count, dtype=float)
        for coordinate_index in range(self.n):
            positive_region_count = int(self.coordinate_region_counts[coordinate_index] // 2)
            split_index = min(max(1, positive_region_count // 2), len(self.positive_boundaries[coordinate_index]) - 1)
            split_boundary = float(self.positive_boundaries[coordinate_index][split_index])
            outer_prob = 1.0 - _normal_cdf(split_boundary)
            inner_prob = _normal_cdf(split_boundary) - 0.5
            counts += np.array([outer_prob, inner_prob, inner_prob, outer_prob], dtype=float) * self.setup_eval_samples
        return counts

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
        max_bound = int(np.max(self.coordinate_bounds))

        for coordinate_index in range(self.n):
            bound = int(self.coordinate_bounds[coordinate_index])
            levels = np.arange(-bound, bound + 1)
            counts = np.array([np.sum(setup_secrets[:, coordinate_index] == level) for level in levels], dtype=float)
            per_coordinate_entropy.append(_entropy_base2(counts))
            occupied_levels_per_coordinate.append(int(np.sum(counts > 0.0)))

        for level in range(1, max_bound + 1):
            positive = np.mean(setup_secrets == level)
            negative = np.mean(setup_secrets == -level)
            symmetry_gap_by_level[str(level)] = float(abs(positive - negative))

        style_group_summary: dict[str, dict[str, float]] = {}
        entropies = np.asarray(per_coordinate_entropy, dtype=float)
        occupied = np.asarray(occupied_levels_per_coordinate, dtype=float)
        for style_id, style_name in STYLE_NAMES.items():
            indices = np.where(self.style_ids == style_id)[0]
            if indices.size == 0:
                continue
            group = setup_secrets[:, indices]
            style_group_summary[style_name] = {
                "coordinate_count": float(indices.size),
                "abs_mean": float(np.mean(np.abs(group))),
                "variance_mean": float(np.mean(np.var(group, axis=0))),
                "entropy_mean": float(np.mean(entropies[indices])),
                "occupied_levels_mean": float(np.mean(occupied[indices])),
            }

        pre_cov = np.cov(setup_secrets.astype(float), rowvar=False)
        pre_cov = np.atleast_2d(pre_cov)
        diag = np.diag(pre_cov)

        coefficient_a = np.concatenate([values for values in self.region_linear_terms])
        coefficient_b = np.concatenate([values for values in self.region_shape_terms])
        coefficient_c = np.concatenate([values for values in self.region_aux_terms])
        coordinate_family_mix = [
            "|".join(_family_name(family_id) for family_id in slot_family_ids.tolist())
            for slot_family_ids in self.region_family_ids
        ]
        nested_family_names = [
            [_family_name(family_id) for family_id in slot_family_ids.tolist()]
            for slot_family_ids in self.region_family_ids
        ]
        nested_boundaries = [[round(value, 6) for value in boundaries] for boundaries in self.positive_boundaries]
        quantizer_counts = {
            _quantizer_name(quantizer_id): int(np.sum(self.quantizer_ids == quantizer_id))
            for quantizer_id in sorted(QUANTIZER_NAMES)
        }
        score_counts = {
            _score_name(score_id): int(np.sum(self.coordinate_score_ids == score_id))
            for score_id in sorted(SCORE_NAMES)
        }

        return {
            "generator_variant": "coordinatewise_meta_heterogeneous_soft_region_generator",
            "coordinates_generated_independently": True,
            "setup_region_occupancy_counts_json": json_dumps(region_counts.tolist()),
            "setup_region_occupancy_entropy": float(_entropy_base2(region_counts)),
            "initial_region_balance_counts_json": json_dumps(np.rint(self._expected_region_counts()).astype(int).tolist()),
            "initial_region_balance_entropy": float(_entropy_base2(self._expected_region_counts())),
            "region_lattice_counts_json": None,
            "region_lattice_entropy": None,
            "pre_normalization_cov_diag_stats": _summary_stats(diag),
            "post_normalization_cov_diag_stats": {"mean": None, "std": None, "min": None, "max": None},
            "pre_whitening_cov_diag_stats": _summary_stats(diag),
            "post_whitening_cov_diag_stats": {"mean": None, "std": None, "min": None, "max": None},
            "whitening_condition_number": None,
            "orthogonal_mixing_seed": None,
            "structure_seed_offset": None,
            "orthogonal_seed_offset": None,
            "signed_permutation_seed": None,
            "quantization_mode": "coordinatewise_randomized_bounded_quantization",
            "noise_injection_mode": "coordinatewise_light_ternary_dither",
            "alpha": None,
            "beta": float(np.mean(self.coordinate_betas)),
            "gamma": None,
            "map_scale": None,
            "offset_scale": None,
            "within_region_sigma": None,
            "uniform_region_mix": None,
            "boundary_focus_mix": None,
            "boundary_temperature": float(np.mean(self.region_temperatures)),
            "shell_balance_mix": None,
            "region_sampling_mode": "soft_scalar_regions",
            "implied_prior_effective_rank": None,
            "implied_prior_pairwise_corr_mean_abs": None,
            "implied_prior_abs_mean": None,
            "implied_prior_boundary_margin_mean": None,
            "coordinate_region_boundaries_json": json_dumps(nested_boundaries),
            "coordinate_region_count_json": json_dumps(self.coordinate_region_counts.astype(int).tolist()),
            "coordinate_style_assignments_json": json_dumps([_style_name(style_id) for style_id in self.style_ids]),
            "coordinate_style_counts_json": json_dumps(
                {_style_name(style_id): int(np.sum(self.style_ids == style_id)) for style_id in sorted(STYLE_NAMES)}
            ),
            "coordinate_score_family_json": json_dumps(
                [_score_name(score_id) for score_id in self.coordinate_score_ids]
            ),
            "function_family_per_coordinate_json": json_dumps(coordinate_family_mix),
            "region_function_family_nested_json": json_dumps(nested_family_names),
            "quantizer_type_per_coordinate_json": json_dumps(
                [_quantizer_name(quantizer_id) for quantizer_id in self.quantizer_ids]
            ),
            "quantizer_counts_json": json_dumps(quantizer_counts),
            "score_counts_json": json_dumps(score_counts),
            "soft_region_temperature_json": json_dumps(np.round(self.region_temperatures, 6).tolist()),
            "coefficient_a_summary_json": json_dumps(_summary_stats(coefficient_a)),
            "coefficient_b_summary_json": json_dumps(_summary_stats(coefficient_b)),
            "coefficient_c_summary_json": json_dumps(_summary_stats(coefficient_c)),
            "coordinate_beta_json": json_dumps(np.round(self.coordinate_betas, 6).tolist()),
            "coordinate_clipping_bounds_json": json_dumps(self.coordinate_bounds.astype(int).tolist()),
            "empirical_coordinate_mean_json": json_dumps(np.round(per_coordinate_mean, 6).tolist()),
            "empirical_coordinate_variance_json": json_dumps(np.round(per_coordinate_variance, 6).tolist()),
            "empirical_coordinate_entropy_json": json_dumps(np.round(entropies, 6).tolist()),
            "setup_fraction_clipped_coordinates": float(clipped_fraction_total / self.setup_eval_samples),
            "average_occupied_quantization_levels": float(np.mean(occupied)),
            "samplewise_occupied_quantization_levels_mean": float(np.mean(occupied_levels_per_sample)),
            "per_coordinate_occupied_levels_json": json_dumps(occupied_levels_per_coordinate),
            "style_group_distribution_summary_json": json_dumps(style_group_summary),
            "symmetry_bin_gap_json": json_dumps(symmetry_gap_by_level),
            "setup_coordinate_variance_trace": float(np.trace(pre_cov)),
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
            "coordinate_region_count_json": self.setup_diagnostics["coordinate_region_count_json"],
            "coordinate_style_assignments_json": self.setup_diagnostics["coordinate_style_assignments_json"],
            "coordinate_style_counts_json": self.setup_diagnostics["coordinate_style_counts_json"],
            "coordinate_score_family_json": self.setup_diagnostics["coordinate_score_family_json"],
            "function_family_per_coordinate_json": self.setup_diagnostics["function_family_per_coordinate_json"],
            "region_function_family_nested_json": self.setup_diagnostics["region_function_family_nested_json"],
            "quantizer_type_per_coordinate_json": self.setup_diagnostics["quantizer_type_per_coordinate_json"],
            "quantizer_counts_json": self.setup_diagnostics["quantizer_counts_json"],
            "score_counts_json": self.setup_diagnostics["score_counts_json"],
            "soft_region_temperature_json": self.setup_diagnostics["soft_region_temperature_json"],
            "coefficient_a_summary_json": self.setup_diagnostics["coefficient_a_summary_json"],
            "coefficient_b_summary_json": self.setup_diagnostics["coefficient_b_summary_json"],
            "coefficient_c_summary_json": self.setup_diagnostics["coefficient_c_summary_json"],
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
            "per_coordinate_occupied_levels_json": self.setup_diagnostics["per_coordinate_occupied_levels_json"],
            "style_group_distribution_summary_json": self.setup_diagnostics["style_group_distribution_summary_json"],
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
            beta_min=float(params.get("beta_min", 0.04)),
            beta_max=float(params.get("beta_max", 0.12)),
            boundary_base=float(params.get("boundary_base", 0.90)),
            boundary_jitter=float(params.get("boundary_jitter", 0.18)),
            temperature_min=float(params.get("temperature_min", 1.35)),
            temperature_max=float(params.get("temperature_max", 2.40)),
            setup_eval_samples=int(params.get("setup_eval_samples", 4096)),
            family_count=int(params.get("family_count", len(FAMILY_NAMES))),
            style_count=int(params.get("style_count", len(STYLE_NAMES))),
            score_count=int(params.get("score_count", len(SCORE_NAMES))),
            quantizer_count=int(params.get("quantizer_count", len(QUANTIZER_NAMES))),
        )
    raise ValueError(f"Unsupported generator type for {model_name}: {generator_type}")
