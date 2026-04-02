from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .secret_generators import build_secret_generator
from .utils import centered_mod, discrete_gaussian, seeded_rng


@dataclass(slots=True)
class DatasetBundle:
    secrets: np.ndarray
    public_matrices: np.ndarray
    public_vectors: np.ndarray
    public_errors: np.ndarray
    public_features: np.ndarray
    recovery_estimates: np.ndarray
    keygen_time_ms: float
    generator_run_diagnostics: dict[str, Any]
    generator_sample_diagnostics: dict[str, np.ndarray]


@dataclass(slots=True)
class DecryptionBundle:
    messages: np.ndarray
    decoded_messages: np.ndarray
    effective_noise: np.ndarray
    phase_values: np.ndarray
    noise_margin: np.ndarray
    encrypt_time_ms: float
    decrypt_time_ms: float


def iterative_wrapped_least_squares(
    A: np.ndarray,
    b: np.ndarray,
    q: int,
    ridge: float = 1e-2,
    iterations: int = 4,
) -> np.ndarray:
    A_center = centered_mod(A, q).astype(float)
    b_center = centered_mod(b, q).astype(float)
    gram = A_center.T @ A_center + ridge * np.eye(A.shape[1], dtype=float)
    rhs = A_center.T @ b_center
    estimate = np.linalg.solve(gram, rhs)
    for _ in range(iterations):
        wrap = np.rint((A_center @ estimate - b_center) / q)
        lifted_target = b_center + q * wrap
        estimate = np.linalg.solve(gram, A_center.T @ lifted_target)
    return estimate


def local_refine_secret(
    A: np.ndarray,
    b: np.ndarray,
    q: int,
    initial: np.ndarray,
    clipping_bound: int,
    passes: int,
) -> np.ndarray:
    candidate = np.clip(np.rint(initial), -clipping_bound, clipping_bound).astype(int)

    def residual_score(secret: np.ndarray) -> float:
        residual = centered_mod(b - (A @ secret), q).astype(float)
        return float(np.mean(residual**2))

    current_score = residual_score(candidate)
    for _ in range(passes):
        improved = False
        for index in range(candidate.size):
            best_value = int(candidate[index])
            best_score = current_score
            for value in range(-clipping_bound, clipping_bound + 1):
                if value == candidate[index]:
                    continue
                trial = candidate.copy()
                trial[index] = value
                trial_score = residual_score(trial)
                if trial_score < best_score:
                    best_value = value
                    best_score = trial_score
            if best_value != candidate[index]:
                candidate[index] = best_value
                current_score = best_score
                improved = True
        if not improved:
            break
    return candidate


def maybe_exhaustive_recovery(
    A: np.ndarray,
    b: np.ndarray,
    q: int,
    clipping_bound: int,
    exhaustive_limit: int,
) -> np.ndarray | None:
    if exhaustive_limit <= 0:
        return None
    search_space = (2 * clipping_bound + 1) ** A.shape[1]
    if search_space > exhaustive_limit:
        return None
    candidates = np.array(
        list(itertools.product(range(-clipping_bound, clipping_bound + 1), repeat=A.shape[1])),
        dtype=int,
    )
    residuals = centered_mod(b[:, None] - ((A @ candidates.T) % q), q).astype(float)
    scores = np.mean(residuals**2, axis=0)
    return candidates[int(np.argmin(scores))]


def recover_secret(
    A: np.ndarray,
    b: np.ndarray,
    q: int,
    clipping_bound: int,
    passes: int,
    exhaustive_limit: int,
) -> np.ndarray:
    exhaustive = maybe_exhaustive_recovery(A, b, q, clipping_bound, exhaustive_limit)
    if exhaustive is not None:
        return exhaustive
    estimate = iterative_wrapped_least_squares(A=A, b=b, q=q)
    return local_refine_secret(
        A=A,
        b=b,
        q=q,
        initial=estimate,
        clipping_bound=clipping_bound,
        passes=passes,
    )


def extract_public_features(A: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    A_center = centered_mod(A, q).astype(float) / q
    b_center = centered_mod(b, q).astype(float) / q
    gram = (A_center.T @ A_center) / max(A.shape[0], 1)
    matched = (A_center.T @ b_center) / max(A.shape[0], 1)
    ridge_estimate = np.linalg.solve(gram + 1e-3 * np.eye(A.shape[1]), matched)
    wrapped_estimate = iterative_wrapped_least_squares(A=A, b=b, q=q)
    residual = centered_mod(b - (A @ np.rint(wrapped_estimate).astype(int)), q).astype(float) / q
    return np.concatenate(
        [
            matched,
            ridge_estimate,
            wrapped_estimate,
            np.diag(gram),
            np.array(
                [
                    b_center.mean(),
                    b_center.std(),
                    b_center.min(),
                    b_center.max(),
                    residual.mean(),
                    residual.std(),
                    np.max(np.abs(residual)),
                ]
            ),
        ]
    )


def generate_dataset(
    n: int,
    m: int,
    q: int,
    noise_sigma: float,
    num_samples: int,
    run_seed: int,
    generator_type: str,
    generator_params: dict[str, int | float | bool | str],
    recovery_passes: int,
    exhaustive_recovery_limit: int,
    model_name: str,
) -> DatasetBundle:
    generator = build_secret_generator(
        model_name=model_name,
        generator_type=generator_type,
        n=n,
        run_seed=run_seed,
        params=generator_params,
    )

    secrets = np.zeros((num_samples, n), dtype=int)
    public_matrices = np.zeros((num_samples, m, n), dtype=int)
    public_vectors = np.zeros((num_samples, m), dtype=int)
    public_errors = np.zeros((num_samples, m), dtype=int)
    public_features = []
    recovery_estimates = np.zeros((num_samples, n), dtype=int)
    sample_diagnostics_rows: list[dict[str, Any]] = []

    start = time.perf_counter()
    clipping_bound = int(generator_params["clipping_bound"])
    for sample_index in range(num_samples):
        sampled_secret = generator.sample_with_metadata(sample_index)
        secret = sampled_secret.secret
        matrix_rng = seeded_rng(run_seed, f"A:{sample_index}")
        noise_rng = seeded_rng(run_seed, f"e:{sample_index}")
        A = matrix_rng.integers(0, q, size=(m, n), endpoint=False)
        e = discrete_gaussian(noise_rng, size=m, sigma=noise_sigma)
        b = (A @ secret + e) % q

        secrets[sample_index] = secret
        public_matrices[sample_index] = A
        public_vectors[sample_index] = b
        public_errors[sample_index] = e
        public_features.append(extract_public_features(A=A, b=b, q=q))
        sample_diagnostics_rows.append(sampled_secret.metadata)
        recovery_estimates[sample_index] = recover_secret(
            A=A,
            b=b,
            q=q,
            clipping_bound=clipping_bound,
            passes=recovery_passes,
            exhaustive_limit=exhaustive_recovery_limit,
        )

    elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(num_samples, 1)
    generator_sample_diagnostics: dict[str, np.ndarray] = {}
    if sample_diagnostics_rows:
        all_keys = sorted({key for row in sample_diagnostics_rows for key in row})
        for key in all_keys:
            generator_sample_diagnostics[key] = np.asarray([row.get(key) for row in sample_diagnostics_rows])
    return DatasetBundle(
        secrets=secrets,
        public_matrices=public_matrices,
        public_vectors=public_vectors,
        public_errors=public_errors,
        public_features=np.asarray(public_features, dtype=float),
        recovery_estimates=recovery_estimates,
        keygen_time_ms=elapsed_ms,
        generator_run_diagnostics=generator.get_setup_diagnostics(),
        generator_sample_diagnostics=generator_sample_diagnostics,
    )


def run_decryption_trials(
    public_matrices: np.ndarray,
    public_vectors: np.ndarray,
    secrets: np.ndarray,
    q: int,
    run_seed: int,
) -> DecryptionBundle:
    num_samples, m, _ = public_matrices.shape
    messages = np.zeros(num_samples, dtype=int)
    decoded_messages = np.zeros(num_samples, dtype=int)
    effective_noise = np.zeros(num_samples, dtype=float)
    phase_values = np.zeros(num_samples, dtype=float)
    noise_margin = np.zeros(num_samples, dtype=float)

    delta = q // 2
    encrypt_total = 0.0
    decrypt_total = 0.0
    for sample_index in range(num_samples):
        matrix = public_matrices[sample_index]
        vector = public_vectors[sample_index]
        secret = secrets[sample_index]

        enc_rng = seeded_rng(run_seed, f"enc:{sample_index}")
        message = int(enc_rng.integers(0, 2))
        randomness = enc_rng.integers(0, 2, size=m)

        enc_start = time.perf_counter()
        u = (matrix.T @ randomness) % q
        v = int((vector @ randomness + message * delta) % q)
        encrypt_total += time.perf_counter() - enc_start

        dec_start = time.perf_counter()
        phase = float(centered_mod(np.array(v - int(u @ secret)), q))
        decoded = int(abs(phase) > (q / 4.0))
        eta = float(centered_mod(np.array(phase - message * delta), q))
        decrypt_total += time.perf_counter() - dec_start

        messages[sample_index] = message
        decoded_messages[sample_index] = decoded
        effective_noise[sample_index] = eta
        phase_values[sample_index] = phase
        noise_margin[sample_index] = (q / 4.0) - abs(eta)

    return DecryptionBundle(
        messages=messages,
        decoded_messages=decoded_messages,
        effective_noise=effective_noise,
        phase_values=phase_values,
        noise_margin=noise_margin,
        encrypt_time_ms=encrypt_total * 1000.0 / max(num_samples, 1),
        decrypt_time_ms=decrypt_total * 1000.0 / max(num_samples, 1),
    )
