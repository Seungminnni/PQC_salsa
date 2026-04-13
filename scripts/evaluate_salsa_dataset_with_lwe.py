#!/usr/bin/env python3
import argparse
import csv
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lwe_experiment.core import extract_public_features, recover_secret, run_decryption_trials
from lwe_experiment.metrics import (
    compute_correctness_metrics,
    compute_distribution_distance_metrics,
    compute_recovery_metrics,
    compute_secret_distribution_metrics,
)
from lwe_experiment.utils import centered_mod


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a SALSA AB dataset using lwe_experiment core/metrics functions."
    )
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to SALSA AB dataset directory.")
    parser.add_argument("--reference_dir", type=str, default="", help="Optional reference dataset directory.")
    parser.add_argument("--trials_per_secret", type=int, default=256, help="Decryption trials per secret.")
    parser.add_argument("--seed", type=int, default=20260407, help="Base seed for decryption trials.")
    parser.add_argument("--num_clusters", type=int, default=4, help="Cluster count for distance metrics.")
    parser.add_argument("--clipping_bound", type=int, default=-1, help="Override clipping bound.")
    parser.add_argument(
        "--recovery_refinement_passes",
        type=int,
        default=4,
        help="Local refinement passes for lwe_experiment recovery.",
    )
    parser.add_argument(
        "--exhaustive_recovery_limit",
        type=int,
        default=0,
        help="Search limit for exhaustive recovery. 0 disables exhaustive search.",
    )
    parser.add_argument("--output_json", type=str, default="", help="Optional output JSON path.")
    parser.add_argument("--output_csv", type=str, default="", help="Optional output CSV path.")
    parser.add_argument(
        "--output_per_secret_csv",
        type=str,
        default="",
        help="Optional per-secret correctness CSV path.",
    )
    return parser.parse_args()


def flatten_row(data: dict) -> dict:
    flattened = {}
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            flattened[key] = json.dumps(value, ensure_ascii=True, sort_keys=True)
        else:
            flattened[key] = value
    return flattened


def infer_clipping_bound(secrets: np.ndarray, override: int) -> int:
    if override > 0:
        return int(override)
    return max(1, int(np.max(np.abs(secrets))))


def load_salsa_arrays(dataset_dir: Path, q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    secrets = np.load(dataset_dir / "secret.npy").T.astype(int)
    public_matrix = np.load(dataset_dir / "orig_A.npy").astype(int)
    public_vectors = np.load(dataset_dir / "orig_b.npy").T.astype(int)
    public_matrices = np.repeat(public_matrix[None, :, :], secrets.shape[0], axis=0)
    public_errors = np.asarray(
        [centered_mod(public_vectors[i] - (public_matrix @ secrets[i]), q).astype(int) for i in range(secrets.shape[0])],
        dtype=int,
    )
    return secrets, public_matrices, public_vectors, public_errors


def run_repeated_decryption(
    public_matrices: np.ndarray,
    public_vectors: np.ndarray,
    secrets: np.ndarray,
    q: int,
    seed: int,
    trials_per_secret: int,
):
    repeated_matrices = np.repeat(public_matrices, trials_per_secret, axis=0)
    repeated_vectors = np.repeat(public_vectors, trials_per_secret, axis=0)
    repeated_secrets = np.repeat(secrets, trials_per_secret, axis=0)
    decryption = run_decryption_trials(
        public_matrices=repeated_matrices,
        public_vectors=repeated_vectors,
        secrets=repeated_secrets,
        q=q,
        run_seed=seed,
    )
    correctness = compute_correctness_metrics(
        secrets=repeated_secrets,
        messages=decryption.messages,
        decoded_messages=decryption.decoded_messages,
        effective_noise=decryption.effective_noise,
        noise_margin=decryption.noise_margin,
    )
    per_secret = (decryption.messages == decryption.decoded_messages).reshape(secrets.shape[0], trials_per_secret).mean(axis=1)
    return decryption, correctness, per_secret


def maybe_reference_metrics(secrets: np.ndarray, reference_dir: str, num_clusters: int, seed: int):
    if not reference_dir:
        return {}
    reference = np.load(Path(reference_dir) / "secret.npy").T.astype(int)
    metrics, _ = compute_distribution_distance_metrics(
        secrets=secrets,
        baseline_reference=reference,
        num_clusters=num_clusters,
        seed=seed,
    )
    return metrics


def compute_recovery_for_salsa(
    public_matrices: np.ndarray,
    public_vectors: np.ndarray,
    secrets: np.ndarray,
    q: int,
    clipping_bound: int,
    passes: int,
    exhaustive_limit: int,
):
    recovered = np.asarray(
        [
            recover_secret(
                A=public_matrices[i],
                b=public_vectors[i],
                q=q,
                clipping_bound=clipping_bound,
                passes=passes,
                exhaustive_limit=exhaustive_limit,
            )
            for i in range(secrets.shape[0])
        ],
        dtype=int,
    )
    return recovered, compute_recovery_metrics(
        true_secrets=secrets,
        recovered_secrets=recovered,
        clipping_bound=clipping_bound,
    )


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    params = pickle.load(open(dataset_dir / "params.pkl", "rb"))
    if not isinstance(params, dict):
        params = params.__dict__

    q = int(params["Q"])
    secrets, public_matrices, public_vectors, public_errors = load_salsa_arrays(dataset_dir=dataset_dir, q=q)
    clipping_bound = infer_clipping_bound(secrets, args.clipping_bound)

    metrics = {
        "dataset_dir": str(dataset_dir),
        "reference_dir": args.reference_dir or None,
        "secret_type": params.get("secret_type", "binary"),
        "n": int(params["N"]),
        "q": q,
        "sigma": float(params["sigma"]),
        "num_secret_samples": int(secrets.shape[0]),
        "num_dimensions": int(secrets.shape[1]),
        "num_secret_seeds": int(params.get("num_secret_seeds", secrets.shape[0])),
        "min_hamming": int(params.get("min_hamming", 3)),
        "max_hamming": int(params.get("max_hamming", 3)),
        "clipping_bound_used": int(clipping_bound),
        "public_matrix_rows": int(public_matrices.shape[1]),
        "trials_per_secret": int(args.trials_per_secret),
        "evaluation_backend": "lwe_experiment",
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        metrics.update(compute_secret_distribution_metrics(secrets=secrets, clipping_bound=clipping_bound))

    pre_projection_path = dataset_dir / "pre_projection_secret.npy"
    if pre_projection_path.is_file():
        pre_projection = np.load(pre_projection_path).T.astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pre_metrics = compute_secret_distribution_metrics(
                secrets=pre_projection,
                clipping_bound=max(clipping_bound, int(np.max(np.abs(pre_projection)))),
            )
        metrics.update({f"pre_projection_{key}": value for key, value in pre_metrics.items()})

    decryption, correctness_metrics, per_secret_correctness = run_repeated_decryption(
        public_matrices=public_matrices,
        public_vectors=public_vectors,
        secrets=secrets,
        q=q,
        seed=args.seed,
        trials_per_secret=args.trials_per_secret,
    )
    metrics.update(correctness_metrics)
    metrics.update(
        {
            "per_secret_decrypt_success_mean": float(np.mean(per_secret_correctness)),
            "per_secret_decrypt_success_std": float(np.std(per_secret_correctness)),
            "per_secret_decrypt_success_min": float(np.min(per_secret_correctness)),
            "per_secret_decrypt_success_max": float(np.max(per_secret_correctness)),
            "public_error_abs_mean": float(np.mean(np.abs(public_errors))),
            "public_error_abs_max": float(np.max(np.abs(public_errors))),
        }
    )

    public_features = np.asarray(
        [extract_public_features(A=public_matrices[i], b=public_vectors[i], q=q) for i in range(secrets.shape[0])],
        dtype=float,
    )
    recovered_secrets, recovery_metrics = compute_recovery_for_salsa(
        public_matrices=public_matrices,
        public_vectors=public_vectors,
        secrets=secrets,
        q=q,
        clipping_bound=clipping_bound,
        passes=args.recovery_refinement_passes,
        exhaustive_limit=args.exhaustive_recovery_limit,
    )
    metrics.update(recovery_metrics)
    metrics.update(
        maybe_reference_metrics(
            secrets=secrets,
            reference_dir=args.reference_dir,
            num_clusters=args.num_clusters,
            seed=args.seed,
        )
    )

    output_json = Path(args.output_json) if args.output_json else dataset_dir / "dataset_metrics_lwe.json"
    output_csv = Path(args.output_csv) if args.output_csv else dataset_dir / "dataset_metrics_lwe.csv"
    output_per_secret_csv = (
        Path(args.output_per_secret_csv)
        if args.output_per_secret_csv
        else dataset_dir / "per_secret_correctness_lwe.csv"
    )

    output_json.write_text(json.dumps(metrics, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        row = flatten_row(metrics)
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    with output_per_secret_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "secret_seed",
            "decrypt_success_rate",
            "secret_l2_norm",
            "secret_linf_norm",
            "mean_abs_public_error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for secret_seed in range(secrets.shape[0]):
            writer.writerow(
                {
                    "secret_seed": secret_seed,
                    "decrypt_success_rate": float(per_secret_correctness[secret_seed]),
                    "secret_l2_norm": float(np.linalg.norm(secrets[secret_seed])),
                    "secret_linf_norm": float(np.max(np.abs(secrets[secret_seed]))),
                    "mean_abs_public_error": float(np.mean(np.abs(public_errors[secret_seed]))),
                }
            )

    print(f"Wrote lwe_experiment-backed metrics to {output_json}")
    print(f"Wrote lwe_experiment-backed table to {output_csv}")
    print(f"Wrote per-secret correctness to {output_per_secret_csv}")


if __name__ == "__main__":
    main()
