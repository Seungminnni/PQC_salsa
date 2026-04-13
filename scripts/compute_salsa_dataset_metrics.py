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

from lwe_experiment.metrics import (
    compute_correctness_metrics,
    compute_distribution_distance_metrics,
    compute_secret_distribution_metrics,
)
from lwe_experiment.utils import centered_mod, seeded_rng


def parse_args():
    parser = argparse.ArgumentParser(description="Compute AB-level metrics for a SALSA dataset.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to AB dataset directory.")
    parser.add_argument("--reference_dir", type=str, default="", help="Optional reference dataset directory.")
    parser.add_argument("--trials_per_secret", type=int, default=256, help="Decryption trials per secret column.")
    parser.add_argument("--seed", type=int, default=20260407, help="Base seed for correctness trials.")
    parser.add_argument("--num_clusters", type=int, default=4, help="Cluster count for distribution distance metrics.")
    parser.add_argument("--clipping_bound", type=int, default=-1, help="Override clipping bound. Defaults to max abs secret.")
    parser.add_argument("--output_json", type=str, default="", help="Optional output JSON path.")
    parser.add_argument("--output_csv", type=str, default="", help="Optional output CSV path.")
    return parser.parse_args()


def infer_clipping_bound(secrets: np.ndarray, override: int) -> int:
    if override > 0:
        return int(override)
    return max(1, int(np.max(np.abs(secrets))))


def compute_correctness_from_dataset(
    dataset_dir: Path,
    secrets: np.ndarray,
    q: int,
    seed: int,
    trials_per_secret: int,
):
    orig_a_path = dataset_dir / "orig_A.npy"
    orig_b_path = dataset_dir / "orig_b.npy"
    if not orig_a_path.is_file() or not orig_b_path.is_file():
        return {}

    public_matrix = np.load(orig_a_path).astype(int)
    public_vectors = np.load(orig_b_path).astype(int)
    delta = q // 2
    total_trials = secrets.shape[0] * trials_per_secret
    repeated_secrets = np.repeat(secrets, trials_per_secret, axis=0)
    messages = np.zeros(total_trials, dtype=int)
    decoded_messages = np.zeros(total_trials, dtype=int)
    effective_noise = np.zeros(total_trials, dtype=float)
    noise_margin = np.zeros(total_trials, dtype=float)

    row = 0
    for secret_index, secret in enumerate(secrets):
        vector = public_vectors[:, secret_index]
        m = public_matrix.shape[0]
        for trial_index in range(trials_per_secret):
            rng = seeded_rng(seed, f"correctness:{secret_index}:{trial_index}")
            message = int(rng.integers(0, 2))
            randomness = rng.integers(0, 2, size=m)
            u = (public_matrix.T @ randomness) % q
            v = int((vector @ randomness + message * delta) % q)
            phase = float(centered_mod(np.array(v - int(u @ secret)), q))
            decoded = int(abs(phase) > (q / 4.0))
            eta = float(centered_mod(np.array(phase - message * delta), q))

            messages[row] = message
            decoded_messages[row] = decoded
            effective_noise[row] = eta
            noise_margin[row] = (q / 4.0) - abs(eta)
            row += 1

    return compute_correctness_metrics(
        secrets=repeated_secrets,
        messages=messages,
        decoded_messages=decoded_messages,
        effective_noise=effective_noise,
        noise_margin=noise_margin,
    )


def maybe_compute_reference_metrics(
    secrets: np.ndarray,
    reference_dir: str,
    num_clusters: int,
    seed: int,
):
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


def flatten_row(data: dict) -> dict:
    flattened = {}
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            flattened[key] = json.dumps(value, ensure_ascii=True, sort_keys=True)
        else:
            flattened[key] = value
    return flattened


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    params = pickle.load(open(dataset_dir / "params.pkl", "rb"))
    if not isinstance(params, dict):
        params = params.__dict__

    secrets = np.load(dataset_dir / "secret.npy").T.astype(int)
    clipping_bound = infer_clipping_bound(secrets, args.clipping_bound)
    metrics = {
        "dataset_dir": str(dataset_dir),
        "reference_dir": args.reference_dir or None,
        "secret_type": params.get("secret_type", "binary"),
        "n": int(params["N"]),
        "q": int(params["Q"]),
        "sigma": float(params["sigma"]),
        "num_secret_samples": int(secrets.shape[0]),
        "num_dimensions": int(secrets.shape[1]),
        "num_secret_seeds": int(params.get("num_secret_seeds", secrets.shape[0])),
        "min_hamming": int(params.get("min_hamming", 3)),
        "max_hamming": int(params.get("max_hamming", 3)),
        "clipping_bound_used": int(clipping_bound),
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

    metrics.update(
        compute_correctness_from_dataset(
            dataset_dir=dataset_dir,
            secrets=secrets,
            q=int(params["Q"]),
            seed=args.seed,
            trials_per_secret=args.trials_per_secret,
        )
    )
    metrics.update(
        maybe_compute_reference_metrics(
            secrets=secrets,
            reference_dir=args.reference_dir,
            num_clusters=args.num_clusters,
            seed=args.seed,
        )
    )

    output_json = Path(args.output_json) if args.output_json else dataset_dir / "dataset_metrics.json"
    output_csv = Path(args.output_csv) if args.output_csv else dataset_dir / "dataset_metrics.csv"
    output_json.write_text(json.dumps(metrics, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        row = flatten_row(metrics)
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(f"Wrote dataset metrics to {output_json}")
    print(f"Wrote dataset metrics table to {output_csv}")


if __name__ == "__main__":
    main()
