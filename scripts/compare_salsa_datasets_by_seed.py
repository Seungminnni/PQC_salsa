#!/usr/bin/env python3
import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lwe_experiment.utils import centered_mod, seeded_rng


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two SALSA AB datasets seed-by-seed.")
    parser.add_argument("--baseline_dir", required=True, type=str)
    parser.add_argument("--proposed_dir", required=True, type=str)
    parser.add_argument("--expected_seeds", default=10, type=int)
    parser.add_argument("--trials_per_secret", default=256, type=int)
    parser.add_argument("--seed", default=20260408, type=int)
    parser.add_argument("--output_prefix", default="", type=str)
    return parser.parse_args()


def load_params(dataset_dir: Path):
    params = pickle.load(open(dataset_dir / "params.pkl", "rb"))
    if not isinstance(params, dict):
        params = params.__dict__
    return params


def entropy_of_vector(secret: np.ndarray) -> float:
    values, counts = np.unique(secret.astype(int), return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def compute_correctness_for_secret(orig_a: np.ndarray, orig_b_col: np.ndarray, secret: np.ndarray, q: int, seed: int, trials: int):
    delta = q // 2
    messages = np.zeros(trials, dtype=int)
    decoded_messages = np.zeros(trials, dtype=int)
    effective_noise = np.zeros(trials, dtype=float)
    noise_margin = np.zeros(trials, dtype=float)
    m = orig_a.shape[0]
    for trial in range(trials):
        rng = seeded_rng(seed, f"seed-compare:{trial}")
        message = int(rng.integers(0, 2))
        randomness = rng.integers(0, 2, size=m)
        u = (orig_a.T @ randomness) % q
        v = int((orig_b_col @ randomness + message * delta) % q)
        phase = float(centered_mod(np.array(v - int(u @ secret)), q))
        decoded = int(abs(phase) > (q / 4.0))
        eta = float(centered_mod(np.array(phase - message * delta), q))
        messages[trial] = message
        decoded_messages[trial] = decoded
        effective_noise[trial] = eta
        noise_margin[trial] = (q / 4.0) - abs(eta)
    return {
        "decrypt_success_rate": float(np.mean(messages == decoded_messages)),
        "decrypt_failure_rate": float(np.mean(messages != decoded_messages)),
        "avg_effective_noise": float(np.mean(np.abs(effective_noise))),
        "noise_margin_mean": float(np.mean(noise_margin)),
    }


def per_secret_metrics(dataset_dir: Path, secret_index: int, trials_per_secret: int, base_seed: int):
    params = load_params(dataset_dir)
    secrets = np.load(dataset_dir / "secret.npy").astype(int)
    orig_a = np.load(dataset_dir / "orig_A.npy").astype(int)
    orig_b = np.load(dataset_dir / "orig_b.npy").astype(int)
    secret = secrets[:, secret_index]
    support = secret != 0
    metrics = {
        "nonzero_count": int(np.sum(support)),
        "support_ratio": float(np.mean(support)),
        "l1_norm": float(np.sum(np.abs(secret))),
        "l2_norm": float(np.linalg.norm(secret)),
        "linf_norm": float(np.max(np.abs(secret))),
        "mean": float(np.mean(secret)),
        "std": float(np.std(secret)),
        "abs_mean": float(np.mean(np.abs(secret))),
        "sparsity_ratio": float(np.mean(secret == 0)),
        "entropy": entropy_of_vector(secret),
        "positive_count": int(np.sum(secret > 0)),
        "negative_count": int(np.sum(secret < 0)),
    }
    metrics.update(
        compute_correctness_for_secret(
            orig_a=orig_a,
            orig_b_col=orig_b[:, secret_index],
            secret=secret,
            q=int(params["Q"]),
            seed=base_seed + secret_index * 1009,
            trials=trials_per_secret,
        )
    )
    return metrics, secret


def safe_ratio(numerator: float, denominator: float):
    if abs(denominator) < 1e-12:
        return None
    return float(numerator / denominator)


def build_summary(rows):
    scalar_keys = [
        key for key, value in rows[0].items()
        if key != "seed_index" and isinstance(value, (int, float, np.integer, np.floating, bool))
    ]
    summary = []
    for key in scalar_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        summary.append(
            {
                "metric": key,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return summary


def main():
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    proposed_dir = Path(args.proposed_dir)

    baseline_secret = np.load(baseline_dir / "secret.npy").astype(int)
    proposed_secret = np.load(proposed_dir / "secret.npy").astype(int)
    baseline_seeds = baseline_secret.shape[1]
    proposed_seeds = proposed_secret.shape[1]
    if baseline_seeds < args.expected_seeds or proposed_seeds < args.expected_seeds:
        raise SystemExit(
            f"Expected at least {args.expected_seeds} secret seeds, got baseline={baseline_seeds}, proposed={proposed_seeds}."
        )

    rows = []
    for seed_index in range(args.expected_seeds):
        baseline_metrics, baseline_vector = per_secret_metrics(
            baseline_dir, seed_index, args.trials_per_secret, args.seed
        )
        proposed_metrics, proposed_vector = per_secret_metrics(
            proposed_dir, seed_index, args.trials_per_secret, args.seed + 500000
        )

        baseline_support = baseline_vector != 0
        proposed_support = proposed_vector != 0
        union_support = baseline_support | proposed_support
        intersection_support = baseline_support & proposed_support
        support_union_count = int(np.sum(union_support))
        support_intersection_count = int(np.sum(intersection_support))
        sign_match_shared = (
            float(np.mean(np.sign(baseline_vector[intersection_support]) == np.sign(proposed_vector[intersection_support])))
            if support_intersection_count > 0
            else None
        )

        row = {
            "seed_index": seed_index,
            **{f"baseline_{k}": v for k, v in baseline_metrics.items()},
            **{f"proposed_{k}": v for k, v in proposed_metrics.items()},
            "coord_equal_ratio": float(np.mean(baseline_vector == proposed_vector)),
            "support_equal": bool(np.array_equal(baseline_support, proposed_support)),
            "support_intersection_count": support_intersection_count,
            "support_union_count": support_union_count,
            "support_jaccard": safe_ratio(support_intersection_count, support_union_count),
            "sign_equal_shared_support_ratio": sign_match_shared,
            "delta_l1_norm": float(proposed_metrics["l1_norm"] - baseline_metrics["l1_norm"]),
            "delta_l2_norm": float(proposed_metrics["l2_norm"] - baseline_metrics["l2_norm"]),
            "delta_abs_mean": float(proposed_metrics["abs_mean"] - baseline_metrics["abs_mean"]),
            "delta_entropy": float(proposed_metrics["entropy"] - baseline_metrics["entropy"]),
            "delta_decrypt_success_rate": float(
                proposed_metrics["decrypt_success_rate"] - baseline_metrics["decrypt_success_rate"]
            ),
            "delta_noise_margin_mean": float(
                proposed_metrics["noise_margin_mean"] - baseline_metrics["noise_margin_mean"]
            ),
        }
        rows.append(row)

    output_prefix = (
        Path(args.output_prefix)
        if args.output_prefix
        else proposed_dir / "baseline_ternary_comparison"
    )
    seedwise_csv = output_prefix.with_name(output_prefix.name + "_seedwise.csv")
    seedwise_json = output_prefix.with_name(output_prefix.name + "_seedwise.json")
    summary_csv = output_prefix.with_name(output_prefix.name + "_summary.csv")
    summary_json = output_prefix.with_name(output_prefix.name + "_summary.json")

    with seedwise_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    seedwise_json.write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")

    summary = build_summary(rows)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Wrote seedwise comparison to {seedwise_csv}")
    print(f"Wrote summary comparison to {summary_csv}")


if __name__ == "__main__":
    main()
