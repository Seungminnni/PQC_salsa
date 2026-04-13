from __future__ import annotations

import shutil
import tracemalloc
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig, ModelConfig
from .core import DatasetBundle, generate_dataset, run_decryption_trials
from .metrics import (
    compute_correctness_metrics,
    compute_distribution_distance_metrics,
    compute_recovery_metrics,
    compute_secret_distribution_metrics,
    public_key_classifier_metrics,
    run_secret_probes,
)
from .plotting import generate_aggregate_figures, generate_run_figures
from .utils import ensure_dir, json_dumps


MASTER_COLUMNS = [
    "date",
    "model_name",
    "run_name",
    "seed",
    "n",
    "m",
    "q",
    "noise_sigma",
    "num_samples",
    "num_functions",
    "function_family",
    "region_count",
    "clipping_bound",
    "decrypt_success_rate",
    "decrypt_failure_rate",
    "secret_pairwise_corr_mean_abs",
    "secret_effective_rank",
    "linear_probe_sign_acc",
    "degree2_probe_auroc",
    "mlp_probe_auroc",
    "secret_recovery_relative_l2_error",
    "baseline_vs_proposed_pk_auroc",
    "keygen_time_ms",
    "encrypt_time_ms",
    "decrypt_time_ms",
    "status",
    "notes",
    "generator_variant",
    "region_occupancy_counts_json",
    "region_occupancy_entropy",
    "setup_region_occupancy_counts_json",
    "setup_region_occupancy_entropy",
    "initial_region_balance_counts_json",
    "initial_region_balance_entropy",
    "region_lattice_counts_json",
    "region_lattice_entropy",
    "pre_normalization_cov_diag_mean",
    "pre_normalization_cov_diag_std",
    "pre_normalization_cov_diag_min",
    "pre_normalization_cov_diag_max",
    "post_normalization_cov_diag_mean",
    "post_normalization_cov_diag_std",
    "post_normalization_cov_diag_min",
    "post_normalization_cov_diag_max",
    "pre_whitening_cov_diag_mean",
    "pre_whitening_cov_diag_std",
    "pre_whitening_cov_diag_min",
    "pre_whitening_cov_diag_max",
    "post_whitening_cov_diag_mean",
    "post_whitening_cov_diag_std",
    "post_whitening_cov_diag_min",
    "post_whitening_cov_diag_max",
    "whitening_condition_number",
    "orthogonal_mixing_seed",
    "structure_seed_offset",
    "orthogonal_seed_offset",
    "signed_permutation_seed",
    "quantization_mode",
    "map_scale",
    "offset_scale",
    "alpha",
    "beta",
    "gamma",
    "noise_injection_mode",
    "within_region_sigma",
    "uniform_region_mix",
    "boundary_focus_mix",
    "boundary_temperature",
    "shell_balance_mix",
    "region_sampling_mode",
    "implied_prior_effective_rank",
    "implied_prior_pairwise_corr_mean_abs",
    "implied_prior_abs_mean",
    "implied_prior_boundary_margin_mean",
    "coordinates_generated_independently",
    "coordinate_region_boundaries_json",
    "function_family_per_coordinate_json",
    "coefficient_a_summary_json",
    "coefficient_b_summary_json",
    "coordinate_beta_json",
    "coordinate_clipping_bounds_json",
    "empirical_coordinate_mean_json",
    "empirical_coordinate_variance_json",
    "empirical_coordinate_entropy_json",
    "setup_fraction_clipped_coordinates",
    "average_occupied_quantization_levels",
    "samplewise_occupied_quantization_levels_mean",
    "symmetry_bin_gap_json",
    "setup_coordinate_variance_trace",
    "setup_total_coordinate_draws",
    "fraction_clipped_coordinates",
    "avg_abs_before_quantization",
    "avg_abs_after_quantization",
    "region_boundary_margin_mean",
    "figure_paths_json",
    "aggregate_figure_paths_json",
]


@dataclass(slots=True)
class RunContext:
    model: ModelConfig
    run_index: int
    seed: int
    timestamp: str
    params: dict[str, Any]
    run_name: str


GENERATOR_DIAGNOSTIC_COLUMNS = [
    "generator_variant",
    "region_occupancy_counts_json",
    "region_occupancy_entropy",
    "setup_region_occupancy_counts_json",
    "setup_region_occupancy_entropy",
    "initial_region_balance_counts_json",
    "initial_region_balance_entropy",
    "region_lattice_counts_json",
    "region_lattice_entropy",
    "pre_normalization_cov_diag_mean",
    "pre_normalization_cov_diag_std",
    "pre_normalization_cov_diag_min",
    "pre_normalization_cov_diag_max",
    "post_normalization_cov_diag_mean",
    "post_normalization_cov_diag_std",
    "post_normalization_cov_diag_min",
    "post_normalization_cov_diag_max",
    "pre_whitening_cov_diag_mean",
    "pre_whitening_cov_diag_std",
    "pre_whitening_cov_diag_min",
    "pre_whitening_cov_diag_max",
    "post_whitening_cov_diag_mean",
    "post_whitening_cov_diag_std",
    "post_whitening_cov_diag_min",
    "post_whitening_cov_diag_max",
    "whitening_condition_number",
    "orthogonal_mixing_seed",
    "structure_seed_offset",
    "orthogonal_seed_offset",
    "signed_permutation_seed",
    "quantization_mode",
    "map_scale",
    "offset_scale",
    "alpha",
    "beta",
    "gamma",
    "noise_injection_mode",
    "within_region_sigma",
    "uniform_region_mix",
    "boundary_focus_mix",
    "boundary_temperature",
    "shell_balance_mix",
    "region_sampling_mode",
    "implied_prior_effective_rank",
    "implied_prior_pairwise_corr_mean_abs",
    "implied_prior_abs_mean",
    "implied_prior_boundary_margin_mean",
    "coordinates_generated_independently",
    "coordinate_region_boundaries_json",
    "function_family_per_coordinate_json",
    "coefficient_a_summary_json",
    "coefficient_b_summary_json",
    "coordinate_beta_json",
    "coordinate_clipping_bounds_json",
    "empirical_coordinate_mean_json",
    "empirical_coordinate_variance_json",
    "empirical_coordinate_entropy_json",
    "setup_fraction_clipped_coordinates",
    "average_occupied_quantization_levels",
    "samplewise_occupied_quantization_levels_mean",
    "symmetry_bin_gap_json",
    "setup_coordinate_variance_trace",
    "setup_total_coordinate_draws",
    "fraction_clipped_coordinates",
    "avg_abs_before_quantization",
    "avg_abs_after_quantization",
    "region_boundary_margin_mean",
]


def _ordered_dataframe(rows: list[dict[str, Any]], preferred: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=preferred)
    remaining = [column for column in frame.columns if column not in preferred]
    return frame[preferred + remaining]


def _write_csv_tables(results_dir: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    ensure_dir(results_dir)
    _ordered_dataframe(tables["master_summary"], MASTER_COLUMNS).to_csv(
        results_dir / "master_summary.csv",
        index=False,
    )
    _ordered_dataframe(
        tables["secret_stats"],
        ["date", "model_name", "run_name", "seed", "status", "notes", *GENERATOR_DIAGNOSTIC_COLUMNS, "figure_paths_json"],
    ).to_csv(results_dir / "secret_stats.csv", index=False)
    _ordered_dataframe(
        tables["probe_results"],
        ["date", "model_name", "run_name", "seed", "status", "notes", *GENERATOR_DIAGNOSTIC_COLUMNS, "figure_paths_json"],
    ).to_csv(results_dir / "probe_results.csv", index=False)
    _ordered_dataframe(
        tables["distribution_stats"],
        ["date", "model_name", "run_name", "seed", "status", "notes", *GENERATOR_DIAGNOSTIC_COLUMNS, "figure_paths_json"],
    ).to_csv(results_dir / "distribution_stats.csv", index=False)
    _ordered_dataframe(
        tables["decryption_stats"],
        ["date", "model_name", "run_name", "seed", "status", "notes", *GENERATOR_DIAGNOSTIC_COLUMNS, "figure_paths_json"],
    ).to_csv(results_dir / "decryption_stats.csv", index=False)


def _write_comparison_table(
    results_dir: Path,
    master_rows: list[dict[str, Any]],
    secret_rows: list[dict[str, Any]],
) -> Path:
    master = pd.DataFrame(master_rows)
    secret = pd.DataFrame(secret_rows)
    if master.empty or secret.empty:
        path = results_dir / "comparison_table.csv"
        pd.DataFrame(
            columns=["metric", "baseline_mean", "proposed_mean", "delta_proposed_minus_baseline", "goal", "success"]
        ).to_csv(path, index=False)
        return path

    merged = master.merge(
        secret[
            [
                "run_name",
                "model_name",
                "secret_abs_mean",
                "secret_symmetry_gap",
                "secret_sparsity_ratio",
                "secret_entropy",
            ]
        ],
        on=["run_name", "model_name"],
        how="left",
    )
    merged = merged[merged["status"] == "success"].copy()

    grouped = merged.groupby("model_name").mean(numeric_only=True)
    proposed_names = [name for name in grouped.index.tolist() if name != "baseline_lwe"]
    if "baseline_lwe" not in grouped.index or len(proposed_names) != 1:
        path = results_dir / "comparison_table.csv"
        pd.DataFrame(
            columns=["metric", "baseline_mean", "proposed_mean", "delta_proposed_minus_baseline", "goal", "success"]
        ).to_csv(path, index=False)
        return path

    baseline = grouped.loc["baseline_lwe"]
    proposed = grouped.loc[proposed_names[0]]
    specs = [
        ("decrypt_success_rate", ">=", "preserve"),
        ("secret_pairwise_corr_mean_abs", "<=ratio1.25", "not_excessively_inflated"),
        ("secret_effective_rank", ">=delta0.35", "no_rank_collapse"),
        ("secret_entropy", ">=delta0.35", "keep_entropy"),
        ("linear_probe_sign_acc", "<=delta0.05", "avoid_large_leakage_increase"),
        ("degree2_probe_auroc", "<=delta0.03", "avoid_large_leakage_increase"),
        ("mlp_probe_auroc", "<=delta0.03", "avoid_large_leakage_increase"),
        ("secret_recovery_relative_l2_error", ">=delta0.08", "avoid_easier_recovery"),
        ("secret_abs_mean", "~=0.15", "close_to_baseline"),
        ("secret_symmetry_gap", "<=delta0.05", "bounded_symmetric"),
        ("secret_sparsity_ratio", "<=delta0.12", "avoid_collapse"),
    ]

    rows: list[dict[str, Any]] = []
    for metric, comparator, goal in specs:
        baseline_value = float(baseline.get(metric, np.nan))
        proposed_value = float(proposed.get(metric, np.nan))
        delta = proposed_value - baseline_value
        if comparator == ">=":
            success = proposed_value >= baseline_value - 0.01
        elif comparator == "<=ratio1.25":
            success = proposed_value <= baseline_value * 1.25
        elif comparator == ">=delta0.35":
            success = proposed_value >= baseline_value - 0.35
        elif comparator == "<=delta0.05":
            success = proposed_value <= baseline_value + 0.05
        elif comparator == "<=delta0.03":
            success = proposed_value <= baseline_value + 0.03
        elif comparator == ">=delta0.08":
            success = proposed_value >= baseline_value - 0.08
        elif comparator == "~=0.15":
            success = abs(proposed_value - baseline_value) <= 0.15
        elif comparator == "<=delta0.12":
            success = proposed_value <= baseline_value + 0.12
        else:
            success = proposed_value <= baseline_value + 0.05
        rows.append(
            {
                "metric": metric,
                "baseline_mean": baseline_value,
                "proposed_mean": proposed_value,
                "delta_proposed_minus_baseline": delta,
                "goal": goal,
                "success": bool(success),
            }
        )

    path = results_dir / "comparison_table.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _reset_output_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _base_row(context: RunContext, config: ExperimentConfig) -> dict[str, Any]:
    return {
        "date": context.timestamp,
        "model_name": context.model.name,
        "run_name": context.run_name,
        "seed": context.seed,
        "n": int(config.global_params["n"]),
        "m": int(config.global_params["m"]),
        "q": int(config.global_params["q"]),
        "noise_sigma": float(config.global_params["noise_sigma"]),
        "num_samples": int(config.global_params["num_samples"]),
        "num_functions": int(context.params.get("num_functions", 0)),
        "function_family": str(context.params.get("function_family", "")),
        "region_count": int(context.params.get("region_count", 0)),
        "clipping_bound": int(context.params.get("clipping_bound", 0)),
    }


def _empty_generator_diagnostics() -> dict[str, Any]:
    return {column: None for column in GENERATOR_DIAGNOSTIC_COLUMNS}


def _entropy_base2(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _collect_generator_diagnostics(
    dataset: DatasetBundle | None,
    context: RunContext,
) -> dict[str, Any]:
    diagnostics = _empty_generator_diagnostics()
    if dataset is None:
        return diagnostics

    run_diag = dict(dataset.generator_run_diagnostics)
    diagnostics.update({key: run_diag.get(key, diagnostics[key]) for key in diagnostics})
    sample_diag = dataset.generator_sample_diagnostics

    region_count_vectors = sample_diag.get("region_counts_vector")
    if region_count_vectors is not None and region_count_vectors.size > 0:
        stacked = np.stack([np.asarray(row, dtype=int) for row in region_count_vectors], axis=0)
        counts = np.sum(stacked, axis=0)
        diagnostics["region_occupancy_counts_json"] = json_dumps(counts.tolist())
        diagnostics["region_occupancy_entropy"] = _entropy_base2(counts)
    else:
        region_indices = sample_diag.get("region_index")
        if region_indices is not None and region_indices.size > 0:
            minlength = max(int(context.params.get("region_count", 0)), int(np.max(region_indices)) + 1)
            counts = np.bincount(region_indices.astype(int), minlength=minlength)
            diagnostics["region_occupancy_counts_json"] = json_dumps(counts.tolist())
            diagnostics["region_occupancy_entropy"] = _entropy_base2(counts)

    if "fraction_clipped_coordinates" in sample_diag:
        diagnostics["fraction_clipped_coordinates"] = float(
            np.mean(sample_diag["fraction_clipped_coordinates"].astype(float))
        )
    if "avg_abs_before_quantization" in sample_diag:
        diagnostics["avg_abs_before_quantization"] = float(
            np.mean(sample_diag["avg_abs_before_quantization"].astype(float))
        )
    if "avg_abs_after_quantization" in sample_diag:
        diagnostics["avg_abs_after_quantization"] = float(
            np.mean(sample_diag["avg_abs_after_quantization"].astype(float))
        )
    if "occupied_quantization_levels" in sample_diag:
        diagnostics["samplewise_occupied_quantization_levels_mean"] = float(
            np.mean(sample_diag["occupied_quantization_levels"].astype(float))
        )
    if "region_boundary_margin" in sample_diag:
        diagnostics["region_boundary_margin_mean"] = float(np.mean(sample_diag["region_boundary_margin"].astype(float)))

    return diagnostics


def _failure_rows(
    context: RunContext,
    config: ExperimentConfig,
    message: str,
) -> dict[str, dict[str, Any]]:
    base = _base_row(context, config)
    common = {
        **base,
        "status": "failed",
        "notes": message,
        **_empty_generator_diagnostics(),
        "figure_paths_json": json_dumps({}),
    }
    master = {
        **common,
        "decrypt_success_rate": None,
        "decrypt_failure_rate": None,
        "secret_pairwise_corr_mean_abs": None,
        "secret_effective_rank": None,
        "linear_probe_sign_acc": None,
        "degree2_probe_auroc": None,
        "mlp_probe_auroc": None,
        "secret_recovery_relative_l2_error": None,
        "baseline_vs_proposed_pk_auroc": None,
        "keygen_time_ms": None,
        "encrypt_time_ms": None,
        "decrypt_time_ms": None,
        "aggregate_figure_paths_json": json_dumps({}),
    }
    return {
        "master_summary": master,
        "secret_stats": dict(common),
        "probe_results": dict(common),
        "distribution_stats": dict(common),
        "decryption_stats": dict(common),
    }


def _copy_aggregate_figures(
    figures_root: Path,
    aggregate_paths: dict[str, str],
    model_names: list[str],
) -> dict[str, dict[str, str]]:
    per_model_paths: dict[str, dict[str, str]] = {}
    for model_name in model_names:
        model_dir = ensure_dir(figures_root / model_name)
        copied: dict[str, str] = {}
        for key, source in aggregate_paths.items():
            target = model_dir / Path(source).name
            shutil.copy2(source, target)
            copied[key] = str(target.resolve())
        per_model_paths[model_name] = copied
    return per_model_paths


def _run_single(
    context: RunContext,
    config: ExperimentConfig,
    baseline_dataset: DatasetBundle | None,
) -> tuple[dict[str, dict[str, Any]], DatasetBundle | None]:
    figures_root = Path(config.global_params["figures_dir"])
    n = int(config.global_params["n"])
    m = int(config.global_params["m"])
    q = int(config.global_params["q"])
    noise_sigma = float(config.global_params["noise_sigma"])
    num_samples = int(config.global_params["num_samples"])
    resolved_params = dict(context.params)
    resolved_notes = ""

    tracemalloc.start()
    try:
        dataset = generate_dataset(
            n=n,
            m=m,
            q=q,
            noise_sigma=noise_sigma,
            num_samples=num_samples,
            run_seed=context.seed,
            generator_type=context.model.generator_type,
            generator_params=resolved_params,
            recovery_passes=int(config.global_params["recovery_refinement_passes"]),
            exhaustive_recovery_limit=int(config.global_params["exhaustive_recovery_limit"]),
            model_name=context.model.name,
        )
        decryption = run_decryption_trials(
            public_matrices=dataset.public_matrices,
            public_vectors=dataset.public_vectors,
            secrets=dataset.secrets,
            q=q,
            run_seed=context.seed,
        )

        reference_dataset = dataset if baseline_dataset is None else baseline_dataset
        correctness_metrics = compute_correctness_metrics(
            secrets=dataset.secrets,
            messages=decryption.messages,
            decoded_messages=decryption.decoded_messages,
            effective_noise=decryption.effective_noise,
            noise_margin=decryption.noise_margin,
        )
        secret_metrics = compute_secret_distribution_metrics(
            secrets=dataset.secrets,
            clipping_bound=int(resolved_params["clipping_bound"]),
        )
        probe_metrics, probe_artifacts = run_secret_probes(
            public_features=dataset.public_features,
            secrets=dataset.secrets,
            seed=context.seed,
            train_fraction=float(config.global_params["train_fraction"]),
            hidden_layers=tuple(config.global_params["probe_hidden_layers"]),
            max_iter=int(config.global_params["probe_max_iter"]),
        )
        recovery_metrics = compute_recovery_metrics(
            true_secrets=dataset.secrets,
            recovered_secrets=dataset.recovery_estimates,
            clipping_bound=int(resolved_params["clipping_bound"]),
        )
        pk_metrics = public_key_classifier_metrics(
            baseline_features=reference_dataset.public_features,
            current_features=dataset.public_features,
            seed=context.seed,
        )
        distribution_metrics, distribution_artifacts = compute_distribution_distance_metrics(
            secrets=dataset.secrets,
            baseline_reference=reference_dataset.secrets,
            num_clusters=int(config.global_params["num_clusters"]),
            seed=context.seed,
        )

        _, peak_bytes = tracemalloc.get_traced_memory()
        memory_usage_mb = peak_bytes / (1024 * 1024)

        run_figure_paths = generate_run_figures(
            figures_root=figures_root,
            model_name=context.model.name,
            run_name=context.run_name,
            secrets=dataset.secrets,
            baseline_reference=reference_dataset.secrets,
            effective_noise=decryption.effective_noise,
            probe_truth=probe_artifacts.y_true,
            probe_prediction=probe_artifacts.mlp_pred,
            recovery_estimates=dataset.recovery_estimates,
            pairwise_distances=distribution_artifacts["pairwise_distances"],
            knn_distances=distribution_artifacts["knn_distances"],
            tsne_max_points=int(config.global_params["tsne_max_points"]),
        )

        base = _base_row(context, config)
        tuned_context = RunContext(
            model=context.model,
            run_index=context.run_index,
            seed=context.seed,
            timestamp=context.timestamp,
            params=resolved_params,
            run_name=context.run_name,
        )
        generator_diagnostics = _collect_generator_diagnostics(dataset=dataset, context=tuned_context)
        common = {
            **base,
            "status": "success",
            "notes": resolved_notes,
            **generator_diagnostics,
            "figure_paths_json": json_dumps(run_figure_paths),
        }

        master = {
            **common,
            **correctness_metrics,
            **probe_metrics,
            **recovery_metrics,
            **pk_metrics,
            "secret_pairwise_corr_mean_abs": secret_metrics["secret_pairwise_corr_mean_abs"],
            "secret_effective_rank": secret_metrics["secret_effective_rank"],
            "keygen_time_ms": float(dataset.keygen_time_ms),
            "encrypt_time_ms": float(decryption.encrypt_time_ms),
            "decrypt_time_ms": float(decryption.decrypt_time_ms),
            "probe_train_time_ms": float(probe_artifacts.probe_train_time_ms),
            "memory_usage_mb": float(memory_usage_mb),
            "aggregate_figure_paths_json": json_dumps({}),
        }

        secret_row = {
            **common,
            **secret_metrics,
            "memory_usage_mb": float(memory_usage_mb),
        }
        probe_row = {
            **common,
            **probe_metrics,
            **recovery_metrics,
            **pk_metrics,
            "probe_train_time_ms": float(probe_artifacts.probe_train_time_ms),
            "memory_usage_mb": float(memory_usage_mb),
        }
        distribution_row = {
            **common,
            **distribution_metrics,
            "memory_usage_mb": float(memory_usage_mb),
        }
        decryption_row = {
            **common,
            **correctness_metrics,
            "keygen_time_ms": float(dataset.keygen_time_ms),
            "encrypt_time_ms": float(decryption.encrypt_time_ms),
            "decrypt_time_ms": float(decryption.decrypt_time_ms),
            "memory_usage_mb": float(memory_usage_mb),
        }

        return (
            {
                "master_summary": master,
                "secret_stats": secret_row,
                "probe_results": probe_row,
                "distribution_stats": distribution_row,
                "decryption_stats": decryption_row,
            },
            dataset if context.model.name == "baseline_lwe" else None,
        )
    except Exception as exc:
        return _failure_rows(context, config, str(exc)), None
    finally:
        tracemalloc.stop()


def run_experiment_suite(config: ExperimentConfig) -> dict[str, Any]:
    config.validate()
    results_dir = _reset_output_dir(Path(config.global_params["results_dir"]))
    figures_root = _reset_output_dir(Path(config.global_params["figures_dir"]))

    tables: dict[str, list[dict[str, Any]]] = {
        "master_summary": [],
        "secret_stats": [],
        "probe_results": [],
        "distribution_stats": [],
        "decryption_stats": [],
    }

    baseline_model = next(model for model in config.models if model.name == "baseline_lwe")
    proposed_models = [model for model in config.models if model.name != "baseline_lwe"]
    ordered_models = [baseline_model] + proposed_models

    for run_index, seed in enumerate(config.run_seeds, start=1):
        baseline_dataset: DatasetBundle | None = None
        for model in ordered_models:
            params = model.resolved_params(config.baseline_params, config.proposed_base_params)
            context = RunContext(
                model=model,
                run_index=run_index,
                seed=seed,
                timestamp=datetime.now().isoformat(timespec="seconds"),
                params=params,
                run_name=f"{model.name}_run_{run_index}",
            )
            if model.name != "baseline_lwe" and baseline_dataset is None:
                run_rows = _failure_rows(
                    context=context,
                    config=config,
                    message="baseline_lwe failed for this paired seed, so proposed comparison was skipped.",
                )
                for table_name, row in run_rows.items():
                    tables[table_name].append(row)
                _write_csv_tables(results_dir, tables)
                continue
            run_rows, produced_baseline = _run_single(
                context=context,
                config=config,
                baseline_dataset=baseline_dataset,
            )
            for table_name, row in run_rows.items():
                tables[table_name].append(row)
            if produced_baseline is not None:
                baseline_dataset = produced_baseline
            _write_csv_tables(results_dir, tables)

    master_frame = pd.DataFrame(tables["master_summary"])
    aggregate_paths = generate_aggregate_figures(figures_root=figures_root, master_summary=master_frame)
    per_model_aggregate_paths = _copy_aggregate_figures(
        figures_root=figures_root,
        aggregate_paths=aggregate_paths,
        model_names=sorted(master_frame["model_name"].dropna().unique().tolist()),
    )

    for table_name, rows in tables.items():
        for row in rows:
            row["aggregate_figure_paths_json"] = json_dumps(
                per_model_aggregate_paths.get(row["model_name"], {})
            )

    _write_csv_tables(results_dir, tables)
    comparison_table_path = _write_comparison_table(
        results_dir=results_dir,
        master_rows=tables["master_summary"],
        secret_rows=tables["secret_stats"],
    )
    return {
        "results_dir": str(results_dir.resolve()),
        "figures_dir": str(figures_root.resolve()),
        "master_summary_path": str((results_dir / "master_summary.csv").resolve()),
        "comparison_table_path": str(comparison_table_path.resolve()),
    }
