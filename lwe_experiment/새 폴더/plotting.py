from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .utils import ensure_dir, sanitize_run_name


sns.set_theme(style="whitegrid")


def _save_figure(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return str(path.resolve())


def _sample_rows(matrix: np.ndarray, max_points: int) -> np.ndarray:
    if matrix.shape[0] <= max_points:
        return matrix
    indices = np.linspace(0, matrix.shape[0] - 1, max_points).astype(int)
    return matrix[indices]


def generate_run_figures(
    figures_root: Path,
    model_name: str,
    run_name: str,
    secrets: np.ndarray,
    baseline_reference: np.ndarray,
    effective_noise: np.ndarray,
    probe_truth: np.ndarray,
    probe_prediction: np.ndarray,
    recovery_estimates: np.ndarray,
    pairwise_distances: np.ndarray,
    knn_distances: np.ndarray,
    tsne_max_points: int,
) -> dict[str, str]:
    run_dir = ensure_dir(figures_root / model_name / sanitize_run_name(run_name))
    figure_paths: dict[str, str] = {}

    flattened = secrets.ravel()
    plt.figure(figsize=(6, 4))
    sns.histplot(flattened, bins=min(21, len(np.unique(flattened))), kde=False)
    plt.title(f"Secret Histogram: {run_name}")
    plt.xlabel("Secret value")
    figure_paths["secret_histogram"] = _save_figure(run_dir / "secret_histogram.png")

    n_coords = secrets.shape[1]
    cols = min(4, n_coords)
    rows = int(np.ceil(n_coords / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for coord in range(rows * cols):
        ax = axes[coord // cols][coord % cols]
        if coord < n_coords:
            sns.histplot(secrets[:, coord], bins=min(11, len(np.unique(secrets[:, coord])) + 1), ax=ax)
            ax.set_title(f"coord {coord}")
        else:
            ax.axis("off")
    fig.suptitle(f"Secret Coordinate Histogram: {run_name}")
    figure_paths["secret_coordinate_histogram"] = _save_figure(run_dir / "secret_coordinate_histogram.png")

    corr = np.corrcoef(secrets, rowvar=False)
    corr = np.nan_to_num(np.atleast_2d(corr))
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, cmap="coolwarm", center=0.0, square=True)
    plt.title(f"Correlation Heatmap: {run_name}")
    figure_paths["correlation_heatmap"] = _save_figure(run_dir / "correlation_heatmap.png")

    covariance = np.cov(secrets, rowvar=False)
    covariance = np.atleast_2d(covariance)
    eigvals = np.sort(np.clip(np.linalg.eigvalsh(covariance), 0.0, None))[::-1]
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, eigvals.size + 1), eigvals, marker="o")
    plt.title(f"Covariance Eigen Spectrum: {run_name}")
    plt.xlabel("Eigenvalue rank")
    plt.ylabel("Eigenvalue")
    figure_paths["covariance_eigen_spectrum"] = _save_figure(run_dir / "covariance_eigen_spectrum.png")

    pca = PCA(n_components=2)
    pca_points = pca.fit_transform(secrets.astype(float))
    plt.figure(figsize=(6, 5))
    plt.scatter(pca_points[:, 0], pca_points[:, 1], c=np.linalg.norm(secrets, axis=1), cmap="viridis", s=28)
    plt.title(f"PCA Scatter: {run_name}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    figure_paths["pca_scatter"] = _save_figure(run_dir / "pca_scatter.png")

    tsne_input = _sample_rows(secrets.astype(float), max_points=tsne_max_points)
    perplexity = max(5, min(30, tsne_input.shape[0] - 1))
    if tsne_input.shape[0] >= 6:
        tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity, init="pca")
        tsne_points = tsne.fit_transform(tsne_input)
        plt.figure(figsize=(6, 5))
        plt.scatter(tsne_points[:, 0], tsne_points[:, 1], s=28, alpha=0.85)
        plt.title(f"t-SNE Scatter: {run_name}")
        figure_paths["tsne_scatter"] = _save_figure(run_dir / "tsne_scatter.png")

    plt.figure(figsize=(6, 4))
    sns.histplot(pairwise_distances, bins=30)
    plt.title(f"Pairwise Distance Histogram: {run_name}")
    plt.xlabel("Pairwise L2 distance")
    figure_paths["pairwise_distance_histogram"] = _save_figure(run_dir / "pairwise_distance_histogram.png")

    plt.figure(figsize=(6, 4))
    sns.histplot(knn_distances, bins=20)
    plt.title(f"kNN Distance Histogram: {run_name}")
    plt.xlabel("Nearest-neighbor distance")
    figure_paths["knn_distance_histogram"] = _save_figure(run_dir / "knn_distance_histogram.png")

    overlay_points = np.vstack([baseline_reference.astype(float), secrets.astype(float)])
    overlay_labels = np.array(["baseline"] * baseline_reference.shape[0] + [model_name] * secrets.shape[0])
    overlay_pca = PCA(n_components=2)
    overlay_proj = overlay_pca.fit_transform(overlay_points)
    plt.figure(figsize=(6, 5))
    for label in np.unique(overlay_labels):
        mask = overlay_labels == label
        plt.scatter(
            overlay_proj[mask, 0],
            overlay_proj[mask, 1],
            label=label,
            s=24,
            alpha=0.7,
        )
    plt.legend()
    plt.title(f"Baseline vs Proposed Overlay: {run_name}")
    figure_paths["baseline_vs_proposed_overlay"] = _save_figure(run_dir / "baseline_vs_proposed_overlay.png")

    plt.figure(figsize=(6, 4))
    sns.histplot(effective_noise, bins=24)
    plt.title(f"Decrypt Noise Histogram: {run_name}")
    plt.xlabel("Effective noise")
    figure_paths["decrypt_noise_histogram"] = _save_figure(run_dir / "decrypt_noise_histogram.png")

    sampled_truth = probe_truth.ravel()
    sampled_pred = probe_prediction.ravel()
    sample_count = min(sampled_truth.size, 400)
    idx = np.linspace(0, sampled_truth.size - 1, sample_count).astype(int)
    plt.figure(figsize=(6, 5))
    plt.scatter(sampled_truth[idx], sampled_pred[idx], alpha=0.55, s=20)
    low = min(sampled_truth.min(), sampled_pred.min())
    high = max(sampled_truth.max(), sampled_pred.max())
    plt.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1.0)
    plt.title(f"Probe Prediction vs Ground Truth: {run_name}")
    plt.xlabel("Ground truth secret coordinate")
    plt.ylabel("Predicted coordinate")
    figure_paths["probe_prediction_vs_ground_truth"] = _save_figure(
        run_dir / "probe_prediction_vs_ground_truth.png"
    )

    recovery_errors = np.linalg.norm(recovery_estimates - secrets, axis=1) / np.maximum(
        np.linalg.norm(secrets, axis=1), 1e-8
    )
    plt.figure(figsize=(6, 4))
    sns.histplot(recovery_errors, bins=24)
    plt.title(f"Recovery Error Histogram: {run_name}")
    plt.xlabel("Relative L2 recovery error")
    figure_paths["recovery_error_histogram"] = _save_figure(run_dir / "recovery_error_histogram.png")

    norms = pd.DataFrame(
        {
            "l2_norm": np.linalg.norm(secrets, axis=1),
            "linf_norm": np.abs(secrets).max(axis=1),
        }
    )
    plt.figure(figsize=(6, 4))
    sns.histplot(norms["l2_norm"], color="#1f77b4", bins=20, alpha=0.6, label="L2")
    sns.histplot(norms["linf_norm"], color="#ff7f0e", bins=10, alpha=0.6, label="Linf")
    plt.legend()
    plt.title(f"Secret Norm Distribution: {run_name}")
    figure_paths["secret_norm_distribution"] = _save_figure(run_dir / "secret_norm_distribution.png")

    return figure_paths


def generate_aggregate_figures(
    figures_root: Path,
    master_summary: pd.DataFrame,
) -> dict[str, str]:
    comparison_dir = ensure_dir(figures_root / "comparison")
    paths: dict[str, str] = {}

    proposed = master_summary[master_summary["model_name"].str.startswith("proposed_")].copy()
    if not proposed.empty:
        function_subset = proposed[proposed["num_functions"] > 0]
        if function_subset["num_functions"].nunique() > 1:
            grouped = (
                function_subset.groupby("num_functions", as_index=False)[
                    ["linear_probe_sign_acc", "mlp_probe_auroc"]
                ]
                .mean(numeric_only=True)
                .sort_values("num_functions")
            )
            plt.figure(figsize=(6, 4))
            plt.plot(grouped["num_functions"], grouped["linear_probe_sign_acc"], marker="o", label="linear sign acc")
            plt.plot(grouped["num_functions"], grouped["mlp_probe_auroc"], marker="s", label="mlp auroc")
            plt.title("Function Count vs Leakage")
            plt.xlabel("num_functions")
            plt.ylabel("Leakage metric")
            plt.legend()
            paths["function_count_vs_leakage_plot"] = _save_figure(comparison_dir / "function_count_vs_leakage.png")

        region_subset = proposed[proposed["region_count"] > 0]
        if region_subset["region_count"].nunique() > 1:
            grouped = (
                region_subset.groupby("region_count", as_index=False)[
                    ["linear_probe_sign_acc", "mlp_probe_auroc"]
                ]
                .mean(numeric_only=True)
                .sort_values("region_count")
            )
            plt.figure(figsize=(6, 4))
            plt.plot(grouped["region_count"], grouped["linear_probe_sign_acc"], marker="o", label="linear sign acc")
            plt.plot(grouped["region_count"], grouped["mlp_probe_auroc"], marker="s", label="mlp auroc")
            plt.title("Region Count vs Leakage")
            plt.xlabel("region_count")
            plt.ylabel("Leakage metric")
            plt.legend()
            paths["region_count_vs_leakage_plot"] = _save_figure(comparison_dir / "region_count_vs_leakage.png")

    if not master_summary.empty:
        plot_df = master_summary[
            [
                "model_name",
                "mlp_probe_auroc",
                "secret_recovery_relative_l2_error",
                "decrypt_failure_rate",
            ]
        ].copy()
        long_df = plot_df.melt(id_vars="model_name", var_name="metric", value_name="value")
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=long_df, x="metric", y="value", hue="model_name")
        plt.xticks(rotation=15)
        plt.title("Boxplot Across Runs")
        paths["boxplot_across_runs"] = _save_figure(comparison_dir / "boxplot_across_runs.png")

    return paths
