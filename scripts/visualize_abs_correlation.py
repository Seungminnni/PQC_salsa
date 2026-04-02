#!/usr/bin/env python3
"""
Visualize the A-b-s correlation structure for reduced LWE samples.

This script is designed for small-q diagnostic experiments where we want to
inspect:
1. coordinate-wise A/b dependence,
2. signal vs observed b after centered lifting, and
3. a 3D manifold of per-coordinate contributions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MPLCONFIGDIR = Path("/tmp/matplotlib-codex")
DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR))

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


@dataclass
class SeedView:
    seed: int
    secret: np.ndarray
    support: np.ndarray
    a_rows: np.ndarray
    b: np.ndarray
    a_lift: np.ndarray
    b_lift: np.ndarray
    signal: np.ndarray
    residual: np.ndarray
    contributions: np.ndarray
    mi_scores: np.ndarray
    corr_scores: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize A-b-s correlations.")
    parser.add_argument("--train-prefix", type=Path, required=True, help="Path to train.prefix.")
    parser.add_argument("--secret-npy", type=Path, required=True, help="Path to secret.npy.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for plots and CSV summaries.")
    parser.add_argument("--q", type=int, required=True, help="Modulus q.")
    parser.add_argument("--seed", type=int, default=1, help="Primary seed to visualize in detail.")
    parser.add_argument("--all-seeds", action="store_true", help="Also generate overview plots for every seed in secret.npy.")
    parser.add_argument("--scatter-samples", type=int, default=3500, help="Maximum points in 2D scatter plots.")
    parser.add_argument("--manifold-samples", type=int, default=1200, help="Maximum points in the 3D manifold plot.")
    parser.add_argument("--edge-samples", type=int, default=180, help="Maximum points used for kNN edges in the 3D manifold.")
    parser.add_argument("--knn-k", type=int, default=4, help="Number of nearest neighbors when drawing light manifold edges.")
    parser.add_argument("--dpi", type=int, default=180, help="Output image DPI.")
    parser.add_argument("--three-d-only", action="store_true", help="Also save a standalone 3D manifold figure.")
    parser.add_argument(
        "--three-d-color",
        type=str,
        choices=["mono", "observed", "residual", "signal"],
        default="mono",
        help="Quantity used to color points in the standalone 3D manifold.",
    )
    parser.add_argument(
        "--three-d-size",
        type=str,
        choices=["residual_abs", "uniform"],
        default="uniform",
        help="Point sizing rule in the standalone 3D manifold.",
    )
    parser.add_argument("--view-elev", type=float, default=24.0, help="Elevation angle for 3D views.")
    parser.add_argument("--view-azim", type=float, default=38.0, help="Azimuth angle for 3D views.")
    return parser.parse_args()


def centered_lift(values: np.ndarray, q: int) -> np.ndarray:
    half = q // 2
    return ((values + half) % q) - half


def load_prefix(prefix_path: Path) -> tuple[np.ndarray, np.ndarray]:
    a_rows = []
    b_rows = []
    with prefix_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            lhs, rhs = line.split(";")
            a_rows.append(np.fromstring(lhs, sep=" ", dtype=np.int64))
            b_rows.append(np.fromstring(rhs, sep=" ", dtype=np.int64))
    if not a_rows:
        raise ValueError(f"No samples found in {prefix_path}")
    return np.vstack(a_rows), np.vstack(b_rows)


def safe_abs_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(abs(np.corrcoef(x, y)[0, 1]))


def discrete_mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    x_vals, x_inv = np.unique(x, return_inverse=True)
    y_vals, y_inv = np.unique(y, return_inverse=True)
    joint = np.zeros((len(x_vals), len(y_vals)), dtype=np.float64)
    np.add.at(joint, (x_inv, y_inv), 1.0)
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    denom = px * py
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log2(joint[mask] / denom[mask])))


def sample_indices(count: int, limit: int, rng: np.random.Generator) -> np.ndarray:
    if limit <= 0 or count <= limit:
        return np.arange(count)
    return np.sort(rng.choice(count, size=limit, replace=False))


def pca_3d(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = values - values.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:3].T
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])), mode="constant")
    explained = singular_values ** 2
    if explained.sum() == 0:
        ratio = np.zeros(3)
    else:
        ratio = explained / explained.sum()
    if ratio.shape[0] < 3:
        ratio = np.pad(ratio[:3], (0, 3 - ratio.shape[0]), mode="constant")
    else:
        ratio = ratio[:3]
    return coords[:, :3], ratio


def centered_axis_limit(points: np.ndarray, floor: float = 1.0) -> float:
    if points.size == 0:
        return floor
    radius = float(np.max(np.abs(points)))
    if not np.isfinite(radius) or radius == 0.0:
        return floor
    return max(floor, radius * 1.1)


def normalize_3d_points(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    scale = np.max(np.abs(points))
    if not np.isfinite(scale) or scale == 0.0:
        return points.copy()
    return points / scale


def knn_edge_bundle(points: np.ndarray, k: int) -> list[tuple[int, int, float]]:
    if len(points) <= 1:
        return []
    sq_norms = np.sum(points ** 2, axis=1, keepdims=True)
    distances = sq_norms + sq_norms.T - 2 * points @ points.T
    np.fill_diagonal(distances, np.inf)
    edges: dict[tuple[int, int], float] = {}
    for idx in range(len(points)):
        neighbors = np.argsort(distances[idx])[:k]
        neighbor_distances = distances[idx, neighbors]
        finite_mask = np.isfinite(neighbor_distances)
        if not np.any(finite_mask):
            continue
        local_max = float(np.max(neighbor_distances[finite_mask]))
        if local_max <= 0:
            local_max = 1.0
        for neighbor in neighbors:
            distance = float(distances[idx, neighbor])
            closeness = max(0.0, 1.0 - distance / local_max)
            edge = tuple(sorted((idx, int(neighbor))))
            edges[edge] = max(edges.get(edge, 0.0), closeness)
    return [(start, end, strength) for (start, end), strength in sorted(edges.items())]


def build_seed_view(a_rows: np.ndarray, b_rows: np.ndarray, secrets: np.ndarray, seed: int, q: int) -> SeedView:
    secret = secrets[:, seed].astype(np.int64)
    support = (secret != 0).astype(np.int64)
    b = b_rows[:, seed].astype(np.int64)
    a_lift = centered_lift(a_rows, q)
    b_lift = centered_lift(b, q)
    signal = centered_lift((a_rows @ secret) % q, q)
    residual = centered_lift((b - (a_rows @ secret) % q) % q, q)
    contributions = centered_lift((a_rows * secret.reshape(1, -1)) % q, q)
    mi_scores = np.array([discrete_mutual_information(a_lift[:, col], b_lift) for col in range(a_rows.shape[1])])
    corr_scores = np.array([safe_abs_corr(a_lift[:, col], b_lift) for col in range(a_rows.shape[1])])
    return SeedView(
        seed=seed,
        secret=secret,
        support=support,
        a_rows=a_rows,
        b=b,
        a_lift=a_lift,
        b_lift=b_lift,
        signal=signal,
        residual=residual,
        contributions=contributions,
        mi_scores=mi_scores,
        corr_scores=corr_scores,
    )


def add_heatmap_panel(fig: plt.Figure, subplot_spec, view: SeedView, q: int) -> None:
    half = q // 2
    ticks = np.arange(-half, half + 1)
    grid = subplot_spec.subgridspec(1, len(view.secret), wspace=0.25)
    vmax = 0.0
    heatmaps = []
    for col in range(len(view.secret)):
        heat = np.zeros((q, q), dtype=np.float64)
        x_idx = view.a_lift[:, col] + half
        y_idx = view.b_lift + half
        np.add.at(heat, (y_idx, x_idx), 1.0)
        heat /= heat.sum()
        heatmaps.append(heat)
        vmax = max(vmax, float(heat.max()))
    for col, heat in enumerate(heatmaps):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(
            heat,
            origin="lower",
            cmap="Blues",
            vmin=0.0,
            vmax=vmax,
            extent=(-half - 0.5, half + 0.5, -half - 0.5, half + 0.5),
            aspect="equal",
        )
        title_color = "tab:blue" if view.support[col] else "black"
        ax.set_title(
            f"j={col}\ns={int(view.secret[col])}, supp={int(view.support[col])}\n"
            f"MI={view.mi_scores[col]:.3f}, |r|={view.corr_scores[col]:.3f}",
            fontsize=9,
            color=title_color,
        )
        ax.set_xticks(ticks[::2])
        ax.set_yticks(ticks[::2])
        ax.set_xlabel("L_q(a_ij)", fontsize=9)
        if col == 0:
            ax.set_ylabel("L_q(b_i)", fontsize=9)
        else:
            ax.set_yticklabels([])


def add_signal_panel(fig: plt.Figure, scatter_spec, hist_spec, view: SeedView, q: int, rng: np.random.Generator, scatter_samples: int) -> None:
    half = q // 2
    scatter_idx = sample_indices(len(view.signal), scatter_samples, rng)
    ax_scatter = fig.add_subplot(scatter_spec)
    ax_hist = fig.add_subplot(hist_spec)

    colors = view.residual[scatter_idx]
    sizes = 16 + 4 * np.abs(colors)
    scatter = ax_scatter.scatter(
        view.signal[scatter_idx],
        view.b_lift[scatter_idx],
        c=colors,
        s=sizes,
        cmap="coolwarm",
        vmin=-half,
        vmax=half,
        alpha=0.65,
        linewidths=0,
    )
    ax_scatter.plot([-half, half], [-half, half], linestyle="--", color="black", linewidth=1.0, alpha=0.7)
    ax_scatter.set_xlim(-half - 0.5, half + 0.5)
    ax_scatter.set_ylim(-half - 0.5, half + 0.5)
    ax_scatter.set_xlabel("z_i = L_q(a_i^T s mod q)")
    ax_scatter.set_ylabel(r"$L_q(b_i)$")
    ax_scatter.set_title("Signal vs observed b")
    cbar = fig.colorbar(scatter, ax=ax_scatter, fraction=0.046, pad=0.04)
    cbar.set_label(r"$e_i = L_q(b_i - a_i^T s)$", rotation=90)

    bins = np.arange(-half - 0.5, half + 1.5, 1.0)
    ax_hist.hist(view.residual, bins=bins, orientation="horizontal", color="tab:orange", alpha=0.8)
    ax_hist.set_ylim(-half - 0.5, half + 0.5)
    ax_hist.set_xlabel("count")
    ax_hist.set_title("Residuals")
    ax_hist.set_ylabel(r"$e_i$")


def add_manifold_panel(
    fig: plt.Figure,
    subplot_spec,
    view: SeedView,
    rng: np.random.Generator,
    manifold_samples: int,
    edge_samples: int,
    knn_k: int,
    color_mode: str,
    size_mode: str,
    view_elev: float,
    view_azim: float,
) -> None:
    manifold_idx = sample_indices(len(view.contributions), manifold_samples, rng)
    contributions = view.contributions[manifold_idx]
    observed = view.b_lift[manifold_idx]
    residual = view.residual[manifold_idx]
    signal = view.signal[manifold_idx]
    coords, explained_ratio = pca_3d(contributions)
    coords = normalize_3d_points(coords)
    ax = fig.add_subplot(subplot_spec, projection="3d")

    dense_cloud = color_mode == "mono"
    edge_limit = min(len(coords), edge_samples if not dense_cloud else max(60, min(edge_samples, 96)))
    edge_idx = sample_indices(len(coords), edge_limit, rng)
    edge_coords = coords[edge_idx]
    edge_k = knn_k if not dense_cloud else max(14, knn_k * 4)
    for start, end, strength in knn_edge_bundle(edge_coords, edge_k):
        xs = edge_coords[[start, end], 0]
        ys = edge_coords[[start, end], 1]
        zs = edge_coords[[start, end], 2]
        ax.plot(
            xs,
            ys,
            zs,
            color="#9fbfda",
            alpha=0.05 + 0.08 * strength if dense_cloud else 0.08 + 0.10 * strength,
            linewidth=0.55 + 0.25 * strength if dense_cloud else 0.45 + 0.20 * strength,
            solid_capstyle="round",
            zorder=1,
        )

    if color_mode == "mono":
        colors = "#73d2f6"
        cmap = None
        cbar_label = None
    elif color_mode == "residual":
        colors = residual
        cmap = "coolwarm"
        cbar_label = r"$e_i$"
    elif color_mode == "signal":
        colors = signal
        cmap = "cividis"
        cbar_label = r"$z_i$"
    else:
        colors = observed
        cmap = "viridis"
        cbar_label = r"$L_q(b_i)$"

    if size_mode == "uniform":
        sizes = np.full(len(coords), 36.0)
    else:
        sizes = 18 + 5 * np.abs(residual)

    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=colors,
        s=sizes,
        cmap=cmap,
        alpha=0.84 if dense_cloud else 0.72,
        edgecolors="#2b7a94",
        linewidths=0.85,
        depthshade=False,
        zorder=3,
    )
    centroid = coords.mean(axis=0)
    ax.scatter(
        [centroid[0]],
        [centroid[1]],
        [centroid[2]],
        marker="*",
        s=220,
        color="crimson",
        edgecolor="black",
        linewidth=0.8,
    )
    axis_limit = centered_axis_limit(coords, floor=1.0)
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_zlim(-axis_limit, axis_limit)
    ax.plot([-axis_limit, axis_limit], [0, 0], [0, 0], color="black", linewidth=1.2, alpha=0.9)
    ax.plot([0, 0], [-axis_limit, axis_limit], [0, 0], color="black", linewidth=1.2, alpha=0.9)
    ax.plot([0, 0], [0, 0], [-axis_limit, axis_limit], color="black", linewidth=1.2, alpha=0.9)
    ax.set_xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}%)")
    ax.set_zlabel(f"PC3 ({explained_ratio[2] * 100:.1f}%)")
    ax.set_title("3D contribution manifold")
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.grid(True, alpha=0.22)
    ax.xaxis.pane.set_facecolor((0.95, 0.97, 1.0, 0.35))
    ax.yaxis.pane.set_facecolor((0.95, 0.97, 1.0, 0.35))
    ax.zaxis.pane.set_facecolor((0.95, 0.97, 1.0, 0.35))
    ticks = [-1.0, 0.0, 1.0]
    tick_labels = [f"{tick * axis_limit:.3f}" for tick in ticks]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_zticklabels(tick_labels)
    if cbar_label is not None:
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.08)
        cbar.set_label(cbar_label, rotation=90)


def save_seed_summary(
    view: SeedView,
    out_dir: Path,
    q: int,
    scatter_samples: int,
    manifold_samples: int,
    edge_samples: int,
    knn_k: int,
    dpi: int,
    color_mode: str,
    size_mode: str,
    view_elev: float,
    view_azim: float,
) -> None:
    rng = np.random.default_rng(1000 + view.seed)
    fig = plt.figure(figsize=(20, 11))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], width_ratios=[1.45, 0.55, 1.3])
    add_heatmap_panel(fig, grid[0, :], view, q)
    add_signal_panel(fig, grid[1, 0], grid[1, 1], view, q, rng, scatter_samples)
    add_manifold_panel(
        fig,
        grid[1, 2],
        view,
        rng,
        manifold_samples,
        edge_samples,
        knn_k,
        color_mode,
        size_mode,
        view_elev,
        view_azim,
    )
    fig.suptitle(
        f"A-b-s correlation summary | seed {view.seed} | "
        f"secret={view.secret.tolist()} | support={view.support.tolist()}",
        fontsize=14,
        y=0.98,
    )
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.06, top=0.92, wspace=0.30, hspace=0.26)
    fig.savefig(out_dir / f"seed_{view.seed}_oracle_summary.png", dpi=dpi)
    plt.close(fig)


def save_seed_3d_only(
    view: SeedView,
    out_dir: Path,
    manifold_samples: int,
    edge_samples: int,
    knn_k: int,
    dpi: int,
    color_mode: str,
    size_mode: str,
    view_elev: float,
    view_azim: float,
) -> None:
    rng = np.random.default_rng(7000 + view.seed)
    fig = plt.figure(figsize=(9, 8))
    ax_spec = fig.add_gridspec(1, 1)[0, 0]
    add_manifold_panel(
        fig,
        ax_spec,
        view,
        rng,
        manifold_samples,
        edge_samples,
        knn_k,
        color_mode,
        size_mode,
        view_elev,
        view_azim,
    )
    fig.suptitle(
        f"Seed {view.seed} 3D contribution manifold | "
        f"secret={view.secret.tolist()} | support={view.support.tolist()}",
        fontsize=12,
        y=0.97,
    )
    fig.subplots_adjust(left=0.02, right=0.93, bottom=0.02, top=0.92)
    fig.savefig(out_dir / f"seed_{view.seed}_3d_only.png", dpi=dpi)
    plt.close(fig)


def save_all_seed_signal_grid(views: list[SeedView], out_dir: Path, q: int, scatter_samples: int, dpi: int) -> None:
    rows = 2
    cols = 3
    half = q // 2
    fig, axes = plt.subplots(rows, cols, figsize=(16, 9), sharex=True, sharey=True)
    rng = np.random.default_rng(4242)
    for ax, view in zip(axes.flat, views):
        idx = sample_indices(len(view.signal), scatter_samples, rng)
        scatter = ax.scatter(
            view.signal[idx],
            view.b_lift[idx],
            c=view.residual[idx],
            s=10 + 2 * np.abs(view.residual[idx]),
            cmap="coolwarm",
            vmin=-half,
            vmax=half,
            alpha=0.5,
            linewidths=0,
        )
        ax.plot([-half, half], [-half, half], linestyle="--", color="black", linewidth=0.8, alpha=0.7)
        ax.set_title(
            f"seed {view.seed}\nsecret={view.secret.tolist()}\nsupport={view.support.tolist()}",
            fontsize=9,
        )
        ax.set_xlim(-half - 0.5, half + 0.5)
        ax.set_ylim(-half - 0.5, half + 0.5)
        ax.set_xlabel(r"$L_q(a_i^T s)$")
        ax.set_ylabel(r"$L_q(b_i)$")
    for ax in axes.flat[len(views):]:
        ax.axis("off")
    cbar = fig.colorbar(scatter, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label(r"$e_i$")
    fig.suptitle("All-seed signal vs observed overview", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "all_seeds_signal_grid.png", dpi=dpi)
    plt.close(fig)


def save_csv_summaries(views: list[SeedView], out_dir: Path) -> None:
    score_path = out_dir / "coordinate_scores.csv"
    with score_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "coord", "secret_value", "support", "mutual_information", "abs_corr"])
        for view in views:
            for coord in range(len(view.secret)):
                writer.writerow(
                    [
                        view.seed,
                        coord,
                        int(view.secret[coord]),
                        int(view.support[coord]),
                        float(view.mi_scores[coord]),
                        float(view.corr_scores[coord]),
                    ]
                )

    metric_path = out_dir / "seed_metrics.csv"
    with metric_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "seed",
                "secret",
                "support",
                "observed_mean",
                "observed_std",
                "signal_mean",
                "signal_std",
                "residual_mean",
                "residual_std",
                "residual_abs_mean",
            ]
        )
        for view in views:
            writer.writerow(
                [
                    view.seed,
                    json.dumps(view.secret.tolist()),
                    json.dumps(view.support.tolist()),
                    float(np.mean(view.b_lift)),
                    float(np.std(view.b_lift)),
                    float(np.mean(view.signal)),
                    float(np.std(view.signal)),
                    float(np.mean(view.residual)),
                    float(np.std(view.residual)),
                    float(np.mean(np.abs(view.residual))),
                ]
            )


def save_metadata(views: list[SeedView], out_dir: Path, args: argparse.Namespace) -> None:
    payload = {
        "train_prefix": str(args.train_prefix),
        "secret_npy": str(args.secret_npy),
        "q": args.q,
        "primary_seed": args.seed,
        "seed_count": len(views),
        "scatter_samples": args.scatter_samples,
        "manifold_samples": args.manifold_samples,
        "edge_samples": args.edge_samples,
        "knn_k": args.knn_k,
        "seeds": [
            {"seed": view.seed, "secret": view.secret.tolist(), "support": view.support.tolist()}
            for view in views
        ],
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    a_rows, b_rows = load_prefix(args.train_prefix)
    secrets = np.load(args.secret_npy)
    if secrets.ndim != 2:
        raise ValueError(f"Expected a 2D secret array, got shape {secrets.shape}")
    if not (0 <= args.seed < secrets.shape[1]):
        raise ValueError(f"Primary seed {args.seed} is out of range for secret.npy with {secrets.shape[1]} columns")

    seed_ids = [args.seed]
    if args.all_seeds:
        seed_ids = list(range(secrets.shape[1]))

    views = [build_seed_view(a_rows, b_rows, secrets, seed_id, args.q) for seed_id in seed_ids]
    primary_view = next(view for view in views if view.seed == args.seed)

    save_seed_summary(
        primary_view,
        args.out_dir,
        args.q,
        args.scatter_samples,
        args.manifold_samples,
        args.edge_samples,
        args.knn_k,
        args.dpi,
        args.three_d_color,
        args.three_d_size,
        args.view_elev,
        args.view_azim,
    )
    if args.all_seeds:
        for view in views:
            if view.seed == args.seed:
                continue
            save_seed_summary(
                view,
                args.out_dir,
                args.q,
                args.scatter_samples,
                args.manifold_samples,
                args.edge_samples,
                args.knn_k,
                args.dpi,
                args.three_d_color,
                args.three_d_size,
                args.view_elev,
                args.view_azim,
            )
        save_all_seed_signal_grid(views, args.out_dir, args.q, args.scatter_samples, args.dpi)
    if args.three_d_only:
        for view in views:
            save_seed_3d_only(
                view,
                args.out_dir,
                args.manifold_samples,
                args.edge_samples,
                args.knn_k,
                args.dpi,
                args.three_d_color,
                args.three_d_size,
                args.view_elev,
                args.view_azim,
            )
    save_csv_summaries(views, args.out_dir)
    save_metadata(views, args.out_dir, args)


if __name__ == "__main__":
    main()
