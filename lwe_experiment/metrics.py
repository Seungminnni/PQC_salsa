from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import entropy as shannon_entropy
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    log_loss,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


@dataclass(slots=True)
class ProbeArtifacts:
    y_true: np.ndarray
    linear_pred: np.ndarray
    degree2_pred: np.ndarray
    mlp_pred: np.ndarray
    probe_train_time_ms: float


def _safe_auc(binary_labels: np.ndarray, scores: np.ndarray) -> float:
    unique = np.unique(binary_labels)
    if unique.size < 2:
        return 0.5
    return float(roc_auc_score(binary_labels, scores))


def _flatten_sign_labels(values: np.ndarray) -> np.ndarray:
    return np.sign(values).astype(int).ravel()


def _bucketize(values: np.ndarray, clipping_bound: int) -> np.ndarray:
    edges = np.linspace(-clipping_bound, clipping_bound, num=6)
    return np.digitize(values, edges[1:-1], right=False)


def _probe_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    sign_true = _flatten_sign_labels(y_true)
    sign_pred = _flatten_sign_labels(y_pred)
    positive_true = (y_true.ravel() > 0).astype(int)
    return {
        "r2": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "sign_acc": float(np.mean(sign_true == sign_pred)),
        "macro_f1": float(f1_score(sign_true, sign_pred, average="macro", labels=[-1, 0, 1])),
        "auroc": _safe_auc(positive_true, y_pred.ravel()),
    }


def compute_correctness_metrics(
    secrets: np.ndarray,
    messages: np.ndarray,
    decoded_messages: np.ndarray,
    effective_noise: np.ndarray,
    noise_margin: np.ndarray,
) -> dict[str, float]:
    abs_noise = np.abs(effective_noise)
    decrypt_success = np.mean(messages == decoded_messages)
    return {
        "decrypt_success_rate": float(decrypt_success),
        "decrypt_failure_rate": float(1.0 - decrypt_success),
        "bit_error_rate": float(np.mean(messages != decoded_messages)),
        "message_error_rate": float(np.mean(messages != decoded_messages)),
        "avg_effective_noise": float(abs_noise.mean()),
        "std_effective_noise": float(abs_noise.std()),
        "max_effective_noise": float(abs_noise.max()),
        "noise_margin_mean": float(noise_margin.mean()),
        "noise_margin_p95": float(np.percentile(noise_margin, 95)),
        "noise_margin_p99": float(np.percentile(noise_margin, 99)),
        "secret_l2_norm_mean": float(np.linalg.norm(secrets, axis=1).mean()),
        "secret_linf_norm_mean": float(np.abs(secrets).max(axis=1).mean()),
    }


def compute_secret_distribution_metrics(
    secrets: np.ndarray,
    clipping_bound: int,
) -> dict[str, float]:
    flattened = secrets.ravel().astype(float)
    secret_mean = float(flattened.mean())
    secret_std = float(flattened.std())
    secret_min = float(flattened.min())
    secret_max = float(flattened.max())
    secret_abs_mean = float(np.abs(flattened).mean())
    secret_sparsity_ratio = float(np.mean(secrets == 0))

    values, counts = np.unique(flattened.astype(int), return_counts=True)
    probabilities = counts / counts.sum()
    secret_entropy = float(shannon_entropy(probabilities, base=2))

    covariance = np.cov(secrets, rowvar=False)
    covariance = np.atleast_2d(covariance)
    eigvals = np.linalg.eigvalsh(covariance)
    eigvals = np.clip(eigvals, 0.0, None)
    covariance_trace = float(np.trace(covariance))

    total_eig = eigvals.sum()
    if total_eig <= 1e-12:
        effective_rank = 0.0
    else:
        normalized = eigvals / total_eig
        normalized = normalized[normalized > 0]
        effective_rank = float(np.exp(-np.sum(normalized * np.log(normalized))))

    positive_eig = eigvals[eigvals > 1e-9]
    if positive_eig.size <= 1:
        condition_number = 0.0
    else:
        condition_number = float(positive_eig.max() / positive_eig.min())

    corr = np.corrcoef(secrets, rowvar=False)
    corr = np.nan_to_num(np.atleast_2d(corr))
    if corr.shape[0] > 1:
        off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
        corr_mean_abs = float(np.mean(np.abs(off_diag)))
        corr_max_abs = float(np.max(np.abs(off_diag)))
    else:
        corr_mean_abs = 0.0
        corr_max_abs = 0.0

    freq_map = {int(v): float(p) for v, p in zip(values, probabilities)}
    symmetry_deltas = []
    for value in range(1, clipping_bound + 1):
        symmetry_deltas.append(abs(freq_map.get(value, 0.0) - freq_map.get(-value, 0.0)))
    symmetry_gap = float(np.mean(symmetry_deltas)) if symmetry_deltas else 0.0

    bucket_hist = np.bincount(
        _bucketize(flattened, clipping_bound=clipping_bound).astype(int),
        minlength=5,
    ).astype(float)
    bucket_hist = bucket_hist / max(bucket_hist.sum(), 1.0)
    bucket_balance_score = float(
        shannon_entropy(bucket_hist + 1e-12, base=2) / np.log2(bucket_hist.size)
    )

    unique_rows, row_counts = np.unique(secrets, axis=0, return_counts=True)
    repeat_pattern_score = float(row_counts.max() / max(secrets.shape[0], 1))

    if secrets.shape[0] > 1:
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(secrets)
        nn_distances = nn.kneighbors(return_distance=True)[0][:, 1]
        neighbor_distance_mean = float(nn_distances.mean())
        neighbor_distance_std = float(nn_distances.std())
    else:
        neighbor_distance_mean = 0.0
        neighbor_distance_std = 0.0

    centroid = secrets.mean(axis=0, keepdims=True)
    centroid_distance_mean = float(np.linalg.norm(secrets - centroid, axis=1).mean())

    return {
        "secret_mean": secret_mean,
        "secret_std": secret_std,
        "secret_min": secret_min,
        "secret_max": secret_max,
        "secret_abs_mean": secret_abs_mean,
        "secret_sparsity_ratio": secret_sparsity_ratio,
        "secret_entropy": secret_entropy,
        "secret_pairwise_corr_mean_abs": corr_mean_abs,
        "secret_pairwise_corr_max_abs": corr_max_abs,
        "secret_cov_trace": covariance_trace,
        "secret_effective_rank": effective_rank,
        "secret_condition_number": condition_number,
        "secret_symmetry_gap": symmetry_gap,
        "secret_bucket_balance_score": bucket_balance_score,
        "secret_repeat_pattern_score": repeat_pattern_score,
        "secret_neighbor_distance_mean": neighbor_distance_mean,
        "secret_neighbor_distance_std": neighbor_distance_std,
        "secret_centroid_distance_mean": centroid_distance_mean,
    }


def compute_recovery_metrics(
    true_secrets: np.ndarray,
    recovered_secrets: np.ndarray,
    clipping_bound: int,
) -> dict[str, float]:
    errors = recovered_secrets.astype(float) - true_secrets.astype(float)
    l2_error = np.linalg.norm(errors, axis=1)
    true_norm = np.linalg.norm(true_secrets, axis=1)
    relative_l2 = l2_error / np.maximum(true_norm, 1e-8)
    dot = np.sum(true_secrets * recovered_secrets, axis=1)
    recovered_norm = np.linalg.norm(recovered_secrets, axis=1)
    cosine = dot / np.maximum(true_norm * recovered_norm, 1e-8)

    rounded = np.clip(np.rint(recovered_secrets), -clipping_bound, clipping_bound).astype(int)
    bucket_true = _bucketize(true_secrets, clipping_bound=clipping_bound)
    bucket_pred = _bucketize(rounded, clipping_bound=clipping_bound)

    return {
        "secret_recovery_l2_error": float(l2_error.mean()),
        "secret_recovery_relative_l2_error": float(relative_l2.mean()),
        "secret_recovery_cosine_similarity": float(np.mean(cosine)),
        "secret_recovery_coord_acc": float(np.mean(rounded == true_secrets)),
        "secret_recovery_sign_acc": float(np.mean(np.sign(rounded) == np.sign(true_secrets))),
        "secret_recovery_bucket_acc": float(np.mean(bucket_true == bucket_pred)),
    }


def run_secret_probes(
    public_features: np.ndarray,
    secrets: np.ndarray,
    seed: int,
    train_fraction: float,
    hidden_layers: tuple[int, ...],
    max_iter: int,
) -> tuple[dict[str, float], ProbeArtifacts]:
    X_train, X_test, y_train, y_test = train_test_split(
        public_features,
        secrets,
        train_size=train_fraction,
        random_state=seed,
        shuffle=True,
    )

    metrics: dict[str, float] = {}
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        linear_model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("regressor", Ridge(alpha=1.0, random_state=seed)),
            ]
        )
        linear_model.fit(X_train, y_train)
        linear_pred = linear_model.predict(X_test)
        linear_scores = _probe_regression_metrics(y_test, linear_pred)
        metrics.update(
            {
                "linear_probe_r2": linear_scores["r2"],
                "linear_probe_mse": linear_scores["mse"],
                "linear_probe_sign_acc": linear_scores["sign_acc"],
                "linear_probe_macro_f1": linear_scores["macro_f1"],
            }
        )

        degree2_model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("regressor", Ridge(alpha=2.0, random_state=seed)),
            ]
        )
        degree2_model.fit(X_train, y_train)
        degree2_pred = degree2_model.predict(X_test)
        degree2_scores = _probe_regression_metrics(y_test, degree2_pred)
        metrics.update(
            {
                "degree2_probe_r2": degree2_scores["r2"],
                "degree2_probe_mse": degree2_scores["mse"],
                "degree2_probe_sign_acc": degree2_scores["sign_acc"],
                "degree2_probe_auroc": degree2_scores["auroc"],
            }
        )

        mlp_model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "regressor",
                    MLPRegressor(
                        hidden_layer_sizes=hidden_layers,
                        activation="tanh",
                        early_stopping=True,
                        max_iter=max_iter,
                        random_state=seed,
                    ),
                ),
            ]
        )
        mlp_model.fit(X_train, y_train)
        mlp_pred = mlp_model.predict(X_test)
        mlp_scores = _probe_regression_metrics(y_test, mlp_pred)
        metrics.update(
            {
                "mlp_probe_mse": mlp_scores["mse"],
                "mlp_probe_sign_acc": mlp_scores["sign_acc"],
                "mlp_probe_auroc": mlp_scores["auroc"],
            }
        )

    train_time_ms = (time.perf_counter() - start) * 1000.0
    artifacts = ProbeArtifacts(
        y_true=y_test,
        linear_pred=linear_pred,
        degree2_pred=degree2_pred,
        mlp_pred=mlp_pred,
        probe_train_time_ms=train_time_ms,
    )
    return metrics, artifacts


def public_key_classifier_metrics(
    baseline_features: np.ndarray,
    current_features: np.ndarray,
    seed: int,
) -> dict[str, float]:
    if baseline_features.shape == current_features.shape and np.array_equal(baseline_features, current_features):
        return {
            "baseline_vs_proposed_pk_auroc": 0.5,
            "baseline_vs_proposed_pk_acc": 0.5,
            "baseline_vs_proposed_pk_logloss": float(np.log(2.0)),
        }

    X = np.vstack([baseline_features, current_features])
    y = np.concatenate([np.zeros(baseline_features.shape[0]), np.ones(current_features.shape[0])])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=seed,
        stratify=y,
        shuffle=True,
    )

    classifier = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=500, random_state=seed)),
        ]
    )
    classifier.fit(X_train, y_train)
    probabilities = classifier.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "baseline_vs_proposed_pk_auroc": _safe_auc(y_test.astype(int), probabilities),
        "baseline_vs_proposed_pk_acc": float(accuracy_score(y_test, predictions)),
        "baseline_vs_proposed_pk_logloss": float(log_loss(y_test, probabilities, labels=[0, 1])),
    }


def compute_distribution_distance_metrics(
    secrets: np.ndarray,
    baseline_reference: np.ndarray,
    num_clusters: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    metrics: dict[str, float] = {}
    diagnostics: dict[str, np.ndarray] = {}

    if secrets.shape[0] > 1:
        pairwise_distances = pdist(secrets.astype(float))
        diagnostics["pairwise_distances"] = pairwise_distances
    else:
        pairwise_distances = np.array([0.0])
        diagnostics["pairwise_distances"] = pairwise_distances

    if secrets.shape[0] > 1:
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(secrets)
        knn_distances = nn.kneighbors(return_distance=True)[0][:, 1]
    else:
        knn_distances = np.array([0.0])
    diagnostics["knn_distances"] = knn_distances

    cluster_count = int(max(2, min(num_clusters, secrets.shape[0] - 1)))
    if cluster_count >= 2 and secrets.shape[0] > cluster_count:
        kmeans = KMeans(n_clusters=cluster_count, n_init=10, random_state=seed)
        labels = kmeans.fit_predict(secrets)
        centers = kmeans.cluster_centers_
        intra = np.linalg.norm(secrets - centers[labels], axis=1)
        inter = pdist(centers)
        metrics["intra_cluster_distance_mean"] = float(intra.mean())
        metrics["inter_cluster_distance_mean"] = float(inter.mean()) if inter.size else 0.0
        metrics["distance_ratio_inter_over_intra"] = float(
            metrics["inter_cluster_distance_mean"] / max(metrics["intra_cluster_distance_mean"], 1e-8)
        )
        try:
            metrics["silhouette_score"] = float(silhouette_score(secrets, labels))
        except ValueError:
            metrics["silhouette_score"] = 0.0
        try:
            metrics["davies_bouldin_score"] = float(davies_bouldin_score(secrets, labels))
        except ValueError:
            metrics["davies_bouldin_score"] = 0.0
        try:
            metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(secrets, labels))
        except ValueError:
            metrics["calinski_harabasz_score"] = 0.0
    else:
        metrics["intra_cluster_distance_mean"] = 0.0
        metrics["inter_cluster_distance_mean"] = 0.0
        metrics["distance_ratio_inter_over_intra"] = 0.0
        metrics["silhouette_score"] = 0.0
        metrics["davies_bouldin_score"] = 0.0
        metrics["calinski_harabasz_score"] = 0.0

    metrics["knn_distance_mean"] = float(knn_distances.mean())
    metrics["knn_distance_std"] = float(knn_distances.std())

    if baseline_reference.shape == secrets.shape and np.array_equal(baseline_reference, secrets):
        metrics["mmd_score"] = 0.0
        metrics["wasserstein_distance_to_baseline"] = 0.0
        metrics["ks_stat_per_dimension_mean"] = 0.0
        return metrics, diagnostics

    combined = np.vstack([secrets, baseline_reference]).astype(float)
    if combined.shape[0] > 1:
        combined_distances = pdist(combined)
        median_distance = float(np.median(combined_distances[combined_distances > 0])) if np.any(combined_distances > 0) else 1.0
    else:
        median_distance = 1.0
    gamma = 1.0 / max(2.0 * (median_distance**2), 1e-8)
    k_xx = rbf_kernel(secrets, secrets, gamma=gamma)
    k_yy = rbf_kernel(baseline_reference, baseline_reference, gamma=gamma)
    k_xy = rbf_kernel(secrets, baseline_reference, gamma=gamma)
    metrics["mmd_score"] = float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())

    wasserstein_per_dim = []
    ks_per_dim = []
    for dim in range(secrets.shape[1]):
        wasserstein_per_dim.append(
            wasserstein_distance(secrets[:, dim].astype(float), baseline_reference[:, dim].astype(float))
        )
        ks_per_dim.append(ks_2samp(secrets[:, dim], baseline_reference[:, dim]).statistic)
    metrics["wasserstein_distance_to_baseline"] = float(np.mean(wasserstein_per_dim))
    metrics["ks_stat_per_dimension_mean"] = float(np.mean(ks_per_dim))
    return metrics, diagnostics
