"""
Fit a single scaler -> KMeans Pipeline and persist it as one artifact.

Addresses the gaps flagged during review:
- random_state + n_init for reproducible clusters across runs
- silhouette score alongside inertia, so K isn't chosen on the elbow alone
- metrics saved to disk so a training run leaves a record of how K was chosen
- ONE Pipeline object instead of a separate scaler.pkl + kmeans_model.pkl.
  Two artifacts can silently drift apart (retrain the model, forget to
  re-save the scaler, ship a stale one) and nothing catches it since
  StandardScaler.transform() doesn't validate against the model it's paired
  with. A Pipeline makes "scale, then cluster" one versioned, atomic unit —
  load it, call .predict(), done.
"""

import itertools
import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config

logger = logging.getLogger(__name__)


def build_pipeline(n_clusters: int, random_state: int = config.RANDOM_STATE, use_pca: bool = False) -> Pipeline:
    """
    Construct an (unfit) scaler -> [pca] -> KMeans pipeline.

    use_pca inserts a PCA step (retaining config.PCA_VARIANCE of variance)
    between scaling and clustering, to strip collinearity between the spend/
    purchase-count features before KMeans sees them. Off by default so
    existing callers/artifacts are unaffected; pass True to compare against
    the no-PCA baseline via evaluate_k_range/check_cluster_stability/
    check_bootstrap_stability below.
    """
    steps: list[tuple[str, object]] = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=config.PCA_VARIANCE, random_state=random_state)))
    steps.append(
        (
            "kmeans",
            KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=config.N_INIT,
            ),
        )
    )
    return Pipeline(steps=steps)


def _transform_up_to_kmeans(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Run X through every pipeline step except the final KMeans -- i.e. the
    space the KMeans (and therefore silhouette_score) actually operates in,
    whether or not a pca step is present."""
    Xt = X
    for name, step in pipe.steps[:-1]:
        Xt = step.transform(Xt)
    return np.asarray(Xt)


def evaluate_k_range(
    X: pd.DataFrame, k_range: range = config.K_SEARCH_RANGE, use_pca: bool = False
) -> dict[int, dict[str, float]]:
    """
    Fit a full scaler->[pca]->KMeans pipeline for each candidate K and record
    inertia (elbow) + silhouette score. Returns a dict keyed by k so the
    caller can plot/justify a choice.
    """
    results: dict[int, dict[str, float]] = {}
    for k in k_range:
        pipe = build_pipeline(k, use_pca=use_pca)
        labels = pipe.fit_predict(X)
        X_transformed = _transform_up_to_kmeans(pipe, X)
        results[k] = {
            "inertia": float(pipe.named_steps["kmeans"].inertia_),
            "silhouette": float(silhouette_score(X_transformed, labels)),
        }
        logger.info(
            "k=%d  inertia=%.1f  silhouette=%.4f",
            k,
            results[k]["inertia"],
            results[k]["silhouette"],
        )
    return results


def fit_pipeline(
    X: pd.DataFrame,
    n_clusters: int = config.N_CLUSTERS,
    random_state: int = config.RANDOM_STATE,
    use_pca: bool = False,
) -> tuple[Pipeline, np.ndarray, float]:
    """
    Fit the scaler -> [pca] -> KMeans pipeline on the given (raw, unscaled)
    feature matrix. Returns (pipeline, cluster_labels, silhouette_score).

    random_state is exposed (rather than always using config.RANDOM_STATE)
    so callers can re-fit across several seeds to check whether a silhouette
    difference is a real effect or just noise from centroid initialization
    (see check_cluster_stability, and the multi-seed feature-selection
    checks in the notebook).
    """
    pipe = build_pipeline(n_clusters, random_state=random_state, use_pca=use_pca)
    labels = pipe.fit_predict(X)

    X_transformed = _transform_up_to_kmeans(pipe, X)
    sil = float(silhouette_score(X_transformed, labels))
    logger.info("Final pipeline: k=%d silhouette=%.4f", n_clusters, sil)

    return pipe, labels, sil


def compute_validity_indices(pipeline: Pipeline, X: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:
    """
    Davies-Bouldin (lower is better, 0 = perfectly separated) and
    Calinski-Harabasz (higher is better) indices, computed in the same space
    KMeans actually clustered in (scaler -> pca if present) -- a second
    opinion alongside silhouette, useful since silhouette alone can be
    distorted by uneven cluster sizes.
    """
    X_transformed = _transform_up_to_kmeans(pipeline, X)
    return {
        "davies_bouldin": float(davies_bouldin_score(X_transformed, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X_transformed, labels)),
    }


def check_cluster_stability(
    X: pd.DataFrame,
    n_clusters: int = config.N_CLUSTERS,
    random_states: tuple[int, ...] = (0, 1, 2, 3, 4),
    use_pca: bool = False,
) -> dict[str, float]:
    """
    Refit the pipeline with several different random_state seeds and
    compare the resulting cluster assignments pairwise with the Adjusted
    Rand Index (1.0 = identical partitions, ~0.0 = no better than chance).

    A single random_state producing "good" clusters doesn't rule out that
    they're an artifact of that particular centroid initialization — this
    checks whether independent runs actually agree with each other.
    """
    labels_by_seed = {
        rs: build_pipeline(n_clusters, random_state=rs, use_pca=use_pca).fit_predict(X) for rs in random_states
    }

    scores = [
        adjusted_rand_score(labels_by_seed[a], labels_by_seed[b]) for a, b in itertools.combinations(random_states, 2)
    ]
    result = {"mean_ari": float(np.mean(scores)), "min_ari": float(np.min(scores))}
    logger.info(
        "Cluster stability across seeds %s: mean_ari=%.4f min_ari=%.4f",
        random_states,
        result["mean_ari"],
        result["min_ari"],
    )
    return result


def check_bootstrap_stability(
    X: pd.DataFrame,
    n_clusters: int = config.N_CLUSTERS,
    n_bootstraps: int = 10,
    random_state: int = config.RANDOM_STATE,
    use_pca: bool = False,
) -> dict[str, float]:
    """
    Refit the pipeline on several bootstrap resamples (rows drawn with
    replacement, same size as X) and compare each resample-fit pipeline's
    predictions on the ORIGINAL X against a reference fit on the full data,
    via Adjusted Rand Index.

    check_cluster_stability() above only varies centroid initialization on
    the same fixed dataset, so it can't say anything about whether the
    segmentation would hold up on a different sample of customers. This
    check varies the training rows instead (holding centroid init fixed),
    which is the more common "would this replicate" stability check used
    in practice.
    """
    rng = np.random.RandomState(random_state)
    reference_labels = build_pipeline(n_clusters, random_state=random_state, use_pca=use_pca).fit_predict(X)

    n = len(X)
    scores = []
    for _ in range(n_bootstraps):
        boot_idx = rng.randint(0, n, size=n)
        X_boot = X.iloc[boot_idx]
        boot_pipe = build_pipeline(n_clusters, random_state=random_state, use_pca=use_pca).fit(X_boot)
        boot_labels_on_full = boot_pipe.predict(X)
        scores.append(adjusted_rand_score(reference_labels, boot_labels_on_full))

    result = {"mean_ari": float(np.mean(scores)), "min_ari": float(np.min(scores))}
    logger.info(
        "Bootstrap cluster stability across %d resamples: mean_ari=%.4f min_ari=%.4f",
        n_bootstraps,
        result["mean_ari"],
        result["min_ari"],
    )
    return result


def _round_floats(value: Any, ndigits: int = 4) -> Any:
    """Recursively round floats in a metrics dict for a readable training_metrics.json."""
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, dict):
        return {k: _round_floats(v, ndigits) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_floats(v, ndigits) for v in value]
    return value


def save_artifacts(pipeline: Pipeline, metrics: dict) -> None:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.PIPELINE_PATH)
    with open(config.METRICS_PATH, "w") as f:
        json.dump(_round_floats(metrics), f, indent=2)
        f.write("\n")
    logger.info("Saved pipeline -> %s", config.PIPELINE_PATH)
    logger.info("Saved metrics  -> %s", config.METRICS_PATH)


def cluster_profile(df: pd.DataFrame, labels: np.ndarray, extra_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Mean of each feature per cluster — the table you actually read to name
    the clusters. Optionally include non-modeling columns (e.g. Education,
    campaign acceptance) for a fuller business profile.

    Deduped (order-preserving) so passing a column already in config.FEATURES
    via extra_cols doesn't silently double it up in the output.
    """
    df = df.copy()
    df["Cluster"] = labels
    cols = list(dict.fromkeys(config.FEATURES + (extra_cols or [])))
    return df.groupby("Cluster")[cols].mean()
