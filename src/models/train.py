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

import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config

logger = logging.getLogger(__name__)


def build_pipeline(n_clusters: int) -> Pipeline:
    """Construct an (unfit) scaler -> KMeans pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(
                    n_clusters=n_clusters,
                    random_state=config.RANDOM_STATE,
                    n_init=config.N_INIT,
                ),
            ),
        ]
    )


#     return pipe, labels, sil
def evaluate_k_range(X: pd.DataFrame, k_range: range = config.K_SEARCH_RANGE) -> dict[int, dict[str, float]]:
    """
    Fit a full scaler->KMeans pipeline for each candidate K and record
    inertia (elbow) + silhouette score. Returns a dict keyed by k so the
    caller can plot/justify a choice.
    """
    results: dict[int, dict[str, float]] = {}
    for k in k_range:
        pipe = build_pipeline(k)
        labels = pipe.fit_predict(X)
        X_scaled = pipe.named_steps["scaler"].transform(X)
        results[k] = {
            "inertia": float(pipe.named_steps["kmeans"].inertia_),
            "silhouette": float(silhouette_score(X_scaled, labels)),
        }
        logger.info(
            "k=%d  inertia=%.1f  silhouette=%.4f",
            k,
            results[k]["inertia"],
            results[k]["silhouette"],
        )
    return results


def fit_pipeline(X: pd.DataFrame, n_clusters: int = config.N_CLUSTERS) -> tuple[Pipeline, np.ndarray, float]:
    """
    Fit the scaler -> KMeans pipeline on the given (raw, unscaled) feature
    matrix. Returns (pipeline, cluster_labels, silhouette_score).
    """
    pipe = build_pipeline(n_clusters)
    labels = pipe.fit_predict(X)

    X_scaled = pipe.named_steps["scaler"].transform(X)
    sil = float(silhouette_score(X_scaled, labels))
    logger.info("Final pipeline: k=%d silhouette=%.4f", n_clusters, sil)

    return pipe, labels, sil


def save_artifacts(pipeline: Pipeline, metrics: dict) -> None:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.PIPELINE_PATH)
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved pipeline -> %s", config.PIPELINE_PATH)
    logger.info("Saved metrics  -> %s", config.METRICS_PATH)


def cluster_profile(df: pd.DataFrame, labels: np.ndarray, extra_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Mean of each feature per cluster — the table you actually read to name
    the clusters. Optionally include non-modeling columns (e.g. Education,
    campaign acceptance) for a fuller business profile.
    """
    df = df.copy()
    df["Cluster"] = labels
    cols = config.FEATURES + (extra_cols or [])
    return df.groupby("Cluster")[cols].mean()
