"""
End-to-end training pipeline: load -> clean -> engineer -> fit -> save.

This is what scripts/train.py calls, and it's also importable from a
notebook or a test — the orchestration logic lives in exactly one place.
"""

import logging
from typing import Any, cast

import pandas as pd

import config
from src.data.loader import load_clean_data
from src.features.engineer import build_model_matrix, engineer_features
from src.models.train import (
    check_bootstrap_stability,
    check_cluster_stability,
    cluster_profile,
    compute_validity_indices,
    fit_pipeline,
    save_artifacts,
)

logger = logging.getLogger(__name__)


def run_training_pipeline(n_clusters: int = config.N_CLUSTERS, use_pca: bool = True) -> dict[str, Any]:
    logger.info("Starting training pipeline (k=%d, use_pca=%s)", n_clusters, use_pca)

    df = load_clean_data()
    df = engineer_features(df)
    X = build_model_matrix(df)

    pipeline, labels, silhouette = fit_pipeline(X, n_clusters=n_clusters, use_pca=use_pca)
    validity = compute_validity_indices(pipeline, X, labels)
    seed_stability = check_cluster_stability(X, n_clusters=n_clusters, use_pca=use_pca)
    bootstrap_stability = check_bootstrap_stability(X, n_clusters=n_clusters, use_pca=use_pca)

    profile = cluster_profile(df, labels)
    logger.info("Cluster profile:\n%s", profile.to_string())

    metrics = {
        "trained_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_clusters": n_clusters,
        "random_state": config.RANDOM_STATE,
        "n_init": config.N_INIT,
        "use_pca": use_pca,
        "pca_n_components": int(pipeline.named_steps["pca"].n_components_) if use_pca else None,
        "pca_explained_variance_ratio": (
            pipeline.named_steps["pca"].explained_variance_ratio_.tolist() if use_pca else None
        ),
        "silhouette_score": silhouette,
        "davies_bouldin_score": validity["davies_bouldin"],
        "calinski_harabasz_score": validity["calinski_harabasz"],
        "seed_stability": seed_stability,
        "bootstrap_stability": bootstrap_stability,
        "n_rows_trained": len(df),
        "cluster_sizes": {cast(int, k): int(v) for k, v in pd.Series(labels).value_counts().items()},
    }

    save_artifacts(pipeline, metrics)

    # Only `metrics` is JSON-serialized (see save_artifacts above) -- the
    # pipeline/labels/profile/df below are returned alongside it purely for
    # callers (e.g. the notebook) that want to keep exploring right after
    # training without re-fitting.
    return {**metrics, "pipeline": pipeline, "labels": labels, "profile": profile, "df": df}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_training_pipeline()
