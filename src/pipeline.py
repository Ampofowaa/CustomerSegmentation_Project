"""
End-to-end training pipeline: load -> clean -> engineer -> fit -> save.

This is what scripts/train.py calls, and it's also importable from a
notebook or a test — the orchestration logic lives in exactly one place.
"""

import logging

import pandas as pd

import config
from src.data.loader import load_clean_data
from src.features.engineer import build_model_matrix, engineer_features
from src.models.train import cluster_profile, fit_pipeline, save_artifacts

logger = logging.getLogger(__name__)


def run_training_pipeline(n_clusters: int = config.N_CLUSTERS) -> dict:
    logger.info("Starting training pipeline (k=%d)", n_clusters)

    df = load_clean_data()
    df = engineer_features(df)
    X = build_model_matrix(df)

    # fit_pipeline fits StandardScaler and KMeans together as one
    # sklearn.pipeline.Pipeline object, so there's a single artifact to
    # save/load rather than a scaler and a model that could get out of sync.
    pipeline, labels, silhouette = fit_pipeline(X, n_clusters=n_clusters)

    profile = cluster_profile(df, labels)
    logger.info("Cluster profile:\n%s", profile.to_string())

    metrics = {
        "n_clusters": n_clusters,
        "silhouette_score": silhouette,
        "n_rows_trained": len(df),
        "cluster_sizes": {
            int(k): int(v) for k, v in pd.Series(labels).value_counts().items()
        },
    }

    save_artifacts(pipeline, metrics)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run_training_pipeline()
