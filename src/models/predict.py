"""
Load the trained pipeline and score new customers.

This is the ONLY place that should call pipeline.predict() outside of
training — the Streamlit app imports predict_segment() rather than
re-implementing scaling/prediction itself.
"""

import functools

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

import config


@functools.lru_cache(maxsize=1)
def load_pipeline() -> Pipeline:
    """Load the fitted scaler->KMeans pipeline once per process (cached)."""
    return joblib.load(config.PIPELINE_PATH)


def predict_segment(feature_row: dict) -> dict:
    """
    Score a single customer.

    feature_row: dict with keys matching config.FEATURES exactly, e.g.
        {"Age": 35, "Income": 50000, "Total_Spending": 1000,
         "NumWebPurchases": 10, "NumStorePurchases": 10,
         "NumWebVisitsMonth": 3, "Recency": 30}

    Returns: {"cluster": int, "cluster_name": str}
    """
    missing = set(config.FEATURES) - set(feature_row)
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    pipeline = load_pipeline()

    # Build the row in config.FEATURES order — never trust dict insertion
    # order or the caller's ordering. The pipeline handles scaling and
    # clustering as one call, so there's no separate transform() step to
    # get out of sync.
    X = pd.DataFrame([{f: feature_row[f] for f in config.FEATURES}])
    cluster = int(pipeline.predict(X)[0])

    return {
        "cluster": cluster,
        "cluster_name": config.CLUSTER_NAMES.get(cluster, f"Cluster {cluster}"),
    }
