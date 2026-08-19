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
from src.features.engineer import build_model_matrix


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
    X = build_model_matrix(pd.DataFrame([feature_row]))
    cluster = int(load_pipeline().predict(X)[0])

    return {
        "cluster": cluster,
        "cluster_name": config.CLUSTER_NAMES.get(cluster, f"Cluster {cluster}"),
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score many customers at once.

    df: any DataFrame containing at least the config.FEATURES columns
    (extra columns, e.g. a customer ID, are echoed back untouched).

    Returns a copy of df with three columns appended: "cluster",
    "cluster_name", and "scored_at" (UTC date of this call, YYYY-MM-DD,
    same value for every row — when this batch was run, not a per-row
    date already present in the data). Raises the same ValueError as
    predict_segment() — via build_model_matrix() — if a required feature
    column is missing.
    """
    X = build_model_matrix(df)
    clusters = load_pipeline().predict(X)

    result = df.copy()
    result["cluster"] = clusters.astype(int)
    result["cluster_name"] = result["cluster"].map(config.CLUSTER_NAMES)
    result["scored_at"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    return result
