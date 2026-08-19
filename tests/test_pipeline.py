"""
Basic unit tests. Run with: pytest

These aren't exhaustive, but they cover the exact bugs flagged in the
original notebook (silent Income drop, chained Age assignment, outlier
capping) so a regression gets caught instead of silently reshipped.
"""

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data.loader import clean_data
from src.features.engineer import add_age, add_total_spending, build_model_matrix
from src.models.train import check_cluster_stability, fit_pipeline


def _make_raw_df(**overrides: Any) -> pd.DataFrame:
    base = {
        "Year_Birth": [1985],
        "Income": [50000.0],
        "Kidhome": [0],
        "Teenhome": [0],
        "MntWines": [100],
        "MntFruits": [10],
        "MntMeatProducts": [50],
        "MntFishProducts": [20],
        "MntSweetProducts": [5],
        "MntGoldProds": [15],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_clean_data_drops_missing_income() -> None:
    df = _make_raw_df(Income=[None])
    cleaned = clean_data(df)
    assert cleaned["Income"].isna().sum() == 0
    assert len(cleaned) == 0


def test_clean_data_caps_high_income() -> None:
    df = _make_raw_df(Income=[666666.0])
    cleaned = clean_data(df)
    assert cleaned["Income"].iloc[0] == config.INCOME_MAX


def test_clean_data_caps_implausible_birth_year() -> None:
    df = _make_raw_df(Year_Birth=[1893])
    cleaned = clean_data(df)
    max_birth_year = pd.Timestamp.now().year - config.AGE_MAX
    assert cleaned["Year_Birth"].iloc[0] == max_birth_year


def test_add_age_does_not_pollute_other_columns() -> None:
    """Regression test for the `df['Age'] = df['current_year'] = ...` bug."""
    df = _make_raw_df()
    result = add_age(df)
    assert "current_year" not in result.columns
    assert "Age" in result.columns


def test_add_total_spending_sums_correctly() -> None:
    df = _make_raw_df()
    result = add_total_spending(df)
    expected = 100 + 10 + 50 + 20 + 5 + 15
    assert result["Total_Spending"].iloc[0] == expected


def test_build_model_matrix_has_expected_columns_in_order() -> None:
    df = pd.DataFrame([{f: 1 for f in config.FEATURES}])
    X = build_model_matrix(df)
    assert list(X.columns) == config.FEATURES


def test_build_model_matrix_raises_on_missing_feature() -> None:
    df = pd.DataFrame([{f: 1 for f in config.FEATURES[:-1]}])  # drop last feature
    with pytest.raises(ValueError, match=config.FEATURES[-1]):
        build_model_matrix(df)


def _make_toy_feature_matrix(n: int = 30) -> pd.DataFrame:
    """Small synthetic feature matrix with two obvious blobs, for fast tests."""
    import numpy as np

    rng = np.random.default_rng(0)
    blob_a = rng.normal(loc=0, scale=1, size=(n // 2, len(config.FEATURES)))
    blob_b = rng.normal(loc=20, scale=1, size=(n // 2, len(config.FEATURES)))
    return pd.DataFrame(np.vstack([blob_a, blob_b]), columns=config.FEATURES)


def test_pipeline_is_reproducible_given_fixed_random_state() -> None:
    """random_state should make repeated fits on the same data identical."""
    X = _make_toy_feature_matrix()

    pipe_a, labels_a, sil_a = fit_pipeline(X, n_clusters=2)
    pipe_b, labels_b, sil_b = fit_pipeline(X, n_clusters=2)

    assert list(labels_a) == list(labels_b)
    assert sil_a == pytest.approx(sil_b)


def test_pipeline_predict_matches_scaler_plus_kmeans_manually() -> None:
    """The pipeline's .predict() should equal manually scaling then predicting."""
    X = _make_toy_feature_matrix()
    pipe, labels, _ = fit_pipeline(X, n_clusters=2)

    manual_scaled = pipe.named_steps["scaler"].transform(X)
    manual_labels = pipe.named_steps["kmeans"].predict(manual_scaled)

    assert list(pipe.predict(X)) == list(manual_labels)


def test_cluster_stability_is_high_for_well_separated_blobs() -> None:
    """Two obvious blobs should get (near-)identical partitions across seeds."""
    X = _make_toy_feature_matrix()
    result = check_cluster_stability(X, n_clusters=2, random_states=(0, 1, 2))

    assert result["mean_ari"] > 0.9
    assert result["min_ari"] > 0.9


def test_cluster_stability_uses_different_seeds_per_run() -> None:
    """Regression test: build_pipeline() must actually vary by random_state."""
    X = _make_toy_feature_matrix()
    pipe_a = fit_pipeline(X, n_clusters=2)[0]
    assert pipe_a.named_steps["kmeans"].random_state == config.RANDOM_STATE

    from src.models.train import build_pipeline

    pipe_b = build_pipeline(n_clusters=2, random_state=99)
    assert pipe_b.named_steps["kmeans"].random_state == 99
