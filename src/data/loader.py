"""
Load and clean the raw customer data.

Kept separate from feature engineering so each step can be unit tested and
reused independently (e.g. cleaning logic is identical whether it feeds the
notebook, the training script, or a future batch-scoring job).
"""

import logging
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)


def load_raw_data(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and parse the customer-since date column."""
    df = pd.read_csv(path)
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and known data-quality issues.

    Unlike the exploratory notebook (which silently drops missing Income),
    this logs exactly what's being removed/capped so a pipeline run leaves
    an audit trail.
    """
    df = df.copy()

    n_missing = df["Income"].isna().sum()
    if n_missing:
        pct = n_missing / len(df) * 100
        logger.info("Dropping %d rows (%.1f%%) with missing Income", n_missing, pct)
        df = df.dropna(subset=["Income"])

    # Cap rather than silently ignore known outliers (birth-year typos,
    # the extreme income outlier). Capping preserves the row instead of
    # discarding potentially useful signal from the rest of its features.
    birth_year_cutoff = pd.Timestamp.now().year - config.AGE_MAX
    n_old = (df["Year_Birth"] < birth_year_cutoff).sum()
    if n_old:
        logger.info(
            "Capping %d rows with implausible Year_Birth (< %d)",
            n_old,
            birth_year_cutoff,
        )
        df["Year_Birth"] = df["Year_Birth"].clip(lower=birth_year_cutoff)

    n_high_income = (df["Income"] > config.INCOME_MAX).sum()
    if n_high_income:
        logger.info("Capping %d rows with Income above %d", n_high_income, config.INCOME_MAX)
        df["Income"] = df["Income"].clip(upper=config.INCOME_MAX)

    return df.reset_index(drop=True)


def load_clean_data(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Convenience wrapper: load + clean in one call."""
    return clean_data(load_raw_data(path))
