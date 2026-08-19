"""
Turn cleaned raw columns into the features the model actually trains on.

Kept as pure functions (DataFrame in, DataFrame out) so the exact same code
path engineers features for training data and for a single row typed into
the Streamlit app at inference time — no risk of the two silently diverging.
"""

import pandas as pd

import config


def add_age(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Age"] = pd.Timestamp.now().year - df["Year_Birth"]
    return df


def add_total_spending(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Total_Spending"] = df[config.SPEND_COLS].sum(axis=1)
    return df


def add_total_children(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalNo_Children"] = df["Kidhome"] + df["Teenhome"]
    return df


def add_customer_tenure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Customer_Tenure"] = (pd.Timestamp("today") - df["Dt_Customer"]).dt.days
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every engineered feature. Safe to call on cleaned raw data."""
    df = add_age(df)
    df = add_total_spending(df)
    df = add_total_children(df)
    if "Dt_Customer" in df.columns:
        df = add_customer_tenure(df)
    return df


def build_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the columns the model expects, in config.FEATURES order.
    This is the function both training and inference call last, right
    before scaling — it's the contract that keeps them in sync.
    """
    missing = [c for c in config.FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"config.FEATURES references column(s) not present after "
            f"cleaning/engineering: {missing}. Either fix the typo in "
            f"config.FEATURES or add the missing column in "
            f"src/features/engineer.py."
        )
    return df[config.FEATURES].copy()
