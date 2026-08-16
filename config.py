"""
Central configuration for the customer segmentation project.

Nothing else in this codebase should hardcode a path, a feature name, or a
hyperparameter — everything lives here so training and inference can never
drift apart (e.g. the notebook using a different feature order than the app).
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

RAW_DATA_PATH = DATA_DIR / "customer_segmentation.csv"
PIPELINE_PATH = MODEL_DIR / "segmentation_pipeline.pkl"
METRICS_PATH = MODEL_DIR / "training_metrics.json"

# --- Raw columns used to engineer features ------------------------------
SPEND_COLS = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]

# --- Final feature set fed to the model ---------------------------------
# ORDER MATTERS: StandardScaler.transform() only cares about column
# position, not names. Train and inference must build this list identically,
# which is exactly why it's defined once, here, and imported everywhere else.
FEATURES = [
    "Age",
    "Income",
    "Total_Spending",
    "NumWebPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "Recency",
]

# --- Cleaning thresholds --------------------------------------------------
AGE_MAX = 90  # caps the birth-year data-entry errors (e.g. Year_Birth=1900)
INCOME_MAX = 200_000  # caps the ~$666,666 income outlier that skews K-Means

# --- Model hyperparameters ------------------------------------------------
N_CLUSTERS = 5
RANDOM_STATE = 42  # fixes centroid init -> reproducible clusters across runs
N_INIT = 10  # sklearn re-runs K-Means 10x and keeps the best (avoids bad local optima)
K_SEARCH_RANGE = range(2, 10)

# --- Human-readable cluster labels ----------------------------------------
# Set by inspecting cluster_profile() in the notebook after training — a
# cluster number means nothing to a business stakeholder, a name does.
# NOTE: cluster IDs are not stable across re-training runs unless
# random_state is fixed (it is, see RANDOM_STATE above) AND the training
# data doesn't change. If you retrain on new data, re-check this mapping
# against the new cluster_profile() output before trusting these labels.
CLUSTER_NAMES = {
    0: "Low-Engagement / At-Risk Customers",  # low income & spend, high recency (inactive)
    1: "Premium Customers (Store-Focused)",  # high income & spend, older, store-heavy
    2: "Digital Buyers",  # moderate-high spend, high web purchase share
    3: "Budget Customers (Recently Active)",  # low income & spend, but low recency
    4: "High-Value Customers",  # highest income & spend overall
}
