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

# Historical + current campaign-acceptance flags, summed into a single
# marketing-responsiveness signal (Total_Campaigns_Accepted).
ACCEPTED_CMP_COLS = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "Response",
]

# Purchase-count columns, summed for Avg_Spend_Per_Purchase's denominator.
# Excludes NumWebVisitsMonth (browsing, not a completed purchase).
PURCHASE_COUNT_COLS = [
    "NumWebPurchases",
    "NumStorePurchases",
    "NumCatalogPurchases",
    "NumDealsPurchases",
]

# --- Final feature set fed to the model ---------------------------------
# ORDER MATTERS: StandardScaler.transform() only cares about column
# position, not names. Train and inference must build this list identically,
# which is exactly why it's defined once, here, and imported everywhere else.
FEATURES = [
    "Age",
    "Income",
    "Recency",
    "Customer_Tenure",
    "Total_Spending",
    "NumWebPurchases",
    "NumStorePurchases",
    "NumCatalogPurchases",
    "NumWebVisitsMonth",
    "NumDealsPurchases",
    "Teenhome",
    "Kidhome",
]

# --- Cleaning thresholds --------------------------------------------------
AGE_MAX = 90  # caps the birth-year data-entry errors (e.g. Year_Birth=1900)
INCOME_MAX = 200_000  # caps the ~$666,666 income outlier that skews K-Means

# --- Model hyperparameters ------------------------------------------------
N_CLUSTERS = 5
RANDOM_STATE = 42  # fixes centroid init -> reproducible clusters across runs
N_INIT = 10  # sklearn re-runs K-Means 10x and keeps the best (avoids bad local optima)
K_SEARCH_RANGE = range(2, 10)

# Fraction of variance PCA must retain when use_pca=True is passed to
# build_pipeline()/fit_pipeline(). A float here tells sklearn's PCA to pick
# however many components are needed to hit this threshold rather than a
# fixed component count, so it adapts if FEATURES changes.
PCA_VARIANCE = 0.90

# --- Human-readable cluster labels ----------------------------------------
# Set by inspecting cluster_profile() in the notebook after training — a
# cluster number means nothing to a business stakeholder, a name does.
# NOTE: cluster IDs are not stable across re-training runs unless
# random_state is fixed (it is, see RANDOM_STATE above) AND the training
# data doesn't change. If you retrain on new data, re-check this mapping
# against the new cluster_profile() output before trusting these labels.
# Re-derived after enabling PCA in the pipeline (see PCA_VARIANCE above) --
# the extra PCA step reshuffles which cluster ID lands on which segment,
# so these indices don't match a pre-PCA run's.
CLUSTER_NAMES = {
    0: "High-Value Customers (No Children)",  # highest income & spend, store-heavy, heaviest catalog usage
    1: "Premium Teen-Family Shoppers (Store-Focused)",  # older, high income & spend, almost entirely teen households
    2: "Deal-Seeking Large Families",  # mid income/spend, by far the heaviest deal usage, large families
    3: "Low-Income Young Families (Browsing, Not Buying)",  # youngest, lowest income/spend, high visits, low purchases
    4: "Budget Multi-Child Households",  # oldest, lowest income & spend, largest households (teens and young children)
}
