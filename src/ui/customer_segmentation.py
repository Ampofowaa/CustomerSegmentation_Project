import io
import os
import re
import sys
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st

# `streamlit run` only puts this file's folder on sys.path; add the repo root
# so `config` and `src` import the same way they do everywhere else.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from src.data.loader import load_clean_data  # noqa: E402
from src.features.engineer import build_model_matrix, engineer_features  # noqa: E402
from src.models.predict import predict_batch, predict_segment  # noqa: E402
from src.models.train import fit_pipeline  # noqa: E402

PREVIEW_ROW_LIMIT = 200
SAMPLE_CSV_PATH = config.PROJECT_ROOT / "sample_batch_customers.csv"

# Set by Docker Compose (http://api:8000), or locally to http://localhost:8000,
# to call the FastAPI service. Unset (e.g. on Streamlit Community Cloud), the
# app scores in-process with the same src.models.predict functions the API uses.
API_URL = os.environ.get("API_URL")


class ScoringError(Exception):
    """A user-facing scoring failure, from either the API or in-process path."""


@st.cache_resource(show_spinner="Training the model (first start only)...")
def ensure_model() -> None:
    """Train and save the pipeline once if no model file exists yet.

    The .pkl isn't committed, so a fresh deployment has none. This is a plain
    fit (same features and settings as scripts/train.py), without the
    stability checks or metrics file, so startup stays fast."""
    if config.PIPELINE_PATH.exists():
        return
    X = build_model_matrix(engineer_features(load_clean_data()))
    pipeline, _, _ = fit_pipeline(X, use_pca=True)
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, config.PIPELINE_PATH)


def _api_error(response: requests.Response, e: Exception) -> ScoringError:
    detail = response.json().get("detail", str(e)) if response.content else str(e)
    return ScoringError(str(detail))


def score_one(payload: dict) -> tuple[int, str]:
    """Return (0-indexed cluster, label) for one customer."""
    if API_URL is None:
        try:
            result = predict_segment(payload)
        except ValueError as e:
            raise ScoringError(str(e))
        return result["cluster"], result["cluster_name"]

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=20)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ScoringError(f"Could not reach the prediction API at {API_URL}. Is it running?")
    except requests.exceptions.HTTPError as e:
        raise _api_error(response, e)
    result = response.json()
    return result["cluster"], result["label"]


def score_csv(name: str, content: bytes) -> tuple[bytes, str]:
    """Return (scored CSV bytes, download file name) for an uploaded CSV."""
    if API_URL is None:
        try:
            scored = predict_batch(pd.read_csv(io.BytesIO(content)))
        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            raise ScoringError(str(e))
        # Same display-only 1-5 shift the API applies to its CSV output.
        scored["cluster"] = scored["cluster"] + 1
        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d_%H-%M-%S")
        return scored.to_csv(index=False).encode(), f"segmented_customers_{stamp}.csv"

    try:
        files = {"file": (name, content, "text/csv")}
        response = requests.post(f"{API_URL}/predict/batch", files=files, timeout=60)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ScoringError(f"Could not reach the prediction API at {API_URL}. Is it running?")
    except requests.exceptions.HTTPError as e:
        raise _api_error(response, e)

    disposition = response.headers.get("content-disposition", "")
    match = re.search(r"filename=([^;]+)", disposition)
    return response.content, match.group(1).strip() if match else "segmented_customers.csv"


if API_URL is None:
    ensure_model()

st.title("Customer Segmentation App")

single_tab, batch_tab = st.tabs(["Single Prediction", "Batch Prediction"])

with single_tab:
    st.write("Enter customer details to predict the segment")

    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    income = st.number_input("Income", min_value=0, max_value=200_000, value=50_000)
    recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)
    customer_tenure = st.number_input("Customer Tenure (days since signup)", min_value=0, max_value=20_000, value=4_800)
    total_spending = st.number_input("Total Spending (sum of purchases)", min_value=0, max_value=5_000, value=1_000)
    num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)
    num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=10)
    num_catalog_purchases = st.number_input("Number of Catalog Purchases", min_value=0, max_value=100, value=4)
    num_web_visits = st.number_input("Number of Web Visits per Month", min_value=0, max_value=50, value=3)
    num_deals_purchases = st.number_input("Number of Deal Purchases", min_value=0, max_value=100, value=2)
    teenhome = st.number_input("Number of Teenagers in Household", min_value=0, max_value=5, value=0)
    kidhome = st.number_input("Number of Young Children in Household", min_value=0, max_value=5, value=0)

    if st.button("Predict Segment"):
        payload = {
            "Age": age,
            "Income": income,
            "Recency": recency,
            "Customer_Tenure": customer_tenure,
            "Total_Spending": total_spending,
            "NumWebPurchases": num_web_purchases,
            "NumStorePurchases": num_store_purchases,
            "NumCatalogPurchases": num_catalog_purchases,
            "NumWebVisitsMonth": num_web_visits,
            "NumDealsPurchases": num_deals_purchases,
            "Teenhome": teenhome,
            "Kidhome": kidhome,
        }
        try:
            cluster, label = score_one(payload)
            # Display-only: the `cluster` id stays 0-indexed (matches
            # config.CLUSTER_NAMES); shift to 1-5 here since this is the
            # number a marketer actually reads.
            st.success(f"Predicted Segment: {label} (Cluster {cluster + 1})")
        except ScoringError as e:
            st.error(f"Prediction failed: {e}")

with batch_tab:
    st.write(
        "Upload a CSV with (at least) the columns Age, Income, Recency, "
        "Customer_Tenure, Total_Spending, NumWebPurchases, NumStorePurchases, "
        "NumCatalogPurchases, NumWebVisitsMonth, NumDealsPurchases, Teenhome, "
        "Kidhome — one row per customer. Any other columns (e.g. a customer ID) "
        "are kept in the output."
    )

    sample_bytes = SAMPLE_CSV_PATH.read_bytes()
    st.caption("No file handy? Score the 8-customer sample, or download it to see the expected format.")
    sample_col, download_col = st.columns(2)
    use_sample = sample_col.button("Score the sample file")
    download_col.download_button(
        "Download sample CSV",
        data=sample_bytes,
        file_name=SAMPLE_CSV_PATH.name,
        mime="text/csv",
        on_click="ignore",
    )

    uploaded_file = st.file_uploader("Customer CSV", type="csv")

    to_score: tuple[str, bytes] | None = None
    if use_sample:
        to_score = (SAMPLE_CSV_PATH.name, sample_bytes)
    elif uploaded_file is not None and st.button("Score CSV"):
        to_score = (uploaded_file.name, uploaded_file.getvalue())

    if to_score is not None:
        try:
            content, file_name = score_csv(*to_score)
            st.success("Scored successfully.")

            st.download_button(
                "Download segmented CSV",
                data=content,
                file_name=file_name,
                mime="text/csv",
                on_click="ignore",
            )

            scored_df = pd.read_csv(io.BytesIO(content))
            if len(scored_df) > PREVIEW_ROW_LIMIT:
                st.caption(
                    f"Showing the first {PREVIEW_ROW_LIMIT} of {len(scored_df)} rows. "
                    "Download the CSV for the full result."
                )
                st.dataframe(scored_df.head(PREVIEW_ROW_LIMIT))
            else:
                st.dataframe(scored_df)
        except ScoringError as e:
            st.error(f"Batch scoring failed: {e}")
