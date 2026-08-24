import io
import os
import re

import pandas as pd
import requests
import streamlit as st

PREVIEW_ROW_LIMIT = 200

# In Docker Compose this resolves to the "api" service name; override
# locally with API_URL=http://localhost:8000 if running outside Compose.
API_URL = os.environ.get("API_URL", "http://api:8000")

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
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=20)
            response.raise_for_status()
            result = response.json()
            # Display-only: the API's `cluster` field stays 0-indexed
            # (matches config.CLUSTER_NAMES); shift to 1-5 here since this
            # is the number a marketer actually reads.
            st.success(f"Predicted Segment: {result['label']} (Cluster {result['cluster'] + 1})")
        except requests.exceptions.ConnectionError:
            st.error(f"Could not reach the prediction API at {API_URL}. Is it running?")
        except requests.exceptions.HTTPError as e:
            detail = response.json().get("detail", str(e)) if response.content else str(e)
            st.error(f"Prediction failed: {detail}")

with batch_tab:
    st.write(
        "Upload a CSV with (at least) the columns Age, Income, Recency, "
        "Customer_Tenure, Total_Spending, NumWebPurchases, NumStorePurchases, "
        "NumCatalogPurchases, NumWebVisitsMonth, NumDealsPurchases, Teenhome, "
        "Kidhome — one row per customer. Any other columns (e.g. a customer ID) "
        "are kept in the output."
    )

    uploaded_file = st.file_uploader("Customer CSV", type="csv")

    if uploaded_file is not None and st.button("Score CSV"):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            response = requests.post(f"{API_URL}/predict/batch", files=files, timeout=60)
            response.raise_for_status()
            st.success("Scored successfully.")

            disposition = response.headers.get("content-disposition", "")
            match = re.search(r"filename=([^;]+)", disposition)
            file_name = match.group(1).strip() if match else "segmented_customers.csv"

            st.download_button(
                "Download segmented CSV",
                data=response.content,
                file_name=file_name,
                mime="text/csv",
            )

            scored_df = pd.read_csv(io.BytesIO(response.content))
            if len(scored_df) > PREVIEW_ROW_LIMIT:
                st.caption(
                    f"Showing the first {PREVIEW_ROW_LIMIT} of {len(scored_df)} rows. "
                    "Download the CSV for the full result."
                )
                st.dataframe(scored_df.head(PREVIEW_ROW_LIMIT))
            else:
                st.dataframe(scored_df)
        except requests.exceptions.ConnectionError:
            st.error(f"Could not reach the prediction API at {API_URL}. Is it running?")
        except requests.exceptions.HTTPError as e:
            detail = response.json().get("detail", str(e)) if response.content else str(e)
            st.error(f"Batch scoring failed: {detail}")
