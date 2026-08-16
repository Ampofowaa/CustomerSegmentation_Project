import os

import requests
import streamlit as st

# In Docker Compose this resolves to the "api" service name; override
# locally with API_URL=http://localhost:8000 if running outside Compose.
API_URL = os.environ.get("API_URL", "http://api:8000")

st.title("Customer Segmentation App")
st.write("Enter customer details to predict the segment")

age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Income", min_value=0, max_value=200_000, value=50_000)
total_spending = st.number_input(
    "Total Spending (sum of purchases)", min_value=0, max_value=5_000, value=1_000
)
num_web_purchases = st.number_input(
    "Number of Web Purchases", min_value=0, max_value=100, value=10
)
num_store_purchases = st.number_input(
    "Number of Store Purchases", min_value=0, max_value=100, value=10
)
num_web_visits = st.number_input(
    "Number of Web Visits per Month", min_value=0, max_value=50, value=3
)
recency = st.number_input(
    "Recency (days since last purchase)", min_value=0, max_value=365, value=30
)

if st.button("Predict Segment"):
    payload = {
        "Age": age,
        "Income": income,
        "Total_Spending": total_spending,
        "NumWebPurchases": num_web_purchases,
        "NumStorePurchases": num_store_purchases,
        "NumWebVisitsMonth": num_web_visits,
        "Recency": recency,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        st.success(
            f"Predicted Segment: {result['label']} (Cluster {result['cluster']})"
        )
    except requests.exceptions.ConnectionError:
        st.error(f"Could not reach the prediction API at {API_URL}. Is it running?")
    except requests.exceptions.HTTPError as e:
        detail = response.json().get("detail", str(e)) if response.content else str(e)
        st.error(f"Prediction failed: {detail}")
