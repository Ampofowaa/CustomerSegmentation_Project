# Customer Segmentation

[![CI](https://github.com/Ampofowaa/CustomerSegmentation_Project/actions/workflows/ci.yaml/badge.svg)](https://github.com/Ampofowaa/CustomerSegmentation_Project/actions/workflows/ci.yaml)

A K-Means customer segmentation pipeline that clusters retail customers by
purchasing behavior and demographics, served behind a FastAPI prediction
endpoint with a Streamlit front end. Restructured from a single exploratory
notebook into a notebook-for-analysis / package-for-production layout.

## Project structure

```
customer_segmentation/
├── config.py                   # single source of truth: paths, features, hyperparameters
├── data/
│   └── customer_segmentation.csv
├── notebooks/
│   └── Analysis_Model.ipynb    # EDA & experimentation ONLY — imports from src/, doesn't duplicate logic
├── src/                        # the actual pipeline — importable, testable
│   ├── data/
│   │   └── loader.py           # load_raw_data(), clean_data()
│   ├── features/
│   │   └── engineer.py         # engineer_features(), build_model_matrix()
│   ├── models/
│   │   ├── train.py            # build_pipeline(), fit_pipeline(), evaluate_k_range(), save_artifacts()
│   │   └── predict.py          # load_pipeline(), predict_segment() — used by BOTH training checks and the API
│   ├── pipeline.py             # orchestrates load -> clean -> engineer -> fit -> save
│   ├── api/
│   │   ├── main.py             # FastAPI app, imports src.models.predict
│   │   └── schema.py           # request/response pydantic models
│   └── ui/
│       └── customer_segmentation.py   # Streamlit app, calls the FastAPI service
├── scripts/
│   └── train.py                 # CLI: python scripts/train.py
├── models/                      # saved artifacts (segmentation_pipeline.pkl, training_metrics.json)
├── docker/
│   ├── Dockerfile.fastapi
│   └── Dockerfile.streamlit
├── tests/
│   └── test_pipeline.py         # pytest — covers the exact bugs found in the original notebook
├── docker-compose.yml
├── dvc.yaml                     # DVC pipeline definition (wraps scripts/train.py)
├── .github/workflows/ci.yaml    # lint + test on push/PR
└── pyproject.toml
```

`api/` and `ui/` live under `src/` alongside the training code rather than
as separate top-level import roots — one package to install, lint,
type-check, and Dockerize. The notebook opens with a Business Goal section
(what decision this segmentation drives, who uses it, what "good" looks
like) before any code runs, so feature choices, k, and cluster names get
justified against that goal rather than picked for technical convenience.

## Dataset

[Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)
(`data/customer_segmentation.csv`) — 2,240 customers of a retail company,
one row per customer. Columns cover:

- **Demographics**: birth year, education, marital status, income, kids/teens at home
- **Behavior**: recency, spend by product category (wine, fruit, meat, fish, sweets, gold), purchases by channel (web, catalog, store), web visits/month
- **Campaign history**: response to 5 past campaigns, complaints

`src/data/loader.py` drops the ~1% of rows with missing `Income` and caps
two known outliers (a handful of implausible birth years, one $666,666
income value) — see `config.AGE_MAX` / `config.INCOME_MAX`. `src/features/engineer.py`
derives `Age` and `Total_Spending` and assembles the final `config.FEATURES`
set (7 features) that the model actually trains on. 2,216 of the original
2,240 rows survive cleaning.

### Why these 7 features (and not the rest)

`config.FEATURES = [Age, Income, Total_Spending, NumWebPurchases,
NumStorePurchases, NumWebVisitsMonth, Recency]`. Everything else in the raw
dataset was considered and dropped for a specific reason, not by accident —
see the "Feature Selection Rationale" cell in `notebooks/Analysis_Model.ipynb`
for the correlation numbers behind these calls:

- **`Kidhome` / `Teenhome` (and the engineered `TotalNo_Children`)** —
  strongly correlated with `Income` (r ≈ -0.51) and `Total_Spending`
  (r ≈ -0.50). K-Means clusters on Euclidean distance, so keeping a feature
  that's just a proxy for income/spend would double-count that signal.
  `add_total_children()` in `src/features/engineer.py` still computes
  `TotalNo_Children` — it's available for exploration/reporting, it just
  isn't part of the model matrix.
- **`NumCatalogPurchases`** — highly correlated with `Income` (r ≈ 0.69) and
  `Total_Spending` (r ≈ 0.78); `NumWebPurchases` and `NumStorePurchases`
  already cover purchase-channel behavior, so this is redundant.
- **`NumDealsPurchases`** — weak correlation with every other feature
  (|r| < 0.11). It doesn't separate customers into meaningfully different
  groups, so it's mostly noise for a distance-based algorithm.
- **`AcceptedCmp1`–`5` and `Response`** — these record how customers reacted
  to *past* campaigns. Using them to build segments that then drive *future*
  targeting would leak old targeting decisions into new ones.
- **`Complain`** — 21 of 2,216 rows are 1; near-zero variance, contributes
  almost nothing to distance-based clustering.
- **`Customer_Tenure`** (engineered in `add_customer_tenure()`) — weakly
  correlated with `Income`/`Total_Spending` (r ≈ 0.16) and largely redundant
  with `Recency` for capturing engagement. Computed but not fed to the model.
- **`Education`, `Marital_Status`** — categorical; K-Means needs numeric
  distances, and one-hot-encoding them would add several low-signal
  dimensions relative to the behavioral/value features already in play.

## Results

The current `models/segmentation_pipeline.pkl` (`StandardScaler` + `KMeans`,
`k=5`, `random_state=42`) was fit on those 2,216 cleaned rows:

| Cluster | Label                              | Size | Share |
|--------:|-------------------------------------|-----:|------:|
| 3       | Budget Customers (Recently Active)  |  562 |  25%  |
| 0       | Low-Engagement / At-Risk Customers  |  531 |  24%  |
| 2       | Digital Buyers                      |  454 |  20%  |
| 4       | High-Value Customers                |  341 |  15%  |
| 1       | Premium Customers (Store-Focused)   |  328 |  15%  |

Silhouette score: **0.188**. That's modest in absolute terms, which is
typical for demographic/behavioral customer data where segments overlap
rather than form tight, well-separated blobs — `evaluate_k_range()` in
`src/models/train.py` was used to check `k=5` against the inertia/silhouette
trade-off across a range of k before settling on it. The clusters are
judged less by that number and more by whether the per-cluster profiles are
distinct and actionable (e.g. `High-Value Customers` have the highest
income and spend; `Low-Engagement / At-Risk Customers` have low spend and
the highest recency, i.e. haven't purchased in a while) — see
`cluster_profile()` in `src/models/train.py` and `config.CLUSTER_NAMES` for
how cluster IDs were mapped to labels.

These numbers regenerate on every `scripts/train.py` / `dvc repro` run and
are written to `models/training_metrics.json` — check that file for the
current run's exact values rather than assuming this table stays in sync.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
- Docker + Docker Compose (optional, for containerized serving)

## Setup

```bash
git clone <repo-url> && cd customer_segmentation

# Install everything (core + api + ui + notebook extras + dev tools)
uv sync --all-extras
```

## Running it

```bash
# Train and save the model
uv run python scripts/train.py
uv run python scripts/train.py --n-clusters 6   # override k

# ...or via DVC, which only re-runs the stage if its tracked deps changed
uv run dvc repro

# Run tests
uv run pytest

# Lint / format / type-check
uv run ruff check .
uv run black --check .
uv run mypy src scripts tests

# Run the API locally (needs a trained model in models/)
uv run uvicorn src.api.main:app --reload

# Run the Streamlit app locally (needs the API running)
uv run streamlit run src/ui/customer_segmentation.py
```

## Running with Docker

```bash
docker compose up -d --build
```

| Service   | URL                          |
|-----------|-------------------------------|
| FastAPI   | http://localhost:8100/docs   |
| Streamlit | http://localhost:8600         |

Host ports are set in `docker-compose.yml` — change the left side of the
`ports:` mapping if `8100`/`8600` conflict with something else already
running locally; the containers still talk to each other over the internal
Docker network regardless of host port.

## API

`POST /predict`

```bash
curl -X POST http://localhost:8100/predict \
  -H "Content-Type: application/json" \
  -d '{
        "Age": 35,
        "Income": 50000,
        "Total_Spending": 1000,
        "NumWebPurchases": 10,
        "NumStorePurchases": 10,
        "NumWebVisitsMonth": 3,
        "Recency": 30
      }'
```

```json
{"cluster": 4, "label": "High-Value Customers"}
```

## CI

`.github/workflows/ci.yaml` runs on every push and pull request: `ruff`,
`black --check`, `mypy`, and `pytest`, all via `uv`.
