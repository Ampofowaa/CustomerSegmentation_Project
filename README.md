# Customer Segmentation

[![CI](https://github.com/Ampofowaa/CustomerSegmentation_Project/actions/workflows/ci.yaml/badge.svg)](https://github.com/Ampofowaa/CustomerSegmentation_Project/actions/workflows/ci.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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
derives `Age`, `Total_Spending`, and `Customer_Tenure`, and assembles the
final `config.FEATURES` set (12 features) that the model actually trains
on. 2,216 of the original 2,240 rows survive cleaning.

### Why these 12 features (and not the rest)

`config.FEATURES = [Age, Income, Recency, Customer_Tenure, Total_Spending,
NumWebPurchases, NumStorePurchases, NumCatalogPurchases, NumWebVisitsMonth,
NumDealsPurchases, Teenhome, Kidhome]`. Everything else in the raw dataset
was considered and dropped for a specific reason, not by accident — see the
"Feature Selection" section in `notebooks/Analysis_Model.ipynb` (grouped by
theme: demographics, monetary value, recency/tenure, purchase channel,
promotion responsiveness) for the correlation numbers behind these calls:

- **`TotalNo_Children`** (engineered from `Kidhome` + `Teenhome`) — blends
  `Kidhome`'s real signal with `Teenhome`'s near-neutral one, diluting
  rather than combining. `Kidhome` and `Teenhome` are kept individually
  instead — each earns its place separately via a paired silhouette check,
  not by mechanical correlation cutoff.
- **`Avg_Spend_Per_Purchase`** (engineered: `Total_Spending` / total
  purchases) — r ≈ 0.93 with `Total_Spending`, the strongest pairwise
  correlation in the candidate pool, and its broader correlation profile is
  essentially `Total_Spending`'s restated at a different scale. A rescaling,
  not a new axis.
- **`AcceptedCmp1`–`5` and `Response`** (aggregated into
  `Total_Campaigns_Accepted`) — these record how customers reacted to
  *past* campaigns. Using them to build segments that then drive *future*
  targeting would leak old targeting decisions into new ones.
- **`Complain`** — 21 of 2,216 rows are 1; near-zero variance, contributes
  almost nothing to distance-based clustering (even though its correlation
  profile is about as clean as any candidate's).
- **`Education`, `Marital_Status`** — categorical; K-Means needs numeric
  distances. `Marital_Status` has no natural order, so one-hot-encoding is
  the only honest representation, and that would add several low-signal
  dimensions relative to the behavioral/value features already in play.
  `Education` does have a natural order (`Basic` < `2n Cycle` <
  `Graduation` < `Master` < `PhD`), so an ordinal encoding is a legitimate
  option worth reconsidering in a future iteration — it just hasn't been
  added yet.

`NumCatalogPurchases` is the one feature kept *despite* a negative paired
silhouette check (mean Δ -0.0049) — it names a real, distinct shopping
channel on par with web and store, and that business case outweighs the
statistical evidence, the same call already made for `Income`,
`NumStorePurchases`, `NumWebPurchases`, and `Kidhome` elsewhere in that
section.

## Results

The current `models/segmentation_pipeline.pkl` (`StandardScaler` + `PCA`
(`config.PCA_VARIANCE = 0.90`, 12 features -> 8 components) + `KMeans`,
`k=5`, `random_state=42`) was fit on those 2,216 cleaned rows:

| Cluster | Label                                          | Size | Share |
|--------:|-------------------------------------------------|-----:|------:|
| 3       | Low-Income Young Families (Browsing, Not Buying) |  544 |  25%  |
| 1       | Premium Teen-Family Shoppers (Store-Focused)     |  520 |  23%  |
| 0       | High-Value Customers (No Children)               |  500 |  23%  |
| 4       | Budget Multi-Child Households                    |  454 |  20%  |
| 2       | Deal-Seeking Large Families                      |  198 |   9%  |

Silhouette score: **0.200** (up from 0.180 without the PCA step). Still
modest in absolute terms, which is typical for demographic/behavioral
customer data where segments overlap rather than form tight, well-separated
blobs — `evaluate_k_range()` in `src/models/train.py` was used to check
`k=5` against the inertia/silhouette trade-off across a range of k before
settling on it (silhouette alone peaks at `k=2`, which is too coarse to act
on for this project's targeting use cases). Adding PCA was a deliberate
experiment prompted by the gap between a low silhouette and a very high
cluster-stability ARI (>0.99 across seeds, see `check_cluster_stability()`):
that combination pointed at collinearity between the spend/purchase-count
features diluting separation rather than the segments being noise, so PCA
was added to strip that collinearity before KMeans. It raised silhouette
modestly and *increased* stability (seed ARI 0.992->0.998, bootstrap ARI
0.946->0.948) without materially changing which customers land in which
segment (ARI of 0.96 between the old and new label sets) — set
`use_pca=False` in `run_training_pipeline()` / `fit_pipeline()` to compare
against the pre-PCA baseline. The clusters are judged less by the silhouette
number and more by whether the per-cluster profiles are distinct and
actionable (e.g. `High-Value Customers` have the highest income, spend, and
catalog usage; `Deal-Seeking Large Families` are the smallest segment but by
far the heaviest users of discounts) — see `cluster_profile()` in
`src/models/train.py` and `config.CLUSTER_NAMES` for how cluster IDs were
mapped to labels.

These numbers regenerate on every `scripts/train.py` / `dvc repro` run and
are written to `models/training_metrics.json` — check that file for the
current run's exact values rather than assuming this table stays in sync.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
- Docker + Docker Compose (optional, for containerized serving)

## Setup

```bash
git clone https://github.com/Ampofowaa/CustomerSegmentation_Project.git && cd customer_segmentation

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

The API image bakes `models/` into it at build time (`COPY models/ ./models/`
in `docker/Dockerfile.fastapi`, no runtime volume mount) — train first so
that directory exists and holds a current pipeline:

```bash
uv run python scripts/train.py   # or: uv run dvc repro

docker compose up -d --build
```

Rebuild (`--build`) any time you retrain locally — otherwise the containers
keep serving whatever model was baked in at the last build.

| Service   | URL                          |
|-----------|-------------------------------|
| FastAPI   | http://localhost:8100/docs   |
| Streamlit | http://localhost:8600         |

Host ports are set in `docker-compose.yml` — change the left side of the
`ports:` mapping if `8100`/`8600` conflict with something else already
running locally; the containers still talk to each other over the internal
Docker network regardless of host port.

The Streamlit app has two tabs:
- **Single Prediction** — a form for one customer, calls `POST /predict`.
- **Batch Prediction** — upload a CSV, calls `POST /predict/batch`, and
  download the scored result. `sample_batch_customers.csv` in the repo
  root is a ready-made file to try this with.

## API

`POST /predict`

```bash
curl -X POST http://localhost:8100/predict \
  -H "Content-Type: application/json" \
  -d '{
        "Age": 35,
        "Income": 50000,
        "Recency": 30,
        "Customer_Tenure": 4800,
        "Total_Spending": 1000,
        "NumWebPurchases": 10,
        "NumStorePurchases": 10,
        "NumCatalogPurchases": 4,
        "NumWebVisitsMonth": 3,
        "NumDealsPurchases": 2,
        "Teenhome": 0,
        "Kidhome": 0
      }'
```

```json
{"cluster": 0, "label": "High-Value Customers (No Children)"}
```

`POST /predict/batch` — upload a CSV with (at least) the same feature
columns; any extra columns (e.g. a customer ID) are echoed back. Returns a
downloadable CSV with `cluster`, `cluster_name`, and `scored_at` appended.
The CSV's `cluster` column is 1-5 (a display-only shift for the person
reading the file) — the `/predict` JSON response above stays 0-4, matching
`config.CLUSTER_NAMES`.

```bash
curl -X POST http://localhost:8100/predict/batch \
  -F "file=@sample_batch_customers.csv" \
  -o segmented_customers.csv
```

## CI

`.github/workflows/ci.yaml` runs on every push and pull request: `ruff`,
`black --check`, `mypy`, and `pytest`, all via `uv`.
