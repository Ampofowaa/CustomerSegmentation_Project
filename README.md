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

## Why this layout

- **Business goal drives the modeling, not the other way around.** The
  notebook opens with a Business Goal section — what decision this
  segmentation is meant to drive, who uses it, and how success is judged —
  before any code runs. Feature choices, k, and cluster names all get
  justified against that goal rather than chosen for technical convenience.
- **Notebook = analysis, package = production.** The notebook is for
  exploring the data and justifying decisions (why 5 clusters, which
  features). Nothing in production imports from the notebook — it imports
  from `src/`. If you find a better feature or cleaning step in the
  notebook, you move it into `src/` deliberately, not by copy-pasting
  notebook cells into a script.
- **One package root.** `api/` and `ui/` live under `src/` alongside the
  training code, rather than as separate top-level import roots — one
  package to install, lint, type-check, and Dockerize.
- **One feature contract.** `config.FEATURES` is defined once. Training
  (`src/pipeline.py`) and inference (`src/models/predict.py`) both build
  their matrix from it, in the same order. This is what the original
  `segmentation.py` was missing — it manually rebuilt the DataFrame and
  relied on hoping the column order matched what the notebook trained on.
- **One pipeline artifact, not a scaler+model pair.** `src/models/train.py`
  builds an `sklearn.pipeline.Pipeline` combining `StandardScaler` and
  `KMeans`, fits it as a single unit, and saves it as one file
  (`models/segmentation_pipeline.pkl`). Two separate artifacts can silently
  drift apart — retrain the model, forget to re-save the scaler, ship a
  stale one — and nothing catches it because `transform()` doesn't validate
  against the model it's paired with. A pipeline makes "scale, then
  cluster" one atomic, versioned unit: load it, call `.predict()`, done.
- **Testable units.** `clean_data`, `engineer_features`, `build_model_matrix`,
  and the pipeline itself are covered by unit tests, including a
  reproducibility test (same data + same `random_state` -> identical
  cluster assignments) and a test that `pipeline.predict()` matches scaling
  and predicting manually. A future change (e.g. a new outlier rule) gets
  caught by `pytest` before it reaches production, instead of being
  discovered when cluster predictions look wrong in the app.
- **Reproducibility.** `random_state` + `n_init` are set once in `config.py`
  and baked into every `KMeans` instantiation (including inside the k-search
  loop), so re-training produces the same clusters given the same data —
  required if you ever need to explain to a stakeholder why a customer's
  segment changed. `dvc.yaml` tracks the training stage's inputs and outputs
  so `dvc repro` only re-trains when data or code actually changed, and
  `models/segmentation_pipeline.pkl` is DVC-tracked rather than committed to
  Git.

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

## See also

`RECOMMENDATIONS.md` — the specific issues found in the original notebook
and app, and how each was addressed in this structure.
