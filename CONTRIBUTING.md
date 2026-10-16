# Developer guide

Full setup, training, serving and API reference. For what the project does and why, see the [README](README.md); for modelling detail, see [docs/methodology.md](docs/methodology.md).

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management (exact versions are pinned in `uv.lock`)
- Docker + Docker Compose (optional, for containerized serving)

## Setup

```bash
git clone https://github.com/Ampofowaa/customer-segmentation.git
cd customer-segmentation

# Install everything (core + api + ui + notebook extras + dev tools)
uv sync --all-extras
```

## Running it

The trained model (`models/segmentation_pipeline.pkl`) is not committed, so
train once before starting the API.

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

# Run the Streamlit app on its own (scores in-process, trains a model on first start if none exists)
uv run streamlit run src/ui/customer_segmentation.py

# ...or have it call the local API instead
API_URL=http://localhost:8000 uv run streamlit run src/ui/customer_segmentation.py
```

Training metrics regenerate on every `scripts/train.py` / `dvc repro` run and
are written to `models/training_metrics.json`. `Age` and `Customer_Tenure` are
computed relative to today's date, so retraining later can shift the numbers
slightly.

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

## Deploying the demo (Streamlit Community Cloud)

With no `API_URL` set, the app uses the same `src.models.predict` functions
as the API, in-process, so it can be hosted on its own:

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub and choose **Create app → Deploy a public app from GitHub**.
2. Repository `Ampofowaa/customer-segmentation`, branch `main`, main file `src/ui/customer_segmentation.py`.
3. Under **Advanced settings**, choose Python 3.12. Leave secrets empty and don't set `API_URL`.
4. Deploy. Dependencies come from `src/ui/requirements.txt` (keep it in step with `uv.lock`).

The model file isn't committed, so the first start after each deploy or wake
from sleep trains it (under a minute) before the page loads.

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

## Full project structure

```
customer_segmentation/
├── config.py                   # single source of truth: paths, features, hyperparameters
├── data/
│   └── customer_segmentation.csv
├── docs/
│   ├── methodology.md          # feature selection, k choice, PCA experiment, segment profiles
│   └── images/                 # plots exported from the notebook
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
type-check, and Dockerize.
