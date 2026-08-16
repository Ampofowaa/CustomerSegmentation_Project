# Recommendations Applied

Issues identified in the original `Analysis_Model.ipynb` / `segmentation.py`,
and where each is now handled in this codebase.

| # | Issue | Where it was | Fix / where it now lives |
|---|---|---|---|
| 1 | Missing income rows silently dropped with no record of how many | `df.dropna(inplace=True)` | `src/data/loader.py::clean_data()` logs the count and % dropped |
| 2 | No outlier handling for implausible birth years (e.g. `Year_Birth=1900`) — only inspected, never acted on | `df.loc[df['Age']> 90]` | `clean_data()` caps `Year_Birth` at `config.AGE_MAX` (90), logged |
| 3 | No outlier handling for the extreme income value (~$666k) that skews K-Means (Euclidean distance is sensitive to scale outliers) | not handled | `clean_data()` caps `Income` at `config.INCOME_MAX` |
| 4 | Chained assignment bug: `df['Age'] = df['current_year'] = ...` silently creates an unused `current_year` column | feature engineering cell | `src/features/engineer.py::add_age()` — single clean assignment; regression test in `tests/test_pipeline.py` |
| 5 | K-Means has no `random_state` — cluster assignments and even cluster *count-to-meaning* mapping can change on every re-run | elbow-method loop and final `KMeans(n_clusters=5)` | `config.RANDOM_STATE = 42` + `config.N_INIT = 10`, applied in every `KMeans(...)` call in `src/models/train.py` |
| 6 | K chosen by elbow method only — subjective, hard to defend to stakeholders | single `wcss` loop | `src/models/train.py::evaluate_k_range()` also computes **silhouette score** per k |
| 7 | Cluster-naming markdown references "Cluster 6", which can't exist with `n_clusters=5` (labels are 0–4) | markdown cell | `config.CLUSTER_NAMES` is keyed 0–4; naming happens once, in one place, instead of a prose note that can drift from the actual model |
| 8 | Streamlit app rebuilds its own `DataFrame` and calls `scaler.transform()` directly — column order isn't guaranteed to match what the model was trained on | `segmentation.py` | `app/segmentation.py` now calls `src/models/predict.py::predict_segment()`, the **same** function that could be used to validate training — one code path, not two |
| 9 | Cluster output is a bare integer (`Cluster 0`) — meaningless to a non-technical user | `st.success(f'Predicted Segment: Cluster {cluster}')` | `predict_segment()` returns both the id and `config.CLUSTER_NAMES[cluster]` |
| 10 | No automated tests anywhere | none | `tests/test_pipeline.py` covers cleaning, feature engineering, and the model-matrix contract |
| 11 | No record of *why* a given k / model was chosen once trained | none | `src/models/train.py::save_artifacts()` writes `models/training_metrics.json` (silhouette score, cluster sizes, row count) alongside the model file |
| 12 | Feature list duplicated in the notebook and the app, with no guarantee they match | both files independently define feature lists | `config.FEATURES` defined once, imported everywhere |
| 13 | No stated business goal — the notebook jumped straight into EDA with no record of what decision the segmentation is meant to drive, who uses it, or what "good" looks like | none | Notebook opens with a **Business Goal** markdown section (problem, goal, who uses it, success criteria) before any code runs — feature choices, k, and cluster names are justified against it |
| 14 | Scaler and KMeans were fit and saved as two separate objects (`scaler.pkl` + `kmeans_model.pkl`) — nothing enforces they stay paired, so a partial re-save (e.g. retrain the model, forget the scaler) fails silently | `train.py` saved two joblib files; `segmentation.py` loaded both separately | `src/models/train.py::build_pipeline()` combines `StandardScaler` + `KMeans` into one `sklearn.pipeline.Pipeline`, fit and saved as a single `models/segmentation_pipeline.pkl`. Also improves reproducibility: `evaluate_k_range()` fits a full pipeline per candidate k (not a bare `KMeans` on pre-scaled data), so the k-search loop exercises the exact same code path as final training |

## Not yet done (good next steps, not included to keep this scope-bounded)

- **Virtual Environment**: use uvlock, pyproject.toml?
- **CI/CD** - pre-commit.config-yaml + github actions?
- **DVC** - instead  of scripts train folder - use dvc to run thigs
- **README** - Business Goals Context of the dataset - columns, meanings, number, source 
- **Industry tip**: write down the feature choice rationale in a markdown cell or README. Six months later nobody remembers why NumDealsPurchases or Kidhome were excluded.
- Notebook edits
- **Cluster stability check**: re-fit with several `random_state` values
  (not just re-running with the same one) and compare assignments (e.g.
  Adjusted Rand Index) to confirm the clusters aren't just an artifact of
  one particular initialization.
- **Monitoring / drift detection**: if this runs in production against new
  customer data over time, track whether cluster sizes/centroids drift and
  whether the model needs periodic re-training.
- **Config validation**: a `pydantic` schema or simple assertions so
  `config.py` fails loudly if `FEATURES` references a column that doesn't
  exist in the data.
- **Batch scoring script**: `scripts/train.py` exists for training; a
  `scripts/score_batch.py` that runs `predict_segment()` over an entire CSV
  and writes labeled output would round out the pipeline for offline use.


