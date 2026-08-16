# Customer Segmentation — Production Project Structure

A K-Means customer segmentation project, restructured from a single
exploratory notebook into a notebook-for-analysis / package-for-production
layout.

## Folder structure

```
customer_segmentation/
├── config.py                  # single source of truth: paths, features, hyperparameters
├── requirements.txt
├── data/
│   └── customer_segmentation.csv
├── notebooks/
│   └── Analysis_Model.ipynb   # EDA & experimentation ONLY — imports from src/, doesn't duplicate logic
├── src/                       # the actual pipeline — importable, testable
│   ├── data/
│   │   └── loader.py          # load_raw_data(), clean_data()
│   ├── features/
│   │   └── engineer.py        # engineer_features(), build_model_matrix()
│   ├── models/
│   │   ├── train.py           # build_pipeline(), fit_pipeline(), evaluate_k_range(), save_artifacts()
│   │   └── predict.py         # load_pipeline(), predict_segment() — used by BOTH training checks and the app
│   └── pipeline.py            # orchestrates load -> clean -> engineer -> fit -> save
├── scripts/
│   └── train.py                # CLI: python scripts/train.py
├── app/
│   └── segmentation.py         # Streamlit app, imports src.models.predict — no duplicated scaling logic
├── models/                     # saved artifacts (segmentation_pipeline.pkl, training_metrics.json)
└── tests/
    └── test_pipeline.py        # pytest — covers the exact bugs found in the original notebook
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
  segment changed.

## Running it

```bash
cd customer_segmentation
pip install -r requirements.txt

# Train and save the model
python scripts/train.py
python scripts/train.py --n-clusters 6   # override k

# Run tests
pytest

# Launch the app (needs a trained model in models/)
streamlit run app/segmentation.py
```

## See also

`RECOMMENDATIONS.md` — the specific issues found in the original notebook
and app, and how each was addressed in this structure.
