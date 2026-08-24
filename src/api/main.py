import io
import logging

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from src.api.schema import CustomerFeatures, SegmentPrediction
from src.models.predict import load_pipeline, predict_batch, predict_segment

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Segmentation API",
    description="Predicts a marketing segment for a given customer profile.",
    version="1.0.0",
)


@app.on_event("startup")
def warm_up_model() -> None:
    """Load the pipeline into memory before accepting requests, so the
    (disk read + unpickle) cost is paid once at boot instead of on
    whichever user's request happens to be first.

    Missing-model is swallowed here (not re-raised) so a container without
    a trained model still starts and serves the same clean 500 from
    /predict and /predict/batch below, rather than crashing at boot."""
    try:
        load_pipeline()
    except FileNotFoundError:
        logger.warning(
            "No model pipeline found at startup. /predict and /predict/batch "
            "will 500 until scripts/train.py (or dvc repro) has been run."
        )


@app.post("/predict", response_model=SegmentPrediction)
def predict(customer: CustomerFeatures) -> SegmentPrediction:
    try:
        result = predict_segment(customer.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Model pipeline not found. Has scripts/train.py been run?",
        )

    return SegmentPrediction(cluster=result["cluster"], label=result["cluster_name"])


@app.post("/predict/batch")
def predict_batch_endpoint(file: UploadFile = File(...)) -> StreamingResponse:
    """
    Score every row in an uploaded CSV. The CSV must contain at least the
    config.FEATURES columns; any other columns (e.g. a customer ID) are
    echoed back untouched alongside the new "cluster" / "cluster_name"
    columns. Responds with a downloadable CSV.
    """
    if file.filename is not None and not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="File must be a .csv")

    try:
        df = pd.read_csv(file.file)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")

    try:
        scored = predict_batch(df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Model pipeline not found. Has scripts/train.py been run?",
        )

    # Display-only: config.CLUSTER_NAMES and every internal cluster ID stay
    # 0-indexed (predict_batch() above uses them as-is); this CSV is what a
    # marketer opens directly, so shift to a 1-5 label here at the boundary.
    scored["cluster"] = scored["cluster"] + 1

    buffer = io.StringIO()
    scored.to_csv(buffer, index=False)
    buffer.seek(0)

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d_%H-%M-%S")
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=segmented_customers_{stamp}.csv"},
    )
