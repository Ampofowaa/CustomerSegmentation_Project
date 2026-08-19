import io

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from src.api.schema import CustomerFeatures, SegmentPrediction
from src.models.predict import predict_batch, predict_segment

app = FastAPI(
    title="Customer Segmentation API",
    description="Predicts a marketing segment for a given customer profile.",
    version="1.0.0",
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

    buffer = io.StringIO()
    scored.to_csv(buffer, index=False)
    buffer.seek(0)

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=segmented_customers_{stamp}.csv"},
    )
