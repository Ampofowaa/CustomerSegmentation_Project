from fastapi import FastAPI, HTTPException

from src.api.schema import CustomerFeatures, SegmentPrediction
from src.models.predict import predict_segment

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
