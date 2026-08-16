from api.schemas import CustomerFeatures, SegmentPrediction
from fastapi import FastAPI, HTTPException
from src.predict import predict_segment

app = FastAPI(
    title="Customer Segmentation API",
    description="Predicts a marketing segment for a given customer profile.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=SegmentPrediction)
def predict(customer: CustomerFeatures):
    try:
        result = predict_segment(customer.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model pipeline not found. Has scripts/train.py been run?",
        )

    return SegmentPrediction(cluster=result["cluster"], label=result["cluster_name"])
