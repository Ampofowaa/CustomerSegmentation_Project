from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    Age: int = Field(..., ge=18, le=100, description="Customer age in years")
    Income: float = Field(..., ge=0, le=200_000, description="Annual income")
    Total_Spending: float = Field(..., ge=0, le=5_000, description="Sum of spending across product categories")
    NumWebPurchases: int = Field(..., ge=0, le=100)
    NumStorePurchases: int = Field(..., ge=0, le=100)
    NumWebVisitsMonth: int = Field(..., ge=0, le=50)
    Recency: int = Field(..., ge=0, le=365, description="Days since last purchase")

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 35,
                "Income": 50000,
                "Total_Spending": 1000,
                "NumWebPurchases": 10,
                "NumStorePurchases": 10,
                "NumWebVisitsMonth": 3,
                "Recency": 30,
            }
        }


class SegmentPrediction(BaseModel):
    cluster: int
    label: str | None = None
