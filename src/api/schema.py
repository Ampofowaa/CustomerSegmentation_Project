from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
                "Kidhome": 0,
            }
        }
    )

    Age: int = Field(..., ge=18, le=100, description="Customer age in years")
    Income: float = Field(..., ge=0, le=200_000, description="Annual income")
    Recency: int = Field(..., ge=0, le=365, description="Days since last purchase")
    Customer_Tenure: int = Field(..., ge=0, le=20_000, description="Days since the customer's Dt_Customer signup date")
    Total_Spending: float = Field(..., ge=0, le=5_000, description="Sum of spending across product categories")
    NumWebPurchases: int = Field(..., ge=0, le=100)
    NumStorePurchases: int = Field(..., ge=0, le=100)
    NumCatalogPurchases: int = Field(..., ge=0, le=100)
    NumWebVisitsMonth: int = Field(..., ge=0, le=50)
    NumDealsPurchases: int = Field(..., ge=0, le=100, description="Number of purchases made with a discount")
    Teenhome: int = Field(..., ge=0, le=5, description="Number of teenagers in the customer's household")
    Kidhome: int = Field(..., ge=0, le=5, description="Number of young children in the customer's household")


class SegmentPrediction(BaseModel):
    cluster: int
    label: str | None = None
