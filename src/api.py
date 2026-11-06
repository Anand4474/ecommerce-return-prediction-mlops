import os
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.features import build_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT/"artifacts"/"best_model.joblib"))

app = FastAPI(
    title="E-Commerce Product Return Prediction API",
    description="Predicts whether an order is likely to be returned.",
    version="1.0.0"
)

class OrderFeatures(BaseModel):
    Product_Category: str
    Product_Price_INR: float | None = Field(default=None, ge=0)
    Discount_Percent: float | None = Field(default=None, ge=0, le=100)
    Expected_Delivery_Days: int = Field(ge=0)
    Actual_Delivery_Days: float | None = Field(default=None, ge=0)
    Customer_Rating_PreOrder: float | None = Field(default=None, ge=0, le=5)
    Previous_Returns: int = Field(default=0, ge=0)
    Previous_Orders: int = Field(default=0, ge=0)
    Payment_Method: str | None = None
    Damaged_Package_Flag: str = "No"
    Package_Size: str = "Medium"
    Promotion_Active: str = "No"

@app.get("/")
def root():
    return {"message": "E-Commerce Return Prediction API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model_exists": MODEL_PATH.exists()}

@app.post("/predict")
def predict(order: OrderFeatures):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not found. Run python -m src.train first.")
    try:
        model = joblib.load(MODEL_PATH)
        row = pd.DataFrame([order.model_dump()])
        X = build_features(row)
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])
        return {
            "prediction": "Returned" if prediction == 1 else "Not Returned",
            "return_probability": round(probability, 4),
            "risk_level": "High" if probability >= 0.70 else ("Medium" if probability >= 0.40 else "Low")
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")
