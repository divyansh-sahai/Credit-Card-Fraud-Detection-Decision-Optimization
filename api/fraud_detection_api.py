
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Locate the project and saved model
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "fraud_random_forest.joblib"

# Load the trained model and metadata
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
threshold = artifact["threshold"]
feature_names = artifact["feature_names"]

# Initialize the API
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Fraud probability and classification using a trained Random Forest model.",
    version="1.0.0"
)


# Define the expected request structure
class Transaction(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Transaction features: Time, V1-V28, and Amount"
    )


@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API",
        "status": "running",
        "model": artifact["model_name"]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "threshold": threshold
    }


@app.post("/predict")
def predict(transaction: Transaction):
    supplied_features = transaction.features

    missing = set(feature_names) - set(supplied_features)
    unexpected = set(supplied_features) - set(feature_names)

    if missing or unexpected:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": sorted(missing),
                "unexpected_features": sorted(unexpected)
            }
        )

    if supplied_features["Amount"] < 0 or supplied_features["Time"] < 0:
        raise HTTPException(
            status_code=422,
            detail="Time and Amount must be non-negative."
        )

    input_df = pd.DataFrame(
        [[supplied_features[name] for name in feature_names]],
        columns=feature_names
    )

    probability = float(model.predict_proba(input_df)[0, 1])
    prediction = int(probability >= threshold)

    return {
        "fraud_probability": round(probability, 6),
        "fraud_prediction": prediction,
        "risk_classification": "Fraud" if prediction == 1 else "Legitimate",
        "decision_threshold": threshold
    }
