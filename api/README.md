
# Fraud Detection API

## Overview
A local REST API for credit card fraud prediction using a trained Random Forest classifier.

## Model
- Algorithm: Random Forest
- Decision threshold: 0.10
- Input features: 30 (`Time`, `V1`–`V28`, and `Amount`)

## Endpoints
- `GET /` — API information
- `GET /health` — Health status
- `POST /predict` — Returns fraud probability and classification

## Run locally
Activate the `fraud_detection` Conda environment, navigate to the project root, and run:

```bash
python -m uvicorn api.fraud_detection_api:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

## Automated tests
From the project root, run:

```bash
python -m pytest tests/test_fraud_detection_api.py -v
```

## Limitations
This is a portfolio demonstration using a historical, anonymized dataset. It is not a production payment system, and the decision threshold and cost assumptions are illustrative.
