
from fastapi.testclient import TestClient

from api.fraud_detection_api import app


client = TestClient(app)


def sample_features():
    return {
        "Time": 0.0,
        "V1": -1.3598071336738,
        "V2": -0.0727811733098497,
        "V3": 2.53634673796914,
        "V4": 1.37815522427443,
        "V5": -0.338320769942518,
        "V6": 0.462387777762292,
        "V7": 0.239598554061257,
        "V8": 0.0986979012610507,
        "V9": 0.363786969611213,
        "V10": 0.0907941719789316,
        "V11": -0.551599533260813,
        "V12": -0.617800855762348,
        "V13": -0.991389847235408,
        "V14": -0.311169353699879,
        "V15": 1.46817697209427,
        "V16": -0.470400525259478,
        "V17": 0.207971241929242,
        "V18": 0.0257905801985591,
        "V19": 0.403992960255733,
        "V20": 0.251412098239705,
        "V21": -0.018306777944153,
        "V22": 0.277837575558899,
        "V23": -0.110473910188767,
        "V24": 0.0669280749146731,
        "V25": 0.128539358273528,
        "V26": -0.189114843888824,
        "V27": 0.133558376740387,
        "V28": -0.0210530534538215,
        "Amount": 149.62,
    }


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["threshold"] == 0.1


def test_valid_prediction():
    response = client.post(
        "/predict",
        json={"features": sample_features()},
    )
    assert response.status_code == 200

    result = response.json()
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["fraud_prediction"] in [0, 1]
    assert result["risk_classification"] in ["Fraud", "Legitimate"]


def test_missing_feature_rejected():
    features = sample_features()
    features.pop("V28")

    response = client.post(
        "/predict",
        json={"features": features},
    )
    assert response.status_code == 422
    assert "V28" in response.json()["detail"]["missing_features"]


def test_negative_amount_rejected():
    features = sample_features()
    features["Amount"] = -10.0

    response = client.post(
        "/predict",
        json={"features": features},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Time and Amount must be non-negative."
    )
