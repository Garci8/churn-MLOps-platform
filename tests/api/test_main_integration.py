from fastapi.testclient import TestClient
from src.api.main import app
import pytest

@pytest.fixture
def valid_payload():
    # Payload con datos controlados y correctos
    return {
        "gender": 1,
        "SeniorCitizen": 0,
        "Partner": 0,
        "Dependents": 0,
        "tenure": 1,
        "PhoneService": 1,
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": 1,
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.05,
        "TotalCharges": 70.05,
        "AvgChargePerMonth": 70.05,
        "ChargeGap": 0.0,
        "NewCustomer": 1,
        "NumServices": 1
    }

def test_lifespan_loads_model_on_startup():
    # Al levantar el TestClient con 'with', se dispara el lifespan
    with TestClient(app) as client:
        # Verificamos que los objetos se han inyectado en la instancia de la app
        assert hasattr(app, "model")
        assert app.model is not None
        
        assert hasattr(app, "json_file")
        assert app.json_file is not None


def test_health_endpoint():
    # 'with' arranca y detiene los eventos lifespan de la app automáticamente
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

def test_valid_predict_v1_endpoint(valid_payload): 
    with TestClient(app) as client:
        response = client.post("/v1/predict", json=valid_payload)
        assert response.status_code == 200
        assert "churn" in response.json()
        assert isinstance(response.json()["churn"], bool)
    
def test_invalid_predict_v1_endpoint(valid_payload): 
    invalid_payload = valid_payload.copy()
    del invalid_payload["gender"]
    with TestClient(app) as client:
        response = client.post("/v1/predict", json=invalid_payload)
        assert response.status_code == 422
    invalid_payload = valid_payload.copy()
    mapping = {1:"Yes",0:"No"}
    invalid_payload["Partner"] = mapping[invalid_payload["Partner"]]
    with TestClient(app) as client:
        response = client.post("/v1/predict", json=invalid_payload)
        assert response.status_code == 422

def test_valid_predict_v2_endpoint(valid_payload): 
    with TestClient(app) as client:
        response = client.post("/v2/predict", json=valid_payload)
        assert response.status_code == 200
        assert "churn" in response.json()
        assert "probabilidad_churn" in response.json()
        assert isinstance(response.json()["churn"], bool) and isinstance(response.json()["probabilidad_churn"], float)
    
def test_invalid_predict_v2_endpoint(valid_payload): 
    invalid_payload = valid_payload.copy()
    del invalid_payload["gender"]
    with TestClient(app) as client:
        response = client.post("/v2/predict", json=invalid_payload)
        assert response.status_code == 422
    invalid_payload = valid_payload.copy()
    mapping = {1:"Yes",0:"No"}
    invalid_payload["Partner"] = mapping[invalid_payload["Partner"]]
    with TestClient(app) as client:
        response = client.post("/v2/predict", json=invalid_payload)
        assert response.status_code == 422

def test_model_info_endpoint():
    with TestClient(app) as client:
        response = client.get("/model/info")
        assert response.status_code == 200
        
        info = response.json()
        # Verificar existencia de claves requeridas en la estructura
        required_keys = [
            "timestamp", "threshold", "model_name", "model_version", 
            "hyperparameters", "validation_metrics", "test_metrics"
        ]
        for key in required_keys:
            assert key in info
            
        # Verificar tipos de datos esperados
        assert isinstance(info["threshold"], float)
        assert isinstance(info["model_name"], str)
        assert isinstance(info["model_version"], str)
        assert isinstance(info["validation_metrics"], dict)
        assert isinstance(info["test_metrics"], dict)
    