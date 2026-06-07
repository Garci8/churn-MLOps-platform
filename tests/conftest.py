import pytest
import os
import joblib
import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

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

@pytest.fixture(scope="session", autouse=True)
def setup_dummy_model():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "best_model.joblib")
    info_path = os.path.join(models_dir, "best_model_info.json")
    
    # Solo los creamos si no existen (ej: entorno de GitHub Actions)
    if not os.path.exists(model_path) or not os.path.exists(info_path):
        # 1. Crear un pipeline dummy de scikit-learn
        dummy_model = DummyClassifier(strategy="constant", constant=0)
        dummy_model.fit(pd.DataFrame([[0]], columns=["dummy"]), [0])
        pipeline = Pipeline([("model", dummy_model)])
        
        joblib.dump(pipeline, model_path)
        
        # 2. Crear un JSON de metadatos dummy
        dummy_info = {
            "timestamp": "2026-06-07 00:00:00",
            "threshold": 0.5,
            "model_name": "dummy",
            "model_version": "0.0.0",
            "hyperparameters": {},
            "validation_metrics": {"accuracy": 1.0, "roc_auc": 1.0, "f1": 1.0},
            "test_metrics": {"accuracy": 1.0, "roc_auc": 1.0, "f1": 1.0},
            "saved_as_best": True
        }
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(dummy_info, f, indent=4)
