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
        # 1. Crear un DataFrame dummy con las 23 columnas esperadas por la API
        dummy_data = {
            "gender": [1,0], "SeniorCitizen": [0,1], "Partner": [0,1], "Dependents": [0,1],
            "tenure": [1,20], "PhoneService": [1,0], "MultipleLines": ["No","Yes"], "InternetService": ["No","Yes"],
            "OnlineSecurity": ["No","Yes"], "OnlineBackup": ["No","Yes"], "DeviceProtection": ["No","Yes"],
            "TechSupport": ["No","Yes"], "StreamingTV": ["No","Yes"], "StreamingMovies": ["No","Yes"],
            "Contract": ["Month-to-month","Month-to-month"], "PaperlessBilling": [1,0], "PaymentMethod": ["Electronic check","Electronic check"],
            "MonthlyCharges": [0.0,0.0], "TotalCharges": [0.0,0.0], "AvgChargePerMonth": [0.0,0.0],
            "ChargeGap": [0.0,0.0], "NewCustomer": [1,0], "NumServices": [1,2]
        }
        df_dummy = pd.DataFrame(dummy_data)

        # 2. Entrenar el DummyClassifier con la estructura correcta
        dummy_model = DummyClassifier(strategy="constant", constant=0)
        dummy_model.fit(df_dummy, [0,1])
        pipeline = Pipeline([("model", dummy_model)])
        
        joblib.dump(pipeline, model_path)

        # 3. Crear un JSON de metadatos dummy
        dummy_info = {
            "timestamp": "2026-06-05 10:32:41",
            "threshold": 0.5,
            "model_name": "rf",
            "model_version": "0.1.0",
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2
            },
            "validation_metrics": {
                "accuracy": 0.7746212121212122,
                "roc_auc": 0.8444081369661267,
                "f1": 0.6404833836858006
            },
            "test_metrics": {
                "accuracy": 0.7861873226111636,
                "roc_auc": 0.8345883626224457,
                "f1": 0.6366559485530546
            },
            "saved_as_best": True
        }
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(dummy_info, f, indent=4)
