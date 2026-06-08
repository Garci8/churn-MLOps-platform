import os
import json
import joblib
from sklearn.pipeline import Pipeline
import pandas as pd
import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
MODEL_INFO_PATH = os.path.join(BASE_DIR, "models", "best_model_info.json")

def test_model_and_info_loading():
    # 1. Verificar que los archivos físicos existen en la ruta
    assert os.path.exists(MODEL_PATH)
    assert os.path.exists(MODEL_INFO_PATH)

    # 2. Verificar que el modelo se carga y es un Pipeline de scikit-learn con métodos de predicción
    model = joblib.load(MODEL_PATH)
    assert isinstance(model, Pipeline)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

    # 3. Verificar que los metadatos JSON se cargan y contienen las propiedades clave
    with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
        info = json.load(f)
        
    assert "threshold" in info
    assert "model_name" in info
    assert "model_version" in info
    assert isinstance(info["threshold"], float)

def test_model_performance():
    # Verificar que las métricas de rendimiento registradas superan los umbrales de calidad
    with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
        info = json.load(f)
    assert info["test_metrics"]["roc_auc"] >= 0.7
    assert info["test_metrics"]["f1"] >= 0.6

def test_model_is_deterministic(valid_payload):
    # Test de invarianza: misma entrada debe retornar la misma probabilidad exacta
    model = joblib.load(MODEL_PATH)
    valid_df = pd.DataFrame([valid_payload])
    det_prob_1 = model.predict_proba(valid_df)[:, 1]
    det_prob_2 = model.predict_proba(valid_df)[:, 1]
    det_prob_3 = model.predict_proba(valid_df)[:, 1]
    assert det_prob_1 == pytest.approx(det_prob_2) 
    assert det_prob_2 == pytest.approx(det_prob_3)

def test_model_directionality(valid_payload):
    # Test de direccionalidad: verificar lógica de negocio de variables clave
    higher_tenure_payload = valid_payload.copy()
    higher_tenure_payload["tenure"] = 72
    model = joblib.load(MODEL_PATH)

    og_valid_df = pd.DataFrame([valid_payload])
    higher_tenure_df = pd.DataFrame([higher_tenure_payload])

    det_prob_og = model.predict_proba(og_valid_df)[:, 1]
    det_prob_tenure = model.predict_proba(higher_tenure_df)[:, 1]

    # A mayor antigüedad (tenure), menor o igual probabilidad de abandono (churn)
    assert det_prob_tenure<=det_prob_og

    higher_monthly_charge_payload = valid_payload.copy()
    higher_monthly_charge_payload["MonthlyCharges"] = 300

    higher_monthly_charge_df = pd.DataFrame([higher_monthly_charge_payload])

    det_prob_higher_monthly_charge = model.predict_proba(higher_monthly_charge_df)[:, 1]

    # A mayor coste mensual (MonthlyCharges), mayor o igual probabilidad de abandono (churn)
    assert det_prob_og<=det_prob_higher_monthly_charge



