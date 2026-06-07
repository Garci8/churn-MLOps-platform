import os
import json
import joblib
from sklearn.pipeline import Pipeline

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
