import os
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
models_dir = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(models_dir, "best_model.joblib")

with open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)
print(model)