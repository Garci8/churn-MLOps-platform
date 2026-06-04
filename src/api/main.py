from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import json
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
MODEL_INFO_PATH = os.path.join(BASE_DIR, "models", "best_model_info.json")

json_file = json.load(open(MODEL_INFO_PATH, "r"))


# Cargar modelo
with open(MODEL_PATH, "rb") as f:
    model = joblib.load(f)

app = FastAPI()

# Pydantic define la estructura de entrada
class ClienteData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: int
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    AvgChargePerMonth: float
    ChargeGap: float
    NewCustomer: int
    NumServices: int

# Endpoint de predicción
@app.post("/predict")
def predict(data: ClienteData):
    df = pd.DataFrame([data.dict()])
    prob = model.predict_proba(df)[0][1]
    churn = bool(prob >= json_file["threshold"])
    return {
        "churn": churn,
        "probabilidad": round(prob, 4)
    }

# Endpoint de salud
@app.get("/health")
def health():
    return {"status": "ok"}

# Endpoint de metadatos del modelo en producción
@app.get("/model/info")
def model_info():
    return json_file