from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import RedirectResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from contextlib import asynccontextmanager
import yaml

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
MODEL_INFO_PATH = os.path.join(BASE_DIR, "models", "best_model_info.json")

config_path = os.path.join(BASE_DIR, "config", "config.yaml")

with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

# Cargar modelo
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Al iniciar
    with open(MODEL_PATH, "rb") as f:
        app.model = joblib.load(f)
    with open(MODEL_INFO_PATH, "rb") as f:
        app.json_file = json.load(f)
    yield
    # Al cerrar
    app.model = None
    app.json_file = None

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, (StarletteHTTPException, RequestValidationError)):
        raise exc
        
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor"}
    )

# Pydantic define la estructura de entrada
# Pydantic define la estructura de entrada
class ClientData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    AvgChargePerMonth: float
    ChargeGap: float
    NewCustomer: int
    NumServices: int

# Ejemplos para la documentación interactiva
predict_examples = {
    "cliente_fidelizado": {
        "summary": "Cliente fidelizado",
        "description": "Cliente con contrato a largo plazo y servicios contratados, bajo riesgo.",
        "value": {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 72,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 85.0,
            "TotalCharges": 6120.0,
            "AvgChargePerMonth": 85.0,
            "ChargeGap": 0.0,
            "NewCustomer": 0,
            "NumServices": 6
        }
    },
    "cliente_riesgo": {
        "summary": "Cliente de riesgo",
        "description": "Cliente nuevo, sin servicios de permanencia y contrato mes a mes.",
        "value": {
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.05,
            "TotalCharges": 70.05,
            "AvgChargePerMonth": 70.05,
            "ChargeGap": 0.0,
            "NewCustomer": 1,
            "NumServices": 1
        }
    }
}

def log_prediction(input_data: dict, prediction_result: dict, endpoint_version: str):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": endpoint_version,
        "model_name": app.json_file.get("model_name"),
        "model_version": app.json_file.get("model_version"),
        "inputs": input_data,
        "prediction": prediction_result
    }
    
    log_path = os.path.join(BASE_DIR, config["data"]["predictions_log_path"])
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

# Endpoint de predicción V1 (solo Churn: true/false)
@app.post("/v1/predict")
def predict_v1(data: ClientData = Body(openapi_examples=predict_examples)):
    try:
        df = pd.DataFrame([data.model_dump()])
        prob = app.model.predict_proba(df)[0][1]
        churn = bool(prob >= app.json_file["threshold"])
        result = {"churn": churn}
        log_prediction(input_data=data.model_dump(),prediction_result=result,endpoint_version="v1")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir en V1: {str(e)}")

# Endpoint de predicción V2 (Churn y probabilidad)
@app.post("/v2/predict")
def predict_v2(data: ClientData = Body(openapi_examples=predict_examples)):
    try:
        df = pd.DataFrame([data.model_dump()])
        prob = app.model.predict_proba(df)[0][1]
        churn = bool(prob >= app.json_file["threshold"])
        result = {"churn": churn, "probabilidad_churn": round(prob, 4)}
        log_prediction(input_data=data.model_dump(),prediction_result=result,endpoint_version="v2")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir en V2: {str(e)}")

# Endpoint de salud
@app.get("/health")
def health():
    return {"status": "ok"}

# Endpoint de metadatos del modelo en producción
@app.get("/model/info")
def model_info():
    return app.json_file