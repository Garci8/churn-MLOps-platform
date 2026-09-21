import os
import json
from datetime import datetime
import pandas as pd
import joblib
import mlflow
import yaml
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from src.data.make_dataset import load_dataset
from src.data.features import preprocess_data
from src.visualization.visualize import generate_shap_summary

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config_path = os.path.join(BASE_DIR, "config", "config.yaml")
config = load_config(config_path)

processed_data_path = os.path.join(BASE_DIR, config["data"]["processed_path"])
MODEL_VERSION = config["model"]["version"]
THRESHOLD = config["model"]["threshold"]
categorical_cols = config["model"]["categorical_cols"]
numerical_cols = config["model"]["numerical_cols"]
TRUSTED_TYPES = config["mlflow"]["skops_trusted_types"]

import socket

def get_tracking_uri() -> str:
    default_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if default_uri.startswith("http://") or default_uri.startswith("https://"):
        try:
            # Extraer host y puerto
            parts = default_uri.replace("http://", "").replace("https://", "").split(":")
            host = parts[0]
            port = int(parts[1].split("/")[0]) if len(parts) > 1 else 80
            with socket.create_connection((host, port), timeout=1.5):
                return default_uri
        except (socket.timeout, ConnectionRefusedError, OSError, ValueError):
            pass
    # Fallback local sqlite si el servidor HTTP no está accesible
    db_path = os.path.join(BASE_DIR, "mlflow", "mlflow.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return f"sqlite:///{db_path}"

mlflow.set_tracking_uri(get_tracking_uri())
mlflow.set_experiment(config["mlflow"]["experiment_name"])

# ==============================================================================
# CONSTRUCCIÓN DE PIPELINES DE MODELO
# ==============================================================================

def build_pipeline(model_name: str, scale_pos_weight: float = 1.0) -> Pipeline:
    """Crea un pipeline basado en el nombre del modelo.

    Configura los preprocesadores (ColumnTransformer) y los estimadores. Solo 
    'logreg' incluye el escalado estándar de las características numéricas continuas.

    Args:
        model_name (str): Identificador del modelo ('logreg', 'rf', 'xgboost').
        scale_pos_weight (float): Relación de desbalanceo (negativos / positivos) para XGBoost.

    Returns:
        Pipeline: Pipeline de scikit-learn que combina el preprocesamiento y el modelo.
    """
    if model_name == "logreg":
        preprocess = ColumnTransformer([
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first"), categorical_cols)
        ], remainder="passthrough")
        model = LogisticRegression(max_iter=3000, class_weight="balanced", tol=1e-5)

    elif model_name == "rf":
        preprocess = ColumnTransformer([
            ("num", "passthrough", numerical_cols),
            ("cat", OneHotEncoder(drop="first"), categorical_cols)
        ], remainder="passthrough")
        model = RandomForestClassifier(random_state=42, class_weight="balanced")
    
    elif model_name == "xgboost":
        preprocess = ColumnTransformer([
            ("num", "passthrough", numerical_cols),
            ("cat", OneHotEncoder(drop="first"), categorical_cols)
        ], remainder="passthrough")
        model = XGBClassifier(
            random_state=42, 
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight
        )

    return Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

# ==============================================================================
# EJECUCIÓN PRINCIPAL DEL SCRIPT
# ==============================================================================

def main() -> None:
    """Función de ejecución principal para el entrenamiento de modelos.

    Carga el dataset procesado, realiza ingeniería de características, divide 
    los datos en conjuntos de entrenamiento (70%), validación (15%) y prueba (15%), 
    entrena y evalúa los modelos (Regresión Logística, Random Forest, XGBoost), 
    selecciona el mejor modelo en validación, mide su rendimiento final en prueba 
    y actualiza el modelo almacenado si este nuevo modelo supera su rendimiento.
    """
    df = load_dataset(processed_data_path)
    df = preprocess_data(df)
    
    target_col = config["model"]["target_col"]
    models_to_train = config["model"]["models_to_train"]
    param_grids = config["param_grids"]
    
    # Separar variables predictoras y objetivo
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split
    test_size = config["data"]["split"]["test_size"]
    val_size = config["data"]["split"]["val_size"]
    random_state = config["data"]["split"]["random_state"]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp)
    
    # Calcular proporción de desbalanceo para XGBoost (negativos / positivos)
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / num_pos

    results = {}
    pipelines = {}

    # Entrenar y evaluar baseline (Dummy Classifier)
    dummy_model = DummyClassifier(strategy="stratified", random_state=random_state)
    dummy_model.fit(X_train, y_train)
    y_val_pred_dummy = dummy_model.predict(X_val)
    y_val_prob_dummy = dummy_model.predict_proba(X_val)[:, 1]
    
    results["dummy"] = {
        "accuracy": accuracy_score(y_val, y_val_pred_dummy),
        "roc_auc": roc_auc_score(y_val, y_val_prob_dummy),
        "f1": f1_score(y_val, y_val_pred_dummy)
    }

    # Entrenar y evaluar en conjunto de Validación
    with mlflow.start_run(run_name="parent_run"):

        for model_name in models_to_train:
            with mlflow.start_run(run_name=model_name, nested=True):
                pipeline = build_pipeline(model_name, scale_pos_weight=scale_pos_weight)
                
                # GridSearchCV para encontrar los mejores hiperparámetros
                grid = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grids[model_name],
                    cv=5,
                    scoring="f1",
                    n_jobs=-1
                )
                grid.fit(X_train, y_train)
                
                best_pipeline = grid.best_estimator_
                
                # Evaluar el mejor pipeline en Validación
                y_val_prob = best_pipeline.predict_proba(X_val)[:, 1]
                y_val_pred = (y_val_prob >= THRESHOLD).astype(int)
                
                results[model_name] = {
                    "accuracy": accuracy_score(y_val, y_val_pred),
                    "roc_auc": roc_auc_score(y_val, y_val_prob),
                    "f1": f1_score(y_val, y_val_pred)
                }
                pipelines[model_name] = best_pipeline

                mlflow.log_params(grid.best_params_)
                mlflow.log_metrics(results[model_name])
                mlflow.sklearn.log_model(
                    best_pipeline, 
                    name=f"model_{model_name}", 
                    skops_trusted_types=TRUSTED_TYPES
                )

        # Imprimir resultados de Validación
        print("\n=== Resultados en Validación ===")
        print("=" * 55)
        print(f"{'Modelo':<15} | {'Accuracy':<10} | {'ROC AUC':<10} | {'F1-Score':<10}")
        print("=" * 55)
        for model_name, metrics in results.items():
            print(f"{model_name:<15} | {metrics['accuracy']:<10.4f} | {metrics['roc_auc']:<10.4f} | {metrics['f1']:<10.4f}")
        print("=" * 55)

        # Seleccionar el mejor modelo (Champion) y el segundo mejor (Candidate/Challenger) basándose en F1-Score
        candidate_results = {k: v for k, v in results.items() if k != "dummy"}
        sorted_candidates = sorted(candidate_results.keys(), key=lambda k: candidate_results[k]["f1"], reverse=True)
        best_model_name = sorted_candidates[0]
        second_model_name = sorted_candidates[1] if len(sorted_candidates) > 1 else sorted_candidates[0]

        mlflow.log_param("best_model_name", best_model_name)
        mlflow.log_param("candidate_model_name", second_model_name)

        registered_model_name = config["mlflow"]["registered_model_name"]
        best_pipeline = pipelines[best_model_name]
        second_pipeline = pipelines[second_model_name]

        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            name="best_model",
            registered_model_name=registered_model_name,
            skops_trusted_types=TRUSTED_TYPES
        )
        
        print(f"\nMejor modelo seleccionado (Champion): {best_model_name}")
        print(f"Segundo mejor modelo seleccionado (Candidate/Challenger): {second_model_name}")

        # Evaluar el mejor modelo en el conjunto de Test
        y_test_prob = best_pipeline.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_prob >= THRESHOLD).astype(int)
        
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_roc_auc = roc_auc_score(y_test, y_test_prob)
        test_f1 = f1_score(y_test, y_test_pred)

        print("\n=== Evaluación Final en Test (Modelo Seleccionado) ===")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"ROC AUC:  {test_roc_auc:.4f}")
        print(f"F1-Score: {test_f1:.4f}")
        mlflow.log_metrics({
            "test_accuracy": test_accuracy,
            "test_roc_auc": test_roc_auc,
            "test_f1": test_f1
        })

        # Guardar el pipeline del mejor modelo con joblib solo si supera al existente
        models_dir = os.path.join(BASE_DIR, "models")
        model_path = os.path.join(models_dir, "best_model.joblib")
        candidate_model_path = os.path.join(models_dir, "candidate_model.joblib")
        should_save = True

        if os.path.exists(model_path):
            try:
                previous_pipeline = joblib.load(model_path)
                y_val_prob_prev = previous_pipeline.predict_proba(X_val)[:, 1]
                y_val_pred_prev = (y_val_prob_prev >= THRESHOLD).astype(int)
                previous_f1 = f1_score(y_val, y_val_pred_prev)
                current_f1 = results[best_model_name]["f1"]

                print(f"\nF1-Score del modelo guardado anteriormente: {previous_f1:.4f}")
                print(f"F1-Score del nuevo mejor modelo ({best_model_name}): {current_f1:.4f}")

                if current_f1 <= previous_f1:
                    should_save = False
                    print("\n[ROLLBACK] El nuevo modelo no supera al anteriormente guardado. Se mantiene el modelo activo anterior.")
                else:
                    print("\n¡El nuevo modelo es mejor! Sobrescribiendo el archivo guardado...")
            except Exception as e:
                print(f"\nNo se pudo evaluar el modelo anteriormente guardado ({e}). Se guardará el nuevo por defecto.")
        
        if should_save:
            os.makedirs(models_dir, exist_ok=True)
            joblib.dump(best_pipeline, model_path)
            joblib.dump(second_pipeline, candidate_model_path)
            print(f"Pipeline del mejor modelo guardado en: {model_path}")
            print(f"Pipeline del modelo candidato guardado en: {candidate_model_path}")

            # Guardar metadatos del candidato
            candidate_metrics_path = os.path.join(models_dir, "candidate_model_info.json")
            candidate_info = {
                "model_name": second_model_name,
                "model_version": MODEL_VERSION,
                "threshold": THRESHOLD,
                "val_accuracy": results[second_model_name]["accuracy"],
                "val_roc_auc": results[second_model_name]["roc_auc"],
                "val_f1": results[second_model_name]["f1"],
                "saved_at": datetime.now().isoformat()
            }
            with open(candidate_metrics_path, "w", encoding="utf-8") as f:
                json.dump(candidate_info, f, indent=4, ensure_ascii=False)

        # Extraer los mejores hiperparámetros encontrados para el modelo seleccionado
        model_step = best_pipeline.named_steps['model']
        best_params = {
            k.replace("model__", ""): model_step.get_params()[k.replace("model__", "")]
            for k in param_grids[best_model_name].keys()
        }
        mlflow.log_params(best_params)

        # Registrar las métricas del modelo actual
        current_run_metrics = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "threshold": THRESHOLD,
            "model_name": best_model_name,
            "model_version": MODEL_VERSION,
            "hyperparameters": best_params,
            "validation_metrics": {
                "accuracy": results[best_model_name]["accuracy"],
                "roc_auc": results[best_model_name]["roc_auc"],
                "f1": results[best_model_name]["f1"]
            },
            "test_metrics": {
                "accuracy": test_accuracy,
                "roc_auc": test_roc_auc,
                "f1": test_f1
            },
            "saved_as_best": should_save
        }

        # 1. Guardar/actualizar el histórico de experimentos
        history_path = os.path.join(models_dir, "metrics_history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
                    if not isinstance(history, list):
                        history = []
            except Exception:
                history = []
        else:
            history = []
        
        history.append(current_run_metrics)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
        print(f"Historial de métricas actualizado en: {history_path}")

        # 2. Guardar métricas del mejor modelo activo si ha sido guardado
        if should_save:
            active_metrics_path = os.path.join(models_dir, "best_model_info.json")
            with open(active_metrics_path, "w", encoding="utf-8") as f:
                json.dump(current_run_metrics, f, indent=4, ensure_ascii=False)
            print(f"Métricas del mejor modelo activo guardadas en: {active_metrics_path}")

        # Generar y guardar gráfico SHAP para el mejor modelo
        generate_shap_summary(best_pipeline, best_model_name, X_val, models_dir)
        shap_path = os.path.join(models_dir, "shap_summary.png") 
        if os.path.exists(shap_path):
            mlflow.log_artifact(shap_path)

if __name__ == "__main__":
    main()