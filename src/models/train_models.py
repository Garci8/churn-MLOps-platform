import os
import json
from datetime import datetime
import pandas as pd
import joblib
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
from src.visualization.visualize import generate_shap_summary

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
processed_data_path = os.path.join(BASE_DIR, "data", "processed", "data.csv")

MODEL_VERSION = "0.1.0"

# Umbral de probabilidad para la clasificación
THRESHOLD = 0.5

# Columnas categóricas a las que se les aplicará One-Hot Encoding
categorical_cols = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
    "Contract", "PaymentMethod"
]

# Columnas numéricas que se escalarán (solo para Logistic Regression)
numerical_cols = [
    "tenure", "MonthlyCharges", "TotalCharges", 
    "AvgChargePerMonth", "ChargeGap", "NumServices"
]

# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Añade características calculadas al DataFrame de abandono de clientes.

    Calcula nuevas variables: AvgChargePerMonth, ChargeGap, NewCustomer y NumServices.

    Args:
        df (pd.DataFrame): DataFrame limpio de abandono de clientes.

    Returns:
        pd.DataFrame: DataFrame con las nuevas columnas añadidas.
    """
    # Crear variables adicionales
    # Variables económicas
    df['AvgChargePerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['ChargeGap'] = df['TotalCharges'] - df['MonthlyCharges'] * df['tenure'] # Diferencia entre el gasto real y lo esperado

    # Variables demográficas
    df["NewCustomer"] = (df["tenure"] < 12).astype(int)

    # Variables de servicio
    df["NumServices"] = (
        (df["InternetService"] != "No").astype(int) +
        (df["OnlineSecurity"] == "Yes").astype(int) +
        (df["OnlineBackup"] == "Yes").astype(int) +
        (df["DeviceProtection"] == "Yes").astype(int) +
        (df["TechSupport"] == "Yes").astype(int) +
        (df["StreamingTV"] == "Yes").astype(int) +
        (df["StreamingMovies"] == "Yes").astype(int)
    )
    
    return df

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
    
    # Separar variables predictoras y objetivo
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    
    # Split: 70% Train, 15% Validation, 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    
    # Calcular proporción de desbalanceo para XGBoost (negativos / positivos)
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / num_pos
    
    # Definir grillas de parámetros para GridSearchCV
    param_grids = {
        "logreg": {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],  # 0.0 es L2, 1.0 es L1
            "model__solver": ["saga"]       # saga soporta l1_ratio
        },
        "rf": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5, 10]
        },
        "xgboost": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [3, 5, 7]
        }
    }

    results = {}
    pipelines = {}

    # Entrenar y evaluar baseline (Dummy Classifier)
    dummy_model = DummyClassifier(strategy="stratified", random_state=42)
    dummy_model.fit(X_train, y_train)
    y_val_pred_dummy = dummy_model.predict(X_val)
    y_val_prob_dummy = dummy_model.predict_proba(X_val)[:, 1]
    
    results["dummy"] = {
        "accuracy": accuracy_score(y_val, y_val_pred_dummy),
        "roc_auc": roc_auc_score(y_val, y_val_prob_dummy),
        "f1": f1_score(y_val, y_val_pred_dummy)
    }

    # Entrenar y evaluar en conjunto de Validación
    for model_name in ["logreg", "rf", "xgboost"]:
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

    # Imprimir resultados de Validación
    print("\n=== Resultados en Validación ===")
    print("=" * 55)
    print(f"{'Modelo':<15} | {'Accuracy':<10} | {'ROC AUC':<10} | {'F1-Score':<10}")
    print("=" * 55)
    for model_name, metrics in results.items():
        print(f"{model_name:<15} | {metrics['accuracy']:<10.4f} | {metrics['roc_auc']:<10.4f} | {metrics['f1']:<10.4f}")
    print("=" * 55)

    # Seleccionar el mejor modelo basado en F1-Score en Validación (excluyendo el baseline dummy)
    candidate_results = {k: v for k, v in results.items() if k != "dummy"}
    best_model_name = max(candidate_results, key=lambda k: candidate_results[k]["f1"])
    best_pipeline = pipelines[best_model_name]
    
    print(f"\nMejor modelo seleccionado: {best_model_name}")

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

    # Guardar el pipeline del mejor modelo con joblib solo si supera al existente
    models_dir = os.path.join(BASE_DIR, "models")
    model_path = os.path.join(models_dir, "best_model.joblib")
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
                print("\nEl nuevo modelo no supera al anteriormente guardado. No se sobrescribe el archivo.")
            else:
                print("\n¡El nuevo modelo es mejor! Sobrescribiendo el archivo guardado...")
        except Exception as e:
            print(f"\nNo se pudo evaluar el modelo anteriormente guardado ({e}). Se guardará el nuevo por defecto.")
    
    if should_save:
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(best_pipeline, model_path)
        print(f"Pipeline del mejor modelo guardado con éxito en: {model_path}")

    # Extraer los mejores hiperparámetros encontrados para el modelo seleccionado
    model_step = best_pipeline.named_steps['model']
    best_params = {
        k.replace("model__", ""): model_step.get_params()[k.replace("model__", "")]
        for k in param_grids[best_model_name].keys()
    }

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

if __name__ == "__main__":
    main()