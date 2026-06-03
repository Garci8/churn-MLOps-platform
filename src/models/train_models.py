import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from xgboost import XGBClassifier
from src.data.make_dataset import load_dataset

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
processed_data_path = os.path.join(BASE_DIR, "data", "processed", "data.csv")

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

def build_pipeline(model_name: str) -> Pipeline:
    """Crea un pipeline basado en el nombre del modelo.

    Configura los preprocesadores (ColumnTransformer) y los estimadores. Solo 
    'logreg' incluye el escalado estándar de las características numéricas continuas.

    Args:
        model_name (str): Identificador del modelo ('logreg', 'rf', 'xgboost').

    Returns:
        Pipeline: Pipeline de scikit-learn que combina el preprocesamiento y el modelo.
    """
    if model_name == "logreg":
        preprocess = ColumnTransformer([
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first"), categorical_cols)
        ], remainder="passthrough")
        model = LogisticRegression(max_iter=1000)

    elif model_name == "rf":
        preprocess = ColumnTransformer([
            ("num", "passthrough", numerical_cols),
            ("cat", OneHotEncoder(drop="first"), categorical_cols)
        ], remainder="passthrough")
        model = RandomForestClassifier(random_state=42)
    
    elif model_name == "xgboost":
        preprocess = ColumnTransformer([
            ("num", "passthrough", numerical_cols),
            ("cat", OneHotEncoder(drop="first"), categorical_cols)
        ], remainder="passthrough")
        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")

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
    
    results = {}
    pipelines = {}

    # Entrenar y evaluar en conjunto de Validación
    for model_name in ["logreg", "rf", "xgboost"]:
        pipeline = build_pipeline(model_name)
        pipeline.fit(X_train, y_train)
        
        # Evaluar en Validación
        y_val_pred = pipeline.predict(X_val)
        y_val_prob = pipeline.predict_proba(X_val)[:, 1]
        
        results[model_name] = {
            "accuracy": accuracy_score(y_val, y_val_pred),
            "roc_auc": roc_auc_score(y_val, y_val_prob),
            "f1": f1_score(y_val, y_val_pred)
        }
        pipelines[model_name] = pipeline

    # Imprimir resultados de Validación
    print("\n=== Resultados en Validación ===")
    print("=" * 55)
    print(f"{'Modelo':<15} | {'Accuracy':<10} | {'ROC AUC':<10} | {'F1-Score':<10}")
    print("=" * 55)
    for model_name, metrics in results.items():
        print(f"{model_name:<15} | {metrics['accuracy']:<10.4f} | {metrics['roc_auc']:<10.4f} | {metrics['f1']:<10.4f}")
    print("=" * 55)

    # Seleccionar el mejor modelo basado en F1-Score en Validación
    best_model_name = max(results, key=lambda k: results[k]["f1"])
    best_pipeline = pipelines[best_model_name]
    
    print(f"\nMejor modelo seleccionado: {best_model_name}")

    # Evaluar el mejor modelo en el conjunto de Test
    y_test_pred = best_pipeline.predict(X_test)
    y_test_prob = best_pipeline.predict_proba(X_test)[:, 1]
    
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
            y_val_pred_prev = previous_pipeline.predict(X_val)
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

if __name__ == "__main__":
    main()