import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
raw_data_path = os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
processed_data_path = os.path.join(BASE_DIR, "data", "processed", "data.csv")

def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def save_dataset(df: pd.DataFrame, path: str)-> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    # Eliminar columnas no deseadas
    df = df.drop(columns=['customerID'])

    # Convertir TotalCharges to numérico, coaccionando errores a NaN, ya que hay filas con '' 
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')

    # Rellenar valores NaN con 0
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # Convertir variables binarias Yes/No a numéricas
    binary_cols = []

    for col in df.columns:
        unique_values = set(df[col].dropna().unique())

        if unique_values == {"Yes", "No"}:
            binary_cols.append(col)
    
    df[binary_cols] = df[binary_cols].replace({"Yes": 1, "No": 0})

    # Convertir variable binaria Gender (Male/Female) a numérica
    df['gender'] = df['gender'].replace({'Male': 1, 'Female': 0})

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

def main():
    df = load_dataset(raw_data_path)

    df = process_data(df)

    save_dataset(df, processed_data_path)


if __name__ == "__main__":
    main()


