import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
raw_data_path = os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
processed_data_path = os.path.join(BASE_DIR, "data", "processed", "data.csv")

# ==============================================================================
# CARGA DE DATOS
# ==============================================================================

def load_dataset(path: str) -> pd.DataFrame:
    """Carga un dataset en formato CSV desde la ruta especificada.

    Args:
        path (str): Ruta del archivo CSV a cargar.

    Returns:
        pd.DataFrame: DataFrame con los datos cargados.
    """
    return pd.read_csv(path)

# ==============================================================================
# GUARDADO DE DATOS
# ==============================================================================

def save_dataset(df: pd.DataFrame, path: str) -> None:
    """Guarda un DataFrame en un archivo CSV.

    Crea los directorios padre si no existen previamente.

    Args:
        df (pd.DataFrame): DataFrame a guardar.
        path (str): Ruta de destino del archivo CSV.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

# ==============================================================================
# LIMPIEZA DE DATOS
# ==============================================================================

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica operaciones de limpieza al DataFrame de abandono de clientes.

    Limpia los datos eliminando la columna 'customerID', convirtiendo 
    'TotalCharges' a valores numéricos, rellenando valores nulos con 0 y 
    codificando variables binarias en formato numérico (1/0).

    Args:
        df (pd.DataFrame): DataFrame original sin procesar.

    Returns:
        pd.DataFrame: DataFrame limpio y codificado parcialmente.
    """
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

    return df

# ==============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================

def main() -> None:
    """Función principal para ejecutar la limpieza de datos."""
    df = load_dataset(raw_data_path)
    df = process_data(df)
    save_dataset(df, processed_data_path)


if __name__ == "__main__":
    main()
