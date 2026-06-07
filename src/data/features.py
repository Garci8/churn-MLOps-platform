import pandas as pd

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
