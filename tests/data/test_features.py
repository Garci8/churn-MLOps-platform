import pytest
import pandas as pd
from src.data.features import preprocess_data


def test_preprocess_data_calculation():
    # 1. Crear un DataFrame de entrada controlado
    input_df = pd.DataFrame({
        'tenure': [0, 12, 5],
        'MonthlyCharges': [50.0, 100.0, 35.0],
        'TotalCharges': [0.0, 1200.0, 150.0],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['Yes', 'No', 'No'],
        'OnlineBackup': ['No', 'Yes', 'No'],
        'DeviceProtection': ['No', 'No', 'No'],
        'TechSupport': ['No', 'No', 'No'],
        'StreamingTV': ['No', 'No', 'No'],
        'StreamingMovies': ['No', 'No', 'No']
    })

    # 2. Ejecutar la función
    result_df = preprocess_data(input_df)

    # 3. Validar AvgChargePerMonth (TotalCharges / (tenure + 1))
    # Fila 0: 0 / 1 = 0.0
    # Fila 2: 150 / 6 = 25.0
    assert result_df['AvgChargePerMonth'].iloc[0] == 0.0
    assert result_df['AvgChargePerMonth'].iloc[1] ==pytest.approx(1200.0/13.0)
    assert result_df['AvgChargePerMonth'].iloc[2] ==pytest.approx(150.0/6.0)

    # 4. Validar ChargeGap (TotalCharges - MonthlyCharges * tenure)
    # Fila 0: 0 - 50*0 = 0.0
    # Fila 1: 1200 - 100*12 = 0.0
    # Fila 2: 150 - 35*5 = -25.0
    assert list(result_df['ChargeGap']) == ([0.0, 0.0, -25.0])

    # 5. Validar NewCustomer (tenure < 12)
    # Fila 0 (tenure 0) -> 1
    # Fila 1 (tenure 12) -> 0
    # Fila 2 (tenure 5) -> 1
    assert list(result_df['NewCustomer']) == [1, 0, 1]

    # 6. Validar NumServices
    # Fila 0: Internet (1) + OnlineSecurity (1) = 2
    # Fila 1: Internet (1) + OnlineBackup (1) = 2
    # Fila 2: Internet 'No' (0) + todos 'No' (0) = 0
    assert list(result_df['NumServices']) == [2, 2, 0]