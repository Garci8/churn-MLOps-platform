import pytest
from pydantic import ValidationError
from src.api.main import ClientData

@pytest.fixture
def valid_payload():
    # Payload con datos controlados y correctos
    return {
        "gender": 1,
        "SeniorCitizen": 0,
        "Partner": 0,
        "Dependents": 0,
        "tenure": 1,
        "PhoneService": 1,
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": 1,
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.05,
        "TotalCharges": 70.05,
        "AvgChargePerMonth": 70.05,
        "ChargeGap": 0.0,
        "NewCustomer": 1,
        "NumServices": 1
    }

def test_client_data_valid(valid_payload):
    # Validar que se crea el objeto correctamente
    client = ClientData(**valid_payload)
    assert client.tenure == 1
    assert client.MonthlyCharges == 70.05

def test_client_data_missing_required_field(valid_payload):
    # Eliminar un campo obligatorio y verificar que lanza ValidationError
    invalid_payload = valid_payload.copy()
    del invalid_payload["tenure"]
    
    with pytest.raises(ValidationError):
        ClientData(**invalid_payload)

def test_client_data_invalid_type(valid_payload):
    # Cambiar un tipo numérico por texto no convertible y verificar que falla
    invalid_payload = valid_payload.copy()
    invalid_payload["MonthlyCharges"] = "no_es_un_numero"
    
    with pytest.raises(ValidationError):
        ClientData(**invalid_payload)
