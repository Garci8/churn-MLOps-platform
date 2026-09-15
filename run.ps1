param (
    [string]$Target = "help"
)

$BASE_DIR = $PSScriptRoot
Set-Location $BASE_DIR

$PYTHON = "$BASE_DIR\venv\Scripts\python.exe"
$PYTEST = "$BASE_DIR\venv\Scripts\pytest.exe"
$env:PYTHONPATH = $BASE_DIR

function Run-Data {
    Write-Host "==> Ejecutando procesamiento de datos..." -ForegroundColor Cyan
    & $PYTHON src/data/make_dataset.py
}

function Run-Train {
    Write-Host "==> Ejecutando entrenamiento de modelos..." -ForegroundColor Cyan
    & $PYTHON src/models/train_models.py
}

function Run-Test {
    Write-Host "==> Ejecutando pruebas unitarias con pytest..." -ForegroundColor Cyan
    & $PYTEST tests/
}

switch ($Target.ToLower()) {
    "data"     { Run-Data }
    "train"    { Run-Train }
    "test"     { Run-Test }
    "pipeline" { Run-Data; Run-Train; Run-Test }
    default {
        Write-Host "Comandos disponibles:" -ForegroundColor Yellow
        Write-Host "  .\run.ps1 data     - Ejecuta la limpieza y procesamiento de datos"
        Write-Host "  .\run.ps1 train    - Entrena los modelos y los registra en MLflow"
        Write-Host "  .\run.ps1 test     - Ejecuta la suite de pruebas unitarias e integración"
        Write-Host "  .\run.ps1 pipeline - Ejecuta todo el pipeline (data -> train -> test)"
    }
}
