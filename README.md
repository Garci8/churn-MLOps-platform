# Churn MLOps Platform

Plataforma completa de predicción de abandono de clientes (churn) para una empresa de telecomunicaciones, construida con prácticas MLOps: pipeline reproducible, seguimiento de experimentos, API de predicción con pruebas A/B, monitorización de drift y reentrenamiento automatizado con rollback de seguridad.

---

## Tabla de contenidos

- [Arquitectura](#arquitectura)
- [Tecnologías](#tecnologías)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Requisitos previos](#requisitos-previos)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Uso](#uso)
  - [Pipeline completo](#pipeline-completo)
  - [Comandos individuales](#comandos-individuales)
  - [API de predicción](#api-de-predicción)
  - [Docker Compose](#docker-compose)
- [Componentes](#componentes)
  - [Pipeline de datos](#1-pipeline-de-datos)
  - [Pipeline de entrenamiento](#2-pipeline-de-entrenamiento)
  - [API de servicio](#3-api-de-servicio)
  - [Monitorización y drift](#4-monitorización-y-drift)
  - [CI/CD](#5-cicd)
- [Pruebas A/B](#pruebas-ab)
- [Rollback automático](#rollback-automático)
- [Tests](#tests)
- [Conjunto de datos](#conjunto-de-datos)

---

## Arquitectura

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Datos crudos│────▶│  Pipeline de     │────▶│  Datos procesados│
│  (CSV)       │     │  datos (Limpieza │     │  (data.csv)      │
└──────────────┘     │ + Características)│    └────────┬─────────┘
                     └──────────────────┘              │
                                                       ▼
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  MLflow      │◀────│  Pipeline de     │◀────│  Train / Val /   │
│ (Seguimiento)│     │  entrenamiento   │     │  Test            │
└──────────────┘     │  (GridSearchCV)  │     └──────────────────┘
                     └───────┬──────────┘
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
        ┌────────────────┐      ┌────────────────┐
        │  Champion      │      │  Candidate     │
        │  (mejor modelo)│      │  (2º mejor)    │
        └───────┬────────┘      └───────┬────────┘
                │                       │
                └───────────┬───────────┘
                            ▼
                   ┌─────────────────┐
                   │  FastAPI         │
                   │  /v1/predict     │──── Pruebas A/B (80/20)
                   │  /v2/predict     │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐     ┌──────────────────┐
                   │  Registro de    │────▶│  Evidently Drift │
                   │  predicciones   │     │  Monitor         │
                   └─────────────────┘     └────────┬─────────┘
                                                    │
                                           Drift detectado?
                                             │           │
                                            Sí          No
                                             │           │
                                             ▼           ▼
                                      Auto-retrain    OK ✓
                                      (con rollback)
```

---

## Tecnologías

| Categoría | Tecnología |
|---|---|
| **Lenguaje** | Python 3.12 |
| **Modelos** | scikit-learn, XGBoost |
| **Optimización** | GridSearchCV (F1-Score, 5-fold CV) |
| **Explicabilidad** | SHAP |
| **API** | FastAPI + Uvicorn |
| **Seguimiento de experimentos** | MLflow (servidor HTTP / respaldo SQLite) |
| **Monitorización** | Evidently AI (Data & Prediction Drift) |
| **Contenedores** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions |
| **Tests** | pytest |
| **Linting** | flake8 |

---

## Estructura del proyecto

```
churn_MLOps/
├── .github/workflows/
│   ├── ci.yml                      # CI: lint → pipeline → test → build Docker
│   └── scheduled_retrain.yml       # Cron mensual + workflow_dispatch
├── config/
│   └── config.yaml                 # Configuración centralizada
├── data/
│   ├── raw/                        # Conjunto de datos original (CSV)
│   └── processed/                  # Datos limpios generados por el pipeline
├── docker/
│   └── docker-compose.yml          # MLflow server + API
├── logs/
│   └── predictions.jsonl           # Registro de predicciones en producción
├── mlflow/                         # Artefactos y BD de MLflow
├── models/
│   ├── best_model.joblib           # Modelo Champion serializado
│   ├── best_model_info.json        # Metadatos del Champion
│   ├── candidate_model.joblib      # Modelo Candidate (pruebas A/B)
│   ├── candidate_model_info.json   # Metadatos del Candidate
│   ├── metrics_history.json        # Histórico de todos los entrenamientos
│   └── shap_summary.png            # Gráfico SHAP del mejor modelo
├── notebooks/
│   └── eda.ipynb                   # Análisis exploratorio de datos
├── reports/
│   └── data_drift.html             # Informe interactivo de Evidently
├── src/
│   ├── api/
│   │   └── main.py                 # FastAPI: endpoints, pruebas A/B, registro
│   ├── data/
│   │   ├── make_dataset.py         # Limpieza y procesamiento de datos
│   │   └── features.py             # Ingeniería de características
│   ├── models/
│   │   └── train_models.py         # Entrenamiento, selección, rollback
│   └── visualization/
│       ├── monitor_drift.py        # Monitorización de drift + auto-retrain
│       └── visualize.py            # Generación de gráficos SHAP
├── tests/                          # Tests unitarios y de integración
├── utils/                          # Utilidades compartidas
├── Dockerfile                      # Imagen multi-stage para la API
├── Makefile                        # CLI para Linux/macOS/CI
├── run.ps1                         # CLI para Windows (PowerShell)
├── requirements.txt                # Dependencias Python
├── .env.example                    # Variables de entorno de ejemplo
└── pytest.ini                      # Configuración de pytest
```

---

## Requisitos previos

- **Python 3.12+**
- **Docker & Docker Compose** (opcional, para despliegue con contenedores)
- **Git**

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/Garci8/churn-MLOps-platform.git
cd churn-MLOps-platform

# 2. Crear y activar entorno virtual
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
```

---

## Configuración

Toda la configuración está centralizada en [`config/config.yaml`](config/config.yaml). No es necesario modificar nada para ejecutar el proyecto con los valores por defecto; los ajustes más relevantes son:

| Sección | Descripción |
|---|---|
| `data` | Rutas de datos, ratios de división train/val/test, rutas de registros e informes |
| `model` | Versión, umbral de clasificación, modelos a entrenar, columnas categóricas/numéricas |
| `param_grids` | Hiperparámetros para GridSearchCV de cada modelo |
| `mlflow` | Nombre del experimento, modelo registrado, tipos de confianza para skops |

**Conexión a MLflow**: el sistema se conecta a `http://localhost:5000` si el servidor está activo; en caso contrario, usa automáticamente una BD SQLite local en `mlflow/mlflow.db` como respaldo.

---

## Uso

### Pipeline completo

Ejecuta todo el flujo de datos → entrenamiento → tests en un solo comando:

```bash
# Linux / macOS / CI
make pipeline

# Windows (PowerShell)
.\run.ps1 pipeline
```

### Comandos individuales

| Comando | Descripción |
|---|---|
| `make data` / `.\run.ps1 data` | Limpieza y procesamiento del conjunto de datos |
| `make train` / `.\run.ps1 train` | Entrenamiento de modelos con GridSearchCV + selección Champion/Candidate |
| `make test` / `.\run.ps1 test` | Ejecución de tests con pytest |
| `make monitor` / `.\run.ps1 monitor` | Genera informe de drift; si detecta drift, lanza reentrenamiento automático |

### API de predicción

```bash
# Lanzar la API en local
uvicorn src.api.main:app --reload --port 8000
```

La API estará disponible en `http://localhost:8000` (redirige a `/docs` para la documentación interactiva Swagger).

**Endpoints disponibles:**

| Método | Endpoint | Descripción |
|---|---|---|
| `POST` | `/v1/predict` | Predicción binaria (churn: true/false) |
| `POST` | `/v2/predict` | Predicción con probabilidad |
| `GET` | `/health` | Estado del servicio |
| `GET` | `/model/info` | Metadatos del modelo en producción |

**Ejemplo de petición:**

```bash
curl -X POST http://localhost:8000/v2/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Respuesta:**

```json
{
  "churn": true,
  "probabilidad_churn": 0.8231,
  "model_variant": "A"
}
```

### Docker Compose

Levanta MLflow + API con un solo comando:

```bash
cd docker
docker compose up -d
```

| Servicio | Puerto | URL |
|---|---|---|
| MLflow UI | 5000 | `http://localhost:5000` |
| Churn API | 8000 | `http://localhost:8000/docs` |

---

## Componentes

### 1. Pipeline de datos

**`src/data/make_dataset.py`** — Carga el CSV original y aplica limpieza:
- Elimina `customerID`
- Convierte `TotalCharges` a numérico (maneja cadenas vacías)
- Codifica `Churn` como 1/0

**`src/data/features.py`** — Ingeniería de características:
- `AvgChargePerMonth`: gasto medio mensual
- `ChargeGap`: diferencia entre gasto real y esperado
- `NewCustomer`: indicador si tenure < 12 meses
- `NumServices`: total de servicios contratados

### 2. Pipeline de entrenamiento

**`src/models/train_models.py`** — Entrenamiento completo:
- **3 modelos**: Logistic Regression, Random Forest, XGBoost
- **Línea base**: DummyClassifier estratificado
- **Optimización**: GridSearchCV con F1-Score y validación cruzada de 5 iteraciones
- **División**: 70% Train / 15% Validación / 15% Test (estratificado)
- **Selección**: Champion (mejor F1) + Candidate (segundo mejor)
- **Balanceo de clases**: `class_weight="balanced"` / `scale_pos_weight`
- **MLflow**: registra parámetros, métricas y modelos como artefactos
- **Rollback**: solo sobrescribe el modelo en disco si el nuevo supera al anterior
- **SHAP**: genera automáticamente gráfico de importancia de características
- **Histórico**: guarda todas las ejecuciones en `metrics_history.json`

### 3. API de servicio

**`src/api/main.py`** — FastAPI con:
- **Carga al inicio** (`lifespan`): precarga Champion y Candidate en memoria
- **Pruebas A/B**: 80% tráfico al Champion, 20% al Candidate
- **Versionado**: `/v1/predict` (binario) y `/v2/predict` (con probabilidad)
- **Registro**: cada predicción se guarda en `logs/predictions.jsonl` con marca temporal, modelo usado y variante
- **Validación**: esquema Pydantic con ejemplos interactivos en Swagger
- **Gestión de errores**: handler global de excepciones

### 4. Monitorización y drift

**`src/visualization/monitor_drift.py`** — Monitorización con Evidently:
- Compara distribuciones de las características de entrenamiento vs predicciones en producción
- Genera informe HTML interactivo en `reports/data_drift.html`
- Si detecta **drift global**, lanza automáticamente `train_models.main()` para reentrenar

### 5. CI/CD

**`ci.yml`** — Se ejecuta en push/PR a `main`:
1. Lint con flake8
2. Pipeline completo (`make pipeline`)
3. Comprobación de drift (`make monitor`)
4. Construcción y publicación de imagen Docker a GHCR (solo en push a `main`)

**`scheduled_retrain.yml`** — Reentrenamiento programado:
- Cron: **día 1 de cada mes** a las 00:00 UTC
- También ejecutable manualmente vía `workflow_dispatch`
- Ejecuta: `data` → `train` → `test` → `monitor`

---

## Pruebas A/B

El sistema implementa pruebas A/B sobre el mismo endpoint:

- **Variante A (Champion)**: el mejor modelo seleccionado, recibe el **80%** del tráfico
- **Variante B (Candidate)**: el segundo mejor modelo, recibe el **20%** del tráfico

Cada respuesta incluye el campo `model_variant` (`"A"` o `"B"`) para poder analizar el rendimiento de cada variante en producción. Ambos modelos se cargan en memoria al iniciar la API.

---

## Rollback automático

El pipeline de entrenamiento implementa un mecanismo de protección:

1. Antes de guardar el nuevo modelo, carga el modelo existente en disco
2. Evalúa ambos modelos (antiguo y nuevo) sobre el mismo conjunto de validación
3. **Solo sobrescribe** si el nuevo modelo tiene un **F1-Score estrictamente superior**
4. Si el nuevo modelo es peor o igual, se imprime `[ROLLBACK]` y se mantiene el modelo anterior

Esto garantiza que el modelo en producción **nunca degrada**, incluso en ejecuciones automatizadas con cron.

---

## Tests

```bash
# Ejecutar toda la suite
make test

# O directamente con pytest
pytest tests/ -v
```

Los tests cubren:
- **Datos**: limpieza, ingeniería de características, carga de conjuntos de datos
- **Modelos**: pipeline de entrenamiento, serialización, métricas
- **API**: endpoints de predicción, estado del servicio, validación de entrada
- **Visualización**: generación de informes de drift

---

## Conjunto de datos

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — conjunto de datos público de Kaggle con ~7.000 registros de clientes de una empresa de telecomunicaciones. Incluido directamente en el repositorio (`data/raw/`) dado su reducido tamaño (< 1 MB).

**Variable objetivo**: `Churn` (Yes/No → 1/0)

---

## Licencia

MIT
