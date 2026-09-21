.PHONY: all data train test monitor pipeline help

PYTHON = python

help:
	@echo "Comandos disponibles:"
	@echo "  make data            - Ejecuta la limpieza y procesamiento de datos"
	@echo "  make train           - Entrena los modelos y los registra en MLflow"
	@echo "  make test            - Ejecuta la suite de pruebas unitarias e integración con pytest"
	@echo "  make monitor         - Genera el informe de Data & Target Drift con Evidently"
	@echo "  make pipeline        - Ejecuta todo el pipeline (data -> train -> test)"

data:
	@echo "==> Ejecutando procesamiento de datos..."
	PYTHONPATH=. $(PYTHON) src/data/make_dataset.py

train:
	@echo "==> Ejecutando entrenamiento de modelos..."
	PYTHONPATH=. $(PYTHON) src/models/train_models.py

test:
	@echo "==> Ejecutando pruebas unitarias..."
	PYTHONPATH=. pytest tests/

monitor:
	@echo "==> Generando informe de Data & Target Drift con Evidently..."
	PYTHONPATH=. $(PYTHON) src/visualization/monitor_drift.py

pipeline: data train test
all: pipeline
