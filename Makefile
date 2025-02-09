# =============================================================================
# Makefile para DeepSampler
#
# Información del proyecto:
#   - Nombre: DeepSampler
#   - Versión: 0.0.1
#   - Autores:
#       * Diego Izaguirre (A01024053)
#       * Fabian Prado (A01024053)
#       * Fernando Candia (A01024053)
#
#   - Dataset: MUSDB18_HQ
#     URL: https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1
#
# Este Makefile define comandos para:
#   - Inicializar el proyecto (setup.py)
#   - Ejecutar el pipeline de data (src/pipelines/data.py)
#   - Ejecutar el pipeline de entrenamiento (src/pipelines/train.py)
#   - Ejecutar tests, formateo de código y limpieza
# =============================================================================

.PHONY: help init pipe-data pipe-train test lint format clean clean-data

PYTHON ?= python3

SETUP_SCRIPT       := setup.py
PIPELINE_DATA      := src/pipelines/data.py
PIPELINE_TRAIN     := src/pipelines/train.py
TESTS_DIR          := tests

help:
	@echo "DeepSampler Makefile - Comandos disponibles:"
	@echo "  make init          - Ejecuta setup.py para inicializar el proyecto."
	@echo "  make pipe-data     - Ejecuta el pipeline de data: $(PIPELINE_DATA)."
	@echo "  make pipe-train    - Ejecuta el pipeline de entrenamiento: $(PIPELINE_TRAIN)."
	@echo "  make test          - Ejecuta los tests (unittest)."
	@echo "  make lint          - Ejecuta linter (flake8)."
	@echo "  make format        - Ejecuta formateador de código (black)."
	@echo "  make clean         - Limpia archivos generados (carpetas __pycache__, data procesada, etc.)."
	@echo "  make clean-data    - Limpia archivos de data procesada."

init:
	@echo "Ejecutando setup.py para inicializar el proyecto..."
	$(PYTHON) $(SETUP_SCRIPT)

pipe-data:
	@echo "Ejecutando pipeline de data..."
	$(PYTHON) $(PIPELINE_DATA)

pipe-train:
	@echo "Ejecutando pipeline de entrenamiento..."
	$(PYTHON) $(PIPELINE_TRAIN)

test:
	@echo "Ejecutando tests..."
	$(PYTHON) -m unittest discover $(TESTS_DIR)

lint:
	@echo "Ejecutando flake8..."
	flake8 --config=configs/.flake8 .

format:
	@echo "Ejecutando black para formatear el código..."
	isort --sp configs/.isort.cfg .
	black .

clean:
	@echo "Limpiando archivos generados..."
	@rm -rf __pycache__
	@rm -rf **/__pycache__
	@echo "Limpieza completada."

clean-data:
	@echo "Limpiando archivos de data procesada..."
	@rm -rf data/
	@echo "Limpieza completada."