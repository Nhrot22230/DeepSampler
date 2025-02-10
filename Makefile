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
# =============================================================================

.PHONY: help init pipe-data pipe-train test format install freeze docs clean clean-data clean-all

PYTHON ?= python3
PIP    ?= pip3

SETUP_SCRIPT      := setup.py
SRC_DIR           := src
PIPELINE_DATA     := $(SRC_DIR)/pipelines/data.py
PIPELINE_TRAIN    := $(SRC_DIR)/pipelines/train.py
TESTS_DIR         := tests
REQUIREMENTS_FILE := requirements.txt
VENV_DIR          := venv
DOCS_BUILD_DIR    := docs/_build


help:
	@echo "DeepSampler Makefile - Comandos disponibles:"
	@echo "  make init          - Ejecuta setup.py para inicializar el proyecto."
	@echo "  make pipe-data     - Ejecuta el pipeline de data: $(PIPELINE_DATA)."
	@echo "  make pipe-train    - Ejecuta el pipeline de entrenamiento: $(PIPELINE_TRAIN)."
	@echo "  make test          - Ejecuta los tests (unittest)."
	@echo "  make format        - Formatea el código (isort y black)."
	@echo "  make install       - Instala dependencias desde $(REQUIREMENTS_FILE) y el proyecto en modo editable."
	@echo "  make freeze        - Congela las dependencias actuales a $(REQUIREMENTS_FILE)."
	@echo "  make docs          - Genera la documentación con Sphinx."
	@echo "  make clean         - Limpia archivos generados (carpetas __pycache__, build, dist, etc.)."
	@echo "  make clean-data    - Limpia archivos de data procesada."
	@echo "  make clean-all     - Limpia todo: archivos generados y data procesada."

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

format:
	@echo "Ejecutando flake8..."
	flake8 --config=configs/.flake8 .
	@echo "Ejecutando black para formatear el código..."
	isort --sp configs/.isort.cfg .
	black .

install:
	@echo "Instalando dependencias desde $(REQUIREMENTS_FILE)..."
	$(PIP) install -r $(REQUIREMENTS_FILE)

freeze:
	@echo "Congelando dependencias en $(REQUIREMENTS_FILE)..."
	$(PIP) freeze > $(REQUIREMENTS_FILE)

docs:
	@echo "Generando documentación con Sphinx..."
	sphinx-build -b html $(SRC_DIR) $(DOCS_BUILD_DIR)
	@echo "Documentación generada en $(DOCS_BUILD_DIR)."

clean:
	@echo "Limpiando archivos generados..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@rm -rf build dist
	@echo "Limpieza completada."

clean-data:
	@echo "Limpiando archivos de data procesada..."
	@rm -rf data/processed/*
	@echo "Limpieza de data completada."

clean-all: clean clean-data
	@echo "Limpieza total completada."
