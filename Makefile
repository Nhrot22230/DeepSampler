# =============================================================================
# Makefile para DeepSampler
#
# Información del proyecto:
#   - Nombre: DeepSampler
#   - Versión: 0.0.1
#   - Autores:
#       * Diego Izaguirre (A01024053)
#       * Fabian Prado   (A01024053)
#       * Fernando Candia (A01024053)
#
# =============================================================================

# Variables de colores
GREEN  := \033[1;32m
YELLOW := \033[1;33m
BLUE   := \033[1;34m
RED    := \033[1;31m
NC     := \033[0m

.PHONY: help init install freeze pipe-data pipe-train pipe-eval pipe-infer test format clean clean-data clean-all

PYTHON ?= python3
PIP    ?= pip3

SETUP_SCRIPT      := setup.py
SRC_DIR           := src
PIPELINE_DATA     := $(SRC_DIR)/pipelines/data.py
PIPELINE_TRAIN    := $(SRC_DIR)/pipelines/train.py
PIPELINE_INFER    := $(SRC_DIR)/pipelines/inference.py
PIPELINE_EVAL     := $(SRC_DIR)/pipelines/eval.py
TESTS_DIR         := test
REQUIREMENTS_FILE := requirements.txt

export PYTHONPATH := $(CURDIR)

# Banner ASCII para la cabecera del help
BANNER := "\n$(BLUE)==================================================================\n                   DEEPSAMPLER - Makefile\n==================================================================\n$(NC)"

# -------------------------------------------------------------------------------
# Ayuda e información general
# -------------------------------------------------------------------------------
help:
	@echo -e $(BANNER)
	@echo -e "$(GREEN)Comandos disponibles:$(NC)"
	@echo -e "  $(YELLOW)init$(NC)         - Ejecuta setup.py para inicializar el proyecto."
	@echo -e "  $(YELLOW)pipe-data$(NC)    - Ejecuta el pipeline de data: $(PIPELINE_DATA)."
	@echo -e "  $(YELLOW)pipe-train$(NC)   - Ejecuta el pipeline de entrenamiento: $(PIPELINE_TRAIN)."
	@echo -e "  $(YELLOW)pipe-eval$(NC)    - Ejecuta el pipeline de evaluación: $(PIPELINE_EVAL)."
	@echo -e "  $(YELLOW)pipe-infer$(NC)   - Ejecuta el pipeline de inferencia: $(PIPELINE_INFER)."
	@echo -e "  $(YELLOW)test$(NC)         - Ejecuta los tests (unittest)."
	@echo -e "  $(YELLOW)format$(NC)       - Formatea el código (isort, black y flake8)."
	@echo -e "  $(YELLOW)install$(NC)      - Instala dependencias desde $(REQUIREMENTS_FILE) y el proyecto en modo editable."
	@echo -e "  $(YELLOW)freeze$(NC)       - Congela las dependencias actuales a $(REQUIREMENTS_FILE)."
	@echo -e "  $(YELLOW)clean$(NC)        - Limpia archivos generados (carpetas __pycache__, build, dist, etc.)."
	@echo -e "  $(YELLOW)clean-data$(NC)   - Limpia archivos de data procesada."
	@echo -e "  $(YELLOW)clean-all$(NC)    - Ejecuta clean y clean-data."

# -------------------------------------------------------------------------------
# Inicialización e instalación
# -------------------------------------------------------------------------------
init:
	@echo -e "$(GREEN)Ejecutando setup.py para inicializar el proyecto...$(NC)"
	$(PYTHON) $(SETUP_SCRIPT)
	@rm -rf data/musdb18hq/*
	@mkdir -p data/musdb18hq
	@unzip data/raw/MUSDB18_HQ.zip -d data/musdb18hq
	@echo -e "$(GREEN)Inicialización completada.$(NC)"

install:
	@echo -e "$(GREEN)Instalando dependencias desde $(REQUIREMENTS_FILE)...$(NC)"
	$(PIP) install -r $(REQUIREMENTS_FILE)

freeze:
	@echo -e "$(GREEN)Congelando dependencias en $(REQUIREMENTS_FILE)...$(NC)"
	$(PIP) freeze > $(REQUIREMENTS_FILE)

# -------------------------------------------------------------------------------
# Pipelines
# -------------------------------------------------------------------------------
pipe-data:
	@echo -e "$(GREEN)Ejecutando pipeline de data...$(NC)"
	@rm -rf data/processed/*
	@mkdir -p data/processed
	$(PYTHON) $(PIPELINE_DATA) $(ARGS)
	@echo -e "$(GREEN)Pipeline de data completado.$(NC)"

pipe-train:
	@echo -e "$(GREEN)Ejecutando pipeline de entrenamiento...$(NC)"
	$(PYTHON) $(PIPELINE_TRAIN)
	@echo -e "$(GREEN)Pipeline de entrenamiento completado.$(NC)"

pipe-eval:
	@echo -e "$(GREEN)Ejecutando pipeline de evaluación...$(NC)"
	$(PYTHON) $(PIPELINE_EVAL)
	@echo -e "$(GREEN)Pipeline de evaluación completado.$(NC)"

pipe-infer:
	@echo -e "$(GREEN)Ejecutando pipeline de inferencia...$(NC)"
	$(PYTHON) $(PIPELINE_INFER)
	@echo -e "$(GREEN)Pipeline de inferencia completado.$(NC)"

# -------------------------------------------------------------------------------
# Testing y formateo
# -------------------------------------------------------------------------------
test:
	@echo -e "$(GREEN)Ejecutando tests...$(NC)"
	pytest $(TESTS_DIR)
	@echo -e "$(GREEN)Tests completados.$(NC)"

format:
	@echo -e "$(GREEN)Ejecutando flake8...$(NC)"
	flake8 --config=configs/.flake8 .
	@echo -e "$(GREEN)Ejecutando isort...$(NC)"
	isort --sp configs/.isort.cfg .
	@echo -e "$(GREEN)Ejecutando black...$(NC)"
	black .
	@echo -e "$(GREEN)Formateo completado.$(NC)"

# -------------------------------------------------------------------------------
# Limpieza
# -------------------------------------------------------------------------------
clean:
	@echo -e "$(RED)Limpiando archivos generados...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@rm -rf build dist
	@echo -e "$(RED)Limpieza completada.$(NC)"

clean-data:
	@echo -e "$(RED)Limpiando archivos de data procesada...$(NC)"
	@rm -rf data/processed/*
	@echo -e "$(RED)Limpieza de data completada.$(NC)"

clean-all: clean clean-data
	@echo -e "$(RED)Limpieza total completada.$(NC)"
