#!/bin/bash
# Este script crea una estructura de directorios robusta para un proyecto de deep learning.
# Se requiere pasar el nombre del directorio base como argumento.
# Ejemplo de uso: ./create_structure.sh mi_proyecto

# Salir inmediatamente si ocurre algún error
set -e

# Función para mostrar el mensaje de uso
usage() {
    echo "Uso: $0 <nombre_del_directorio_del_proyecto>"
    exit 1
}

# Verificar que se haya pasado un argumento
if [ -z "$1" ]; then
    echo "Error: No se proporcionó el nombre del directorio del proyecto."
    usage
fi

PROJECT_ROOT="$1"

# Verificar que el nombre no sea solo espacios
if [[ "$PROJECT_ROOT" =~ ^[[:space:]]*$ ]]; then
    echo "Error: El nombre del directorio no puede estar vacío o contener solo espacios."
    usage
fi

# Validar que el nombre no sea '.' o '..'
if [[ "$PROJECT_ROOT" == "." || "$PROJECT_ROOT" == ".." ]]; then
    echo "Error: '$PROJECT_ROOT' no es un nombre de directorio válido."
    usage
fi

# Comprobar si el directorio ya existe
if [ -d "$PROJECT_ROOT" ]; then
    echo "Error: El directorio '$PROJECT_ROOT' ya existe. Elija otro nombre o elimine el directorio existente."
    exit 1
fi

echo "Creando la estructura de directorios en '$PROJECT_ROOT'..."

# Create directories
mkdir -p "$PROJECT_ROOT/configs/experiments"
mkdir -p "$PROJECT_ROOT/data/external"
mkdir -p "$PROJECT_ROOT/data/raw"
mkdir -p "$PROJECT_ROOT/data/processed"
mkdir -p "$PROJECT_ROOT/experiments/logs"
mkdir -p "$PROJECT_ROOT/experiments/checkpoints"
mkdir -p "$PROJECT_ROOT/experiments/results"
mkdir -p "$PROJECT_ROOT/notebooks"
mkdir -p "$PROJECT_ROOT/scripts"
mkdir -p "$PROJECT_ROOT/src/data"
mkdir -p "$PROJECT_ROOT/src/models"
mkdir -p "$PROJECT_ROOT/src/pipelines"
mkdir -p "$PROJECT_ROOT/src/training"
mkdir -p "$PROJECT_ROOT/src/evaluation"
mkdir -p "$PROJECT_ROOT/src/utils"
mkdir -p "$PROJECT_ROOT/tests"

# Create placeholder files

# Configs
touch "$PROJECT_ROOT/configs/config.yaml"
touch "$PROJECT_ROOT/configs/experiments/exp1.yaml"
touch "$PROJECT_ROOT/configs/experiments/exp2.yaml"

# Notebooks
touch "$PROJECT_ROOT/notebooks/exploratory.ipynb"
touch "$PROJECT_ROOT/notebooks/analysis.ipynb"

# Scripts
touch "$PROJECT_ROOT/scripts/run_training.sh"
touch "$PROJECT_ROOT/scripts/run_evaluation.sh"

# Source code initialization files and modules
touch "$PROJECT_ROOT/src/__init__.py"

# src/data
touch "$PROJECT_ROOT/src/data/__init__.py"
touch "$PROJECT_ROOT/src/data/dataset.py"
touch "$PROJECT_ROOT/src/data/transforms.py"

# src/models
touch "$PROJECT_ROOT/src/models/__init__.py"
touch "$PROJECT_ROOT/src/models/base_model.py"
touch "$PROJECT_ROOT/src/models/sde_model.py"

# src/pipelines
touch "$PROJECT_ROOT/src/pipelines/__init__.py"
touch "$PROJECT_ROOT/src/pipelines/train_pipeline.py"

# src/training
touch "$PROJECT_ROOT/src/training/__init__.py"
touch "$PROJECT_ROOT/src/training/trainer.py"
touch "$PROJECT_ROOT/src/training/optimizer.py"

# src/evaluation
touch "$PROJECT_ROOT/src/evaluation/__init__.py"
touch "$PROJECT_ROOT/src/evaluation/tester.py"
touch "$PROJECT_ROOT/src/evaluation/metrics.py"

# src/utils
touch "$PROJECT_ROOT/src/utils/__init__.py"
touch "$PROJECT_ROOT/src/utils/logger.py"
touch "$PROJECT_ROOT/src/utils/config_parser.py"
touch "$PROJECT_ROOT/src/utils/helpers.py"

# Tests
touch "$PROJECT_ROOT/tests/__init__.py"
touch "$PROJECT_ROOT/tests/test_data.py"
touch "$PROJECT_ROOT/tests/test_model.py"
touch "$PROJECT_ROOT/tests/test_pipeline.py"

# Root-level files
touch "$PROJECT_ROOT/environment.yml"
touch "$PROJECT_ROOT/requirements.txt"
touch "$PROJECT_ROOT/README.md"
touch "$PROJECT_ROOT/setup.py"

echo "Project directory structure created successfully."
