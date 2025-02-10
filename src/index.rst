DeepSampler: Separación de Fuentes de Audio en Tiempo Real
===========================================================

.. contents:: Tabla de Contenidos
   :local:
   :depth: 2

Introducción
------------
En el complejo panorama de señales sonoras que nos rodea, la capacidad de aislar y distinguir fuentes de audio específicas se ha convertido en un desafío tecnológico fundamental. La separación de fuentes de audio emerge como una tecnología crítica que busca replicar y superar la extraordinaria capacidad humana de enfocarse en sonidos de interés en entornos acústicamente densos.

Esta tecnología tiene aplicaciones amplias y transformadoras, desde sistemas de comunicación y asistentes de voz hasta dispositivos de ayuda auditiva, mejora de grabaciones musicales y sistemas de reconocimiento de voz. Los avances recientes en inteligencia artificial —particularmente en redes neuronales profundas y técnicas como la factorización de matrices no negativas— han impulsado la precisión y eficacia de estos métodos.

Audio Source Separation
-----------------------
La separación de fuentes de audio es una disciplina avanzada del procesamiento de señales que tiene como objetivo estimar y extraer señales sonoras individuales a partir de mezclas complejas. En esencia, se busca descomponer grabaciones que contienen múltiples fuentes sonoras (como voces, instrumentos musicales o ruidos ambientales) en sus componentes originales, permitiendo un análisis y manipulación precisos.

Entre los enfoques más utilizados se encuentran:

- **Factorización No Negativa de Matrices (NMF)**
- **Redes Neuronales Profundas (DNN)**
- **Análisis de Componentes Esparsos (SCA)**

Planteamiento del Problema
--------------------------
A pesar de la existencia de diversos modelos y trabajos enfocados en la separación de fuentes de audio, hasta la fecha no se ha logrado implementar una separación en tiempo real. Esto representa una oportunidad significativa para mejorar y revolucionar múltiples campos:

- **Vehículos Autónomos:** Permitiendo el reconocimiento de sonidos críticos, como una sirena de policía.
- **Transcripción de Voz a Texto:** Mejorando la precisión al separar la voz del ruido ambiental.
- **Industria Musical:** Facilitando a músicos y equipos de sonido el aislamiento de instrumentos específicos para una edición y análisis más detallados.

Propuesta de Solución
---------------------
DeepSampler: Arquitectura Híbrida para la Separación de Audio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
En este proyecto proponemos **DeepSampler**, una novedosa arquitectura basada en U-Net que incorpora un Transformer en el espacio latente para abordar el reto de la separación de fuentes de audio en tiempo real.

Detalles de la Arquitectura
^^^^^^^^^^^^^^^^^^^^^^^^^^^
- **Encoder:**

  - Utilización de convoluciones 1D para capturar representaciones tanto espectrales como temporales.
  - Downsampling progresivo mediante convoluciones con stride.
  - Normalización de capas y uso de activaciones no lineales (ReLU/Leaky ReLU).

- **Espacio Latente con Transformer:**

  - Sustitución de módulos LSTM tradicionales por un Transformer avanzado, específicamente la variante **BS-RoFormer**.
  - División en subbandas que permite un procesamiento jerárquico de la información espectral.
  - Aplicación de Multi-Head Self-Attention tanto a nivel de banda (Time-Transformer) como entre bandas (Subband-Transformer).
  - Implementación de Rotary Position Embeddings (RoPE) para mejorar la representación de las relaciones posicionales.

- **Decoder:**

  - Uso de convoluciones transpuestas para realizar el upsampling progresivo.
  - Incorporación de conexiones de salto (skip connections) para preservar detalles espectrales importantes.
  - Generación de una máscara de magnitud para cada fuente de audio que permite una reconstrucción precisa.

- **Post-Procesamiento:**

  - Aplicación de la máscara de magnitud sobre el espectrograma de entrada.
  - Reconstrucción de la señal en el dominio temporal mediante la Transformada Inversa de STFT (iSTFT).

Estructura del Repositorio
--------------------------
La siguiente estructura de directorios representa una organización robusta y escalable para este proyecto:

.. code-block:: text

   project_root/
   ├── configs/                   # Archivos de configuración para experimentos y parámetros del modelo.
   │   ├── config.yaml            # Archivo de configuración principal (ej. parámetros del modelo, rutas de datos, etc.)
   │   └── experiments/           # Configuraciones específicas para experimentos (opcional)
   │       ├── exp1.yaml
   │       └── exp2.yaml
   ├── data/                      # Archivos de datos.
   │   ├── external/              # Datos de fuentes externas.
   │   ├── raw/                   # Datos originales sin modificar.
   │   └── processed/             # Datos procesados y limpios para el entrenamiento.
   ├── experiments/               # Artefactos y logs de experimentos.
   │   ├── logs/                  # Logs de entrenamiento (ej. TensorBoard, WandB, etc.)
   │   ├── checkpoints/           # Pesos guardados del modelo y checkpoints.
   │   └── results/               # Resultados de evaluación, figuras o reportes.
   ├── notebooks/                 # Notebooks de Jupyter para exploración, análisis y prototipado.
   │   ├── exploratory.ipynb
   │   └── analysis.ipynb
   ├── scripts/                   # Scripts en Shell o Python para ejecutar tareas.
   │   ├── run_training.sh        # Script para iniciar el entrenamiento.
   │   └── run_evaluation.sh      # Script para ejecutar la evaluación.
   ├── src/                       # Código fuente principal del proyecto.
   │   ├── __init__.py
   │   ├── models/                # Definiciones y arquitecturas del modelo.
   │   │   ├── __init__.py
   │   │   ├── base_model.py      # Clases base o utilidades comunes para modelos.
   │   │   └── sde_model.py       # Ejemplo de modelo para detección de eventos sonoros.
   │   ├── pipelines/             # Pipelines de extremo a extremo (entrenamiento, inferencia, etc.).
   │   │   ├── __init__.py
   │   │   ├── data.py            # Código para inicializar datos.
   │   │   └── train.py           # Código para inicializar modelo, ciclo de entrenamiento, etc.
   │   ├── evaluation/            # Código de evaluación y pruebas.
   │   │   ├── __init__.py
   │   │   ├── tester.py          # Script para ejecutar inferencia o ciclos de prueba.
   │   │   └── metrics.py         # Métricas de evaluación y funciones de análisis.
   │   └── utils/                 # Funciones utilitarias y ayudantes (logging, configuración, etc.).
   │       ├── __init__.py
   │       ├── logger.py          # Configuración personalizada de logging.
   │       ├── config_parser.py   # Utilidades para parsear archivos de configuración.
   │       └── helpers.py         # Funciones de ayuda generales.
   ├── tests/                     # Pruebas unitarias e integradas.
   │   ├── __init__.py
   │   ├── data/                  # Manejo de datos: definiciones de datasets, cargadores, aumentaciones.
   │   │   ├── __init__.py
   │   │   ├── dataset            # Clases de dataset personalizadas.
   │   │   └── transforms         # Utilidades de aumentación y transformación de datos.
   │   ├── training/              # Utilidades de entrenamiento, entrenadores y optimizadores.
   │   │   ├── __init__.py
   │   │   ├── trainer            # Ciclos de entrenamiento encapsulados.
   │   │   └── optimizer          # Configuraciones de optimizadores y schedulers de tasa de aprendizaje.
   │   ├── test_data.py           # Pruebas para pipelines de datos.
   │   ├── test_model.py          # Pruebas para arquitecturas de modelos.
   │   └── test_pipeline.py       # Pruebas para pipelines de entrenamiento/inferencia.
   ├── environment.yml            # Archivo de entorno Conda (si se utiliza Conda).
   ├── requirements.txt           # Dependencias pip.
   ├── README.md                  # Vista general del proyecto e instrucciones de instalación.
   └── setup.py                   # Script de instalación si se empaqueta el proyecto.

Conclusión
----------
DeepSampler representa una solución innovadora para la separación de fuentes de audio en tiempo real, integrando avanzadas técnicas de procesamiento de señales y modelos híbridos basados en U-Net y Transformers. Este proyecto no solo aborda desafíos técnicos importantes, sino que también abre nuevas posibilidades en diversas áreas tecnológicas y de investigación.

.. note::
   Para más detalles, consulta la documentación interna en cada módulo y el README del proyecto.
