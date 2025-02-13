# DeepSampler: Separación de Fuentes de Audio en Tiempo Real

## Introducción

En el complejo panorama de señales sonoras que nos rodea, la capacidad de aislar y distinguir fuentes de audio específicas se ha convertido en un desafío tecnológico fundamental. La separación de fuentes de audio emerge como una tecnología crítica que busca replicar y superar la extraordinaria capacidad humana de enfocarse en sonidos de interés en entornos acústicamente densos.

Esta tecnología tiene aplicaciones amplias y transformadoras, desde sistemas de comunicación y asistentes de voz hasta dispositivos de ayuda auditiva, mejora de grabaciones musicales y sistemas de reconocimiento de voz. Los avances recientes en inteligencia artificial —particularmente en redes neuronales profundas y técnicas como la factorización de matrices no negativas— han impulsado la precisión y eficacia de estos métodos.

## Audio Source Separation

La separación de fuentes de audio es una disciplina avanzada del procesamiento de señales que tiene como objetivo estimar y extraer señales sonoras individuales a partir de mezclas complejas. En esencia, se busca descomponer grabaciones que contienen múltiples fuentes sonoras (como voces, instrumentos musicales o ruidos ambientales) en sus componentes originales, permitiendo un análisis y manipulación precisos.

Entre los enfoques más utilizados se encuentran:
- **Factorización No Negativa de Matrices (NMF)**
- **Redes Neuronales Profundas (DNN)**
- **Análisis de Componentes Esparsos (SCA)**

## Planteamiento del Problema

A pesar de la existencia de diversos modelos y trabajos enfocados en la separación de fuentes de audio, hasta la fecha no se ha logrado implementar una separación en tiempo real. Esto representa una oportunidad significativa para mejorar y revolucionar múltiples campos:

- **Vehículos Autónomos:** Permitiendo el reconocimiento de sonidos críticos, como una sirena de policía.
- **Transcripción de Voz a Texto:** Mejorando la precisión al separar la voz del ruido ambiental.
- **Industria Musical:** Facilitando a músicos y equipos de sonido el aislamiento de instrumentos específicos para una edición y análisis más detallados.

El principal desafío identificado es la falta de un modelo capaz de realizar la separación de fuentes de audio en tiempo real, abriendo la puerta a aplicaciones innovadoras en diversos sectores.

## Propuesta de Solución

### DeepSampler: Arquitectura Híbrida para la Separación de Audio

En este proyecto proponemos **DeepSampler**, una novedosa arquitectura basada en U-Net que incorpora un Transformer en el espacio latente para abordar el reto de la separación de fuentes de audio en tiempo real.

#### Detalles de la Arquitectura

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

## Estructura del Repositorio

A continuación se muestra un ejemplo de una estructura de directorios robusta y escalable para un proyecto de deep learning. Esta estructura está diseñada para separar las responsabilidades, tales como manejo de datos, definición del modelo, pipelines de entrenamiento, evaluación y experimentación, e incorpora las mejores prácticas en investigación y desarrollo de software para machine learning.

```
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
|   |   ├── data.py            # Código para inicializar datos
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
```

---

### **Explicación de Carpetas Clave**

- **configs/**
  Aquí se almacenan distintos archivos de configuración (YAML, JSON, etc.) que definen hiperparámetros, rutas de datos, parámetros del modelo y otros ajustes, permitiendo cambiar de experimento sin modificar el código.

- **data/**
  Organiza tus datos en subcarpetas:
  - **raw/**: Datos originales sin modificar.
  - **processed/**: Datos procesados y listos para el entrenamiento.
  - **external/**: Datos provenientes de fuentes externas o de terceros.

- **experiments/**
  Contiene logs, checkpoints y resultados de experimentos. Esta separación facilita el seguimiento de las diferentes versiones y comparaciones de rendimiento a lo largo del tiempo.

- **notebooks/**
  Se utilizan para la exploración, visualización y análisis preliminares. Son ideales para la depuración y el desarrollo iterativo.

- **scripts/**
  Scripts (en shell o Python) que permiten ejecutar tareas como el entrenamiento, evaluación o despliegue del modelo desde la línea de comandos.

- **src/**
  Alberga el código fuente principal del proyecto, organizado en módulos que se encargan de:
  - **data/**: Manejo de datasets y transformaciones.
  - **models/**: Definición de arquitecturas y modelos.
  - **pipelines/**: Integración de datos, modelo y procesos de entrenamiento en un flujo de trabajo completo.
  - **training/**: Bucles de entrenamiento, optimizadores y estrategias de actualización.
  - **evaluation/**: Código para evaluar el rendimiento del modelo y calcular métricas.
  - **utils/**: Funciones y utilidades generales (logging, parseo de configuraciones, etc.).

- **tests/**
  Contiene pruebas unitarias e integradas para garantizar la robustez y correcto funcionamiento del código. Se recomienda utilizar frameworks como PyTest para automatizar estas pruebas.

- **environment.yml & requirements.txt**
  Archivos que definen las dependencias del proyecto, ya sea mediante Conda o pip.

- **setup.py**
  Facilita el empaquetado y distribución del proyecto como un módulo reutilizable.

---

### **Aplicaciones y Contribuciones**

La implementación de DeepSampler promete abrir nuevas posibilidades en:
- **Sistemas de Seguridad y Vehículos Autónomos:** Reconocimiento de sonidos críticos en tiempo real.
- **Asistentes de Voz y Transcripción:** Mejora en la separación y procesamiento de la voz en entornos ruidosos.
- **Industria Musical:** Aislamiento y manipulación de instrumentos en mezclas complejas.

