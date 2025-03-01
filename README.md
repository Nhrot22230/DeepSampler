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

En la industria musical y en la producción de audio, existe una necesidad recurrente de trabajar con pistas de sonido individuales, es decir, de poder aislar elementos específicos de una canción, como la voz o los instrumentos, para distintos fines. Sin embargo, separar estos componentes de una pista de audio completa no es una tarea sencilla y, hasta hace poco, los métodos tradicionales eran manuales, tardados y no siempre efectivos.

Diferentes profesionales y entusiastas de la música enfrentan dificultades debido a la falta de herramientas accesibles y eficientes para este propósito. Algunas de las principales aplicaciones donde la separación de fuentes musicales es clave incluyen:

**Sampleo de Voces para Producción Musical**:
En la música electrónica, los productores suelen necesitar fragmentos de voces aisladas (sin la música de fondo) para crear nuevas composiciones. Tradicionalmente, esto se lograba con técnicas complejas, como invertir la fase de una pista instrumental para "cancelar" el fondo musical y quedarse solo con la voz. Sin embargo, este método requiere mucho tiempo y mayor expertiz.

**Transcripción de Música**:
Cuando un músico necesita escribir la partitura de una canción, es común que tenga dificultades para distinguir ciertos sonidos, especialmente los de frecuencias graves, como el bajo o el bombo de la batería. Al separar los elementos del audio en pistas individuales, la transcripción se vuelve más clara y precisa.

**Creación de Pistas de Karaoke**:
Para generar versiones instrumentales de canciones (karaoke), es necesario eliminar la voz principal sin afectar el acompañamiento musical. Un sistema de separación eficiente facilita este proceso sin comprometer necesariamente la calidad del sonido.

**Reemplazo de Voces con Inteligencia Artificial**:
En la industria de la producción musical, ha surgido un mercado donde se "imita" la voz de artistas famosos para interpretar canciones que originalmente no grabaron. Para lograrlo, primero es necesario extraer la voz original de una canción y luego procesarla con inteligencia artificial antes de integrarla nuevamente en la mezcla.

**Uso en Tecnología Vocaloid y Síntesis de Voz**:
La síntesis de voz, como la utilizada en Vocaloid, se beneficia de la separación de fuentes para comprender mejor la estructura de las voces reales y poder generar nuevas interpretaciones con voces sintéticas.

A pesar de la creciente necesidad de este tipo de herramientas, muchas de las soluciones actuales requieren conocimientos avanzados en producción musical o no son accesibles para el público general. Por ello, este proyecto busca desarrollar una aplicación intuitiva y eficiente basada en modelos de inteligencia artificial para la separación de fuentes musicales, facilitando su uso tanto para profesionales como para aficionados.

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
DeepSampler/
├── configs/                   # Archivos de configuración para experimentos y parámetros globales.
│   ├── config.yaml            # Configuración principal (parámetros del modelo, rutas, etc.).
│   └── experiments/           # Configuraciones específicas para distintos experimentos.
├── data/                      # Datos utilizados en el proyecto.
│   ├── raw/                   # Datos originales sin procesar.
|   └── musdb18hq/             # MUSDB18_HQ dataset.
├── experiments/               # Resultados, logs y checkpoints de los entrenamientos.
│   ├── checkpoints/           # Pesos guardados y checkpoints de los modelos.
│   ├── logs/                  # Logs generados durante los entrenamientos.
│   └── results/               # Resultados de evaluación y salidas de los modelos.
├── notebooks/                 # Notebooks Jupyter para exploración, análisis y prototipado.
│   ├── analysis.ipynb
│   ├── exploratory.ipynb
│   ├── loss.ipynb
│   └── trail_error.ipynb
├── environment.yml            # Archivo de entorno Conda (si se utiliza Conda) con todas las dependencias.
├── Makefile                   # Makefile para automatizar tareas comunes (instalación, ejecución, etc.).
├── README.md                  # Documentación general del proyecto e instrucciones de uso.
├── requirements.txt           # Dependencias a instalar vía pip.
├── scripts/                   # Scripts de ejecución en Shell para tareas específicas.
├── setup.py                   # Script de instalación del paquete (si se desea empaquetar el proyecto).
├── setup.sh                   # Script de configuración inicial (por ejemplo, para crear entornos o instalar dependencias).
├── src/                       # Código fuente principal del proyecto.
│   ├── __init__.py
│   ├── models/                # Definición y arquitecturas de los modelos.
│   │   ├── __init__.py
│   │   ├── deep_sampler.py    # Modelo DeepSampler.
│   │   ├── dense_net.py       # Modelo basado en DenseNet.
│   │   ├── scunet.py          # Modelo SCUNet.
│   │   ├── u_net.py           # Modelo U-Net.
│   │   └── components/        # Componentes reutilizables para la construcción de modelos.
│   │       ├── decoder.py
│   │       ├── encoder.py
│   │       ├── freq_conv.py
│   │       └── __init__.py
│   ├── pipelines/             # Pipelines de extremo a extremo (entrenamiento, inferencia, evaluación, etc.).
│   │   ├── __init__.py
│   │   ├── data.py            # Código para la inicialización y manejo de datos.
│   │   ├── eval.py            # Pipeline de evaluación.
│   │   ├── inference.py       # Pipeline de inferencia.
│   │   └── train.py           # Pipeline de entrenamiento.
│   └── utils/                 # Funciones utilitarias y herramientas de soporte.
│       ├── logging/           # Configuración y utilidades para el logging personalizado.
│       │   ├── __init__.py
│       │   └── logger.py
│       ├── training/          # Utilidades para el entrenamiento (pérdidas, entrenadores, etc.).
│       │   ├── __init__.py
│       │   ├── loss.py        # Definición de funciones de pérdida.
│       │   └── trainer.py     # Implementación del ciclo de entrenamiento.
│       ├── data/              # Utilidades relacionadas con el manejo y preprocesamiento de datos.
│       │   └── dataset/       # Definición de datasets personalizados.
│       │       ├── __init__.py
│       │       └── musdb18_dataset.py
│       └── audio/             # Procesamiento y manipulación de audio.
│           ├── __init__.py
│           ├── audio_chunk.py
│           └── processing.py
└── tests/                     # Pruebas unitarias e integradas para el proyecto.
    ├── __init__.py
    ├── test_data.py
    ├── test_model.py
    └── test_pipeline.py
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
