# DataScience-III
# Clasificador de Texto para Detección de Phishing con PyTorch y NLP

Este repositorio contiene la resolución práctica de la **Evaluación Final** para la asignatura *Modelos Avanzados de Aprendizaje Automático*. El proyecto consiste en el diseño e implementación de un pipeline completo de Procesamiento del Lenguaje Natural (NLP) y el entrenamiento de una red neuronal profunda utilizando **PyTorch** para clasificar texto libre.

## 🚀 Estructura del Repositorio

* `notebook_evaluacion.ipynb`: Cuaderno de Google Colab con el flujo de código completo (EDA, preprocesamiento, vectorización y entrenamiento).
* `best_model.pth`: Archivo de parámetros y pesos optimizados de la red neuronal, guardado automáticamente mediante la estrategia de *Early Stopping*.
* `requirements.txt`: Listado de librerías base necesarias para garantizar la reproducibilidad del entorno.

---

## 📊 Resumen del Pipeline Implementado

| Etapa | Descripción | Herramientas / Librerías |
| :--- | :--- | :--- |
| **1. Dataset & Auditoría** | Carga y verificación de un corpus de texto crudo con más de 2,000 registros para tareas de seguridad informática. | `pandas` |
| **2. EDA** | Análisis de frecuencias brutas y visualización gráfica de la distribución de clases. | `matplotlib`, `seaborn` |
| **3. Preprocesamiento** | Limpieza de URLs y caracteres especiales mediante Regex. Tokenización y lematización avanzada. | `re`, `nltk`, `spaCy` |
| **4. Vectorización** | Transformación de texto a matrices de características numéricas ponderadas. | `scikit-learn` (`TfidfVectorizer`) |
| **5. Modelado Deep Learning** | Arquitectura densa (`nn.Module`) con Batch Normalization y Dropout (p=0.5). | `torch` (PyTorch) |
| **6. Optimización** | Entrenamiento con optimizador Adam, función de pérdida CrossEntropyLoss y Early Stopping (paciencia = 5). | `torch.optim` |

---

## 🛠️ Cómo Ejecutar el Proyecto

1. Clona este repositorio en tu entorno local o súbelo a tu cuenta personal de Google Drive.
2. Abre el archivo `notebook_evaluacion.ipynb` en **Google Colab**.
3. Asegúrate de activar el entorno de ejecución por hardware utilizando una **GPU** (`Entorno de ejecución` > `Cambiar tipo de entorno de ejecución` > `T4 GPU`).
4. Ejecuta las celdas secuencialmente. El script instalará las dependencias necesarias y descargará los diccionarios idiomáticos de `NLTK` y `spaCy` de forma automática.

## 📈 Conclusiones Basadas en Criterios de Evaluación
El modelo alcanza una convergencia óptima deteniéndose de manera temprana gracias al monitoreo de la pérdida de validación, previniendo el sobreajuste (*overfitting*). La inclusión de **Batch Normalization** estabilizó los gradientes en las primeras épocas y el **Dropout al 50%** forzó a la red neuronal densa a generalizar de forma robusta sobre los vectores semánticos generados por **TF-IDF**.
