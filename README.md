# DataScience-III
# Clasificador de Texto para Detección de Phishing con PyTorch y NLP

Este repositorio contiene la resolución práctica de la **Evaluación Final** para la asignatura *Modelos Avanzados de Aprendizaje Automático*. El proyecto consiste en el diseño e implementación de un pipeline completo de Procesamiento del Lenguaje Natural (NLP) y el entrenamiento de una red neuronal utilizando **PyTorch** para clasificar textos como *phishing* o *legítimos*.

## 🚀 Estructura del Repositorio

* `Entrega_Final_Data_science_III.ipynb`: Cuaderno de Google Colab con el flujo de código completo (EDA, preprocesamiento, vectorización y entrenamiento).
* `best_model.pth`: Pesos de la red neuronal, guardados automáticamente mediante la estrategia de *Early Stopping*.
* `requirements.txt`: Librerías necesarias para reproducir el entorno.

---

## 📊 Resumen del Pipeline Implementado

| Etapa | Descripción | Herramientas / Librerías |
| :--- | :--- | :--- |
| **1. Dataset & Auditoría** | Corpus de 2.500 mensajes etiquetados (*phishing* / *legítimo*) y verificación del mínimo de 2.000 registros. | `pandas` |
| **2. EDA** | Análisis de frecuencias brutas y visualización gráfica de la distribución de clases. | `matplotlib`, `seaborn` |
| **3. Preprocesamiento** | Limpieza de URLs y caracteres especiales mediante Regex. Tokenización y comparación de lematización NLTK vs. spaCy (se usa spaCy, que lematiza según el contexto gramatical). | `re`, `nltk`, `spaCy` |
| **4. Vectorización** | Transformación de texto a matrices de características numéricas ponderadas. | `scikit-learn` (`TfidfVectorizer`) |
| **5. Modelado Deep Learning** | Arquitectura densa (`nn.Module`) con Batch Normalization y Dropout (p=0.5). | `torch` (PyTorch) |
| **6. Optimización** | Entrenamiento con optimizador Adam, función de pérdida CrossEntropyLoss y Early Stopping (paciencia = 5). | `torch.optim` |

---

## 🛠️ Cómo Ejecutar el Proyecto

1. Abrí `Entrega_Final_Data_science_III.ipynb` en **Google Colab** (o en un entorno local con `pip install -r requirements.txt`).
2. Instalá el modelo de spaCy en inglés: `python -m spacy download en_core_web_sm`.
3. Ejecutá las celdas en orden. No hace falta GPU: el entrenamiento corre en segundos en CPU.

## 📈 Resultados y Conclusiones

El pipeline completo funciona de punta a punta: limpieza, lematización, TF-IDF, red neuronal y Early Stopping, que detiene el entrenamiento cuando la pérdida de validación deja de mejorar.

El modelo llega a **100% de exactitud** en validación desde las primeras épocas. Este resultado se explica por el dataset y no indica que el modelo generalice:

* **Limitación del dataset:** el corpus usado es simulado. Está formado por 5 mensajes de ejemplo repetidos 500 veces cada uno (por eso TF-IDF encuentra solo 39 palabras distintas), así que los textos de validación son idénticos a los de entrenamiento.
* **Próximo paso:** entrenar y evaluar con un corpus real de emails o SMS (por ejemplo, SpamAssassin, Enron-Spam o SMS Spam Collection), eliminando duplicados antes de dividir en entrenamiento y validación.
