# Procesamiento del lenguaje natural II

Introducción

La materia Procesamiento del Lenguaje Natural II forma parte del plan del Master en Inteligencia Artificial (MIA) de la UBA y constituye la base teórica y práctica para comprender cómo las computadoras procesan, representan y generan lenguaje humano.

---

El Trabajo Práctico 1 de la materia Procesamiento del Lenguaje Natural II tiene como objetivo introducir al alumno en los fundamentos prácticos del análisis del lenguaje natural utilizando técnicas clásicas y modelos de aprendizaje automático.

Este trabajo sienta las bases del PLN moderno, reforzando conceptos esenciales como:

  * procesamiento y limpieza de texto
  * tokenización
  * representación vectorial
  * embeddings tradicionales,
  * modelos de clasificación de texto,
  * entrenamiento supervisado,
  * evaluación de métricas,
  * análisis de resultados.

🎯 Objetivos del TP1

  * Comprender y aplicar técnicas de preprocesamiento de texto.
  * Analizar distintas representaciones vectoriales:
      * Bag of Words (BoW)
      * TF-IDF
      * Embeddings distribucionales

  * Entrenar modelos supervisados de clasificación.
  * Evaluar desempeño mediante métricas estándar (accuracy, f1-score, pérdida).
  * Explorar distintos hiperparámetros y observar su impacto.
  * Implementar ciclos de entrenamiento utilizando un trainer modular (según trainer.py).
  * Realizar análisis experimental a través de notebooks (según trabajo_practico_1.ipynb).

🛠️ Tecnologías y librerías utilizadas en el TP1

El trabajo práctico incorpora un conjunto de herramientas orientadas al PLN clásico y aprendizaje automático:

✔️ Procesamiento de texto

   * NLTK
   * spaCy
   * regex
   * Normalización y tokenización

✔️ Representación vectorial

   * scikit-learn (CountVectorizer, TF-IDF)
   * Embeddings básicos utilizados en modelos lineales o feed-forward

✔️ Modelado y entrenamiento

   * PyTorch — para modelos simples de clasificación
   * trainer.py — módulo propio para:
     * entrenamiento estructurado
     * early stopping
     * evaluación
     * métricas
     * manejo de batches y optimización

✔️ Experimentación

   * Jupyter Notebook (trabajo_practico_1.ipynb)
     * análisis exploratorio
     * experimentos
     * comparación de modelos
     * reflexiones finales

📄 Resultado del TP1

El resultado final es un pipeline completo que abarca:

 1. Lectura y procesamiento del corpus
 2 .Vectorización del texto mediante métodos clásicos
 3. Entrenamiento de un modelo de clasificación usando PyTorch
 4. Implementación de un “trainer” modular para facilitar experimentos
 5. Evaluación mediante métricas y análisis de desempeño

---

📝 Trabajo Práctico 2 — Chatbot con RAG (Retrieval-Augmented Generation)

El objetivo del TP2 es implementar un chatbot que utilice información externa almacenada en una base vectorial para responder preguntas, aplicando la arquitectura RAG (Retrieval-Augmented Generation).

El sistema debe ser capaz de:

 * Leer y procesar un documento PDF (en este caso, el CV del alumno).
 * Limpiar y segmentar el texto en fragmentos (chunking).
 * Generar embeddings para cada fragmento del CV.
 * Almacenar esos embeddings en una base vectorial.
 * Recuperar los fragmentos más relevantes ante una consulta.
 * Utilizar un modelo generativo para construir una respuesta final basada en el contexto recuperado.

✔️ Tecnologías y librerías utilizadas

  * Python 3.11
  * Streamlit — interfaz gráfica para el chatbot
  * PyPDF2 / pdfminer.six — extracción de texto desde PDF
  * Pinecone — base vectorial utilizada para indexación semántica
  * Sentence-Transformers / BGE / MPNet — modelos de embeddings
  * Groq (Llama 3) — modelo generativo para la respuesta final
  * dotenv — manejo de claves y variables de entorno
  * Similitud coseno / búsqueda k-NN — mecanismo de recuperación

✔️ Flujo general del TP1

1. Ingestión del CV

  * Lectura del PDF
  * Limpieza del texto
  * Segmentación en chunks
  * Generación de embeddings
  * Subida a Pinecone

2. Recuperación de información (Retriever)

  * Para cada pregunta del usuario
  * Se generan embeddings de la consulta
  * Se buscan los chunks más cercanos en la base vectorial

3. Generación de respuesta (RAG)

  * Se construye un contexto a partir de los chunks recuperados
  * Se envía el contexto + pregunta al modelo
  * El modelo genera una respuesta fundamentada

✔️ Resultado del TP2

El resultado final es un chatbot funcional que responde preguntas sobre el CV del alumno utilizando:

  * Recuperación semántica
  * Construcción de contexto
  * Generación aumentada con LLM
  * Interfaz lista para usar desde Streamlit
  * El sistema garantiza respuestas precisas, fundamentadas y basadas directamente en la información del documento original.


✔️ Directorio

<img width="990" height="161" alt="image" src="https://github.com/user-attachments/assets/250f50bf-8c20-42e2-bdb9-d99e63a24120" />


