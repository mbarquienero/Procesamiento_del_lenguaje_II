# Procesamiento del lenguaje natural II

Introducción

La materia Procesamiento del Lenguaje Natural II forma parte del plan del Master en Inteligencia Artificial (MIA) de la UBA y constituye la base teórica y práctica para comprender cómo las computadoras procesan, representan y generan lenguaje humano.

Durante el curso se abordan conceptos clave como:

* Representación vectorial del lenguaje
* Embeddings (estáticos y contextuales)
* Tokenización y chunking
* Preprocesamiento de texto
* Recuperación de información (IR)
* Similaridad semántica
* Introducción a modelos Transformer y embeddings modernos

Los trabajos prácticos permiten aplicar estos conceptos en desarrollos reales orientados al análisis y modelado de texto.

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

 TP2/
 ├── app.py                 # Aplicación principal en Streamlit. Orquesta el pipeline RAG y la interfaz del chatbot.
 ├── ingest_cv.py           # Script de ingestión del CV: extrae texto, genera embeddings y sube los vectores a Pinecone.
 ├── retriever_cv.py        # Módulo de recuperación semántica: busca los chunks más relevantes en el índice vectorial.
 ├── requirements.txt       # Listado de dependencias necesarias para ejecutar todo el TP1.
 ├── data/
 │   ├── cv_mauro.pdf       # CV del alumno utilizado como base documental para el sistema RAG.
 └── README.md              # Documentación detallada del TP1 con instrucciones, arquitectura y descripción del proyecto.

