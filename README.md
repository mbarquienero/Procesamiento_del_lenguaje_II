# Procesamiento del lenguaje natural II

🧠 Introducción

La materia Procesamiento del Lenguaje Natural II forma parte del plan de estudios del Master en Inteligencia Artificial (MIA) de la UBA.
Su propósito es introducir los fundamentos teóricos y prácticos del análisis y modelado del lenguaje natural mediante técnicas modernas de extracción, representación y procesamiento de texto.

A lo largo del curso se abordan temas como:

Representación vectorial del lenguaje

Embeddings (estáticos y contextuales)

Recuperación de información (IR)

Similaridad semántica

Preprocesamiento de texto

Tokenización y chunking

Introducción a modelos tipo Transformer y sus embeddings

Los trabajos prácticos permiten llevar estos conceptos a la práctica mediante la implementación de sistemas reales basados en NLP.

📝 Trabajo Práctico 1 — Chatbot con RAG (Retrieval-Augmented Generation)

El TP1 consiste en implementar un chatbot capaz de generar respuestas utilizando la técnica de Retrieval-Augmented Generation (RAG).
El objetivo central es que el modelo no dependa únicamente de su conocimiento interno, sino que pueda recuperar información desde una base de documentos vectorizados (en este caso, el CV del alumno) y generar respuestas fundamentadas.

✔️ Objetivos del TP1

Procesar un documento PDF (CV del alumno).

Extraer el texto, limpiarlo y segmentarlo correctamente (chunking).

Generar embeddings para cada fragmento del CV.

Indexar los embeddings en una base vectorial.

Implementar un sistema de recuperación semántica (retriever).

Integrar el contexto recuperado con un modelo generativo vía RAG.

Construir una interfaz conversacional usando Streamlit.

✔️ Tecnologías y librerías utilizadas

Python 3.11

Streamlit – interfaz web interactiva

PyPDF2 / pdfminer.six – extracción de texto desde PDF

Pinecone – base vectorial para almacenamiento de embeddings

Sentence-Transformers / bge-small-en / all-mpnet-base-v2 – embeddings semánticos

LLM vía API (Groq / Llama 3) – generación de respuestas

Similitud coseno / búsqueda KNN

dotenv – manejo de claves y configuración

✔️ Resultado final

Un chatbot funcional capaz de responder preguntas sobre el CV del alumno mediante:

Recuperación semántica (Retriever)

Construcción de contexto

Generación aumentada (RAG)
