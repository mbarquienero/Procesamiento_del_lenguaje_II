# retriever_cv.py

import os
from typing import List, Dict, Optional

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# Índice por defecto (compatibilidad hacia atrás)
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "cv-mauro-index")

_pc: Pinecone | None = None
_indices_cache: dict[str, any] = {}
_embedding_model: SentenceTransformer | None = None


# ==========================
# Helpers de inicialización
# ==========================

def _get_pinecone_client() -> Pinecone:
    """
    Devuelve una instancia única de cliente Pinecone para toda la aplicación.
    """
    global _pc
    if _pc is None:
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY no está configurada en .env")
        _pc = Pinecone(api_key=PINECONE_API_KEY)
    return _pc


def _get_index(index_name: Optional[str] = None):
    """
    Devuelve una instancia de índice de Pinecone.

    Si no se indica nombre, se utiliza el índice por defecto definido en la
    variable de entorno PINECONE_INDEX_NAME.
    """
    nombre = index_name or INDEX_NAME

    if not nombre:
        raise RuntimeError(
            "No se indicó nombre de índice y PINECONE_INDEX_NAME está vacío."
        )

    global _indices_cache
    if nombre not in _indices_cache:
        pc = _get_pinecone_client()
        _indices_cache[nombre] = pc.Index(nombre)

    return _indices_cache[nombre]


def _get_embedding_model() -> SentenceTransformer:
    """
    Devuelve un modelo de embeddings de frase cargado una sola vez.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embedding_model


# ==========================
# Funciones de RAG sobre CVs
# ==========================

def buscar_en_cv(
    consulta: str,
    top_k: int = 5,
    index_name: Optional[str] = None,
) -> List[Dict]:
    """
    Dada una consulta en lenguaje natural, devuelve los top_k chunks
    más relevantes del CV almacenados en el índice indicado.

    :param consulta: Pregunta o texto a buscar.
    :param top_k: Cantidad de fragmentos de CV a recuperar.
    :param index_name: Nombre del índice de Pinecone a utilizar. Si es None,
                       se usa el índice por defecto.
    """
    modelo = _get_embedding_model()
    indice = _get_index(index_name=index_name)

    # 1) Embedding de la consulta
    embedding_consulta = modelo.encode([consulta])[0].tolist()

    # 2) Query al índice
    resultado = indice.query(
        vector=embedding_consulta,
        top_k=top_k,
        include_metadata=True,
    )

    documentos: List[Dict] = []
    for match in resultado["matches"]:
        meta = match.get("metadata", {}) or {}
        documentos.append(
            {
                "id": match["id"],
                "score": match["score"],
                "texto": meta.get("texto", ""),
                "chunk_index": meta.get("chunk", -1),
                "tipo": meta.get("tipo", "desconocido"),
            }
        )

    return documentos


def construir_contexto(documentos: List[Dict]) -> str:
    """
    Construye un string con los textos recuperados, listo para ser
    inyectado como 'contexto' en el prompt del LLM.
    """
    if not documentos:
        return "No se encontraron fragmentos relevantes del CV."

    partes = []
    for doc in documentos:
        partes.append(
            f"[Tipo: {doc['tipo']} | Score: {doc['score']:.3f} | "
            f"Chunk: {doc['chunk_index']}]\n"
            f"{doc['texto']}\n"
        )

    return "\n---\n".join(partes)


def retrieve_cv_context(
    consulta: str,
    top_k: int = 5,
    index_name: Optional[str] = None,
) -> str:
    """
    Función de alto nivel: recibe una consulta en lenguaje natural
    y devuelve un contexto textual armado a partir de los chunks
    del CV correspondiente al índice indicado.

    :param consulta: Pregunta o texto del usuario.
    :param top_k: Cantidad de fragmentos a recuperar.
    :param index_name: Nombre del índice de Pinecone del CV de la persona.
    """
    docs = buscar_en_cv(consulta, top_k=top_k, index_name=index_name)
    return construir_contexto(docs)


# ==========================
# Prueba rápida desde consola
# ==========================

if __name__ == "__main__":
    pregunta = "¿En qué trabaja actualmente esta persona?"
    print(f"CONSULTA: {pregunta}\n")

    documentos = buscar_en_cv(pregunta, top_k=3)
    print("DOCUMENTOS ENCONTRADOS:\n")
    for d in documentos:
        print(
            f"- ID: {d['id']} | Tipo: {d['tipo']} | "
            f"Score: {d['score']:.3f} | Chunk: {d['chunk_index']}"
        )
        print(d["texto"][:300], "...\n")

    print("===== CONTEXTO CONSTRUIDO =====\n")
    print(retrieve_cv_context(pregunta, top_k=3))
