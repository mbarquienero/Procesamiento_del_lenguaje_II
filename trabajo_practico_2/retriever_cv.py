# retriever_cv.py

import os
from typing import List, Dict

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "cv-mauro-index")

# Singletons simples para no recargar nada innecesariamente
_pc: Pinecone | None = None
_indice = None
_embedding_model: SentenceTransformer | None = None


# ==========================
# Helpers de inicialización
# ==========================

def _get_pinecone_client() -> Pinecone:
    global _pc
    if _pc is None:
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY no está configurada en .env")
        _pc = Pinecone(api_key=PINECONE_API_KEY)
    return _pc


def _get_index():
    global _indice
    if _indice is None:
        pc = _get_pinecone_client()
        _indice = pc.Index(INDEX_NAME)
    return _indice


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model


# ==========================
# Funciones de RAG sobre el CV
# ==========================

def buscar_en_cv(consulta: str, top_k: int = 5) -> List[Dict]:
    """
    Dada una consulta, devuelve los top_k chunks más relevantes del CV
    almacenados en el índice Pinecone.
    """
    modelo = _get_embedding_model()
    indice = _get_index()

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
            f"[Tipo: {doc['tipo']} | Score: {doc['score']:.3f} | Chunk: {doc['chunk_index']}]\n"
            f"{doc['texto']}\n"
        )

    return "\n---\n".join(partes)


def retrieve_cv_context(consulta: str, top_k: int = 5) -> str:
    """
    Función de alto nivel: recibe una consulta en lenguaje natural
    y devuelve un contexto textual armado a partir de los chunks del CV.
    """
    docs = buscar_en_cv(consulta, top_k=top_k)
    return construir_contexto(docs)


# ==========================
# Prueba rápida desde consola
# ==========================

if __name__ == "__main__":
    pregunta = "¿En qué trabaja Mauro actualmente?"
    print(f"CONSULTA: {pregunta}\n")

    documentos = buscar_en_cv(pregunta, top_k=3)
    print("DOCUMENTOS ENCONTRADOS:\n")
    for d in documentos:
        print(
            f"- ID: {d['id']} | Tipo: {d['tipo']} | Score: {d['score']:.3f} | Chunk: {d['chunk_index']}"
        )
        print(d["texto"][:300], "...\n")

    print("===== CONTEXTO CONSTRUIDO =====\n")
    print(retrieve_cv_context(pregunta, top_k=3))
