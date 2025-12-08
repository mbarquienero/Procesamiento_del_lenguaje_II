# ingest_cv.py — indexa el CV (experiencia, certificaciones, educación, etc.)

import os
from typing import List

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ==========================
# 1. Variables de entorno
# ==========================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "cv-mauro-index")

# data/cv_mauro.pdf
CV_PATH = os.path.join("data", "cv_mauro.pdf")


# ==========================
# 2. Lectura del CV
# ==========================

def leer_cv_pdf(path: str) -> str:
    """Lee un PDF y devuelve todo el texto concatenado."""
    reader = PdfReader(path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text)


def leer_cv_txt(path: str) -> str:
    """Lee un .txt y devuelve todo el texto concatenado."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def cargar_texto_cv(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return leer_cv_pdf(path)
    elif ext in [".txt", ".md"]:
        return leer_cv_txt(path)
    else:
        raise ValueError(f"Extensión no soportada para el CV: {ext}")


# ==========================
# 3. Chunking genérico
# ==========================

def chunkear_texto(texto: str, max_chars: int = 300, solapamiento: int = 50) -> List[str]:
    """
    Corta el texto completo del CV en chunks más pequeños.
    """
    texto = " ".join(texto.split())

    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + max_chars
        chunk = texto[inicio:fin]
        chunks.append(chunk.strip())
        inicio = fin - solapamiento

    return chunks

# ==========================
# 4. Generador de embeddings
# ==========================

class GeneradorEmbeddings:
    def __init__(self, modelo: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.modelo = SentenceTransformer(modelo)
        self.dimension = self.modelo.get_sentence_embedding_dimension()

    def generar_embeddings_lote(self, textos: List[str]):
        return self.modelo.encode(textos, show_progress_bar=True).tolist()


# =============
# 5. Pinecone
# =============
def configurar_pinecone() -> Pinecone:
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY no configurada en .env")
    return Pinecone(api_key=PINECONE_API_KEY)


def crear_o_recuperar_indice(pc: Pinecone, nombre_indice: str, dimension: int):
    indices_existentes = [idx["name"] for idx in pc.list_indexes()]

    if nombre_indice not in indices_existentes:
        print(f"Creando índice '{nombre_indice}'...")
        pc.create_index(
            name=nombre_indice,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )
    else:
        print(f"✔ Índice '{nombre_indice}' ya existe.")

    return pc.Index(nombre_indice)


# ==========================
# 6. Pipeline de ingesta
# ==========================

def indexar_cv():
    print("Leyendo CV:", CV_PATH)
    texto_cv = cargar_texto_cv(CV_PATH)

    print(f"Longitud total del texto del CV: {len(texto_cv)} caracteres")

    # 1) Chunking del CV
    print("Generando chunks del CV completo...")
    textos_a_indexar = chunkear_texto(texto_cv, max_chars=300, solapamiento=50)
    print(f"   Chunks generados: {len(textos_a_indexar)}")

    # 2) Embeddings
    print("Cargando modelo de embeddings...")
    generador = GeneradorEmbeddings()

    # 3) Pinecone
    print("Configurando Pinecone...")
    pc = configurar_pinecone()
    indice = crear_o_recuperar_indice(pc, INDEX_NAME, generador.dimension)

    # 4) Generar embeddings
    print("Generando embeddings...")
    embeddings = generador.generar_embeddings_lote(textos_a_indexar)

    # 5) Armar vectores
    vectores = []
    for i, (texto, emb) in enumerate(zip(textos_a_indexar, embeddings)):
        vectores.append(
            {
                "id": f"cv_completo_{i}",
                "values": emb,
                "metadata": {
                    "texto": texto,
                    "chunk": i,
                    "tipo": "cv_completo",
                },
            }
        )

    # 6) Upsert
    print("🚀 Subiendo vectores a Pinecone...")
    indice.upsert(vectors=vectores)

    print("✅ Indexación completa.")


if __name__ == "__main__":
    indexar_cv()
