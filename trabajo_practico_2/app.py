import os

from dotenv import load_dotenv
import streamlit as st

from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_groq import ChatGroq

from retriever_cv import retrieve_cv_context

# ==========================
# Cargar variables de entorno
# ==========================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY no está configurada en el archivo .env.\n"
        "Agrega una línea GROQ_API_KEY=tu_api_key en .env."
    )


# ==========================
# Construcción de la LLMChain (RAG)
# ==========================

def construir_cadena_llm(
    model_name: str,
    system_prompt: str,
    mem_length: int = 5,
) -> LLMChain:
    """
    Construye una LLMChain de LangChain con:
      - LLM de Groq (ChatGroq)
      - Prompt de sistema + contexto del CV
      - Memoria de conversación (últimos k mensajes)
    """

    llm = ChatGroq(
        model_name=model_name,
        groq_api_key=GROQ_API_KEY,
        temperature=0.7,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "{system_prompt}\n\n"
                "Tu única fuente de verdad es el siguiente contexto extraído del CV "
                "del alumno. Úsalo para responder de la forma más completa, natural y "
                "realista posible sobre su profesión, experiencia, estudios, habilidades "
                "y proyectos.\n\n"
                "Reglas importantes:\n"
                "1) No inventes nombres de empresas, puestos ni rangos de fechas que no aparezcan en el contexto.\n"
                "2) Puedes inferir años que caen dentro de un rango explícitamente mencionado.\n"
                "3) Puedes identificar la primera experiencia laboral del CV tomando la fecha más antigua "
                "que aparezca en el contexto, incluso si esa experiencia aparece fragmentada en varios períodos.\n"
                "4) Si la información es incompleta o contradictoria, dilo explícitamente.\n"
                "5) Si hay varias experiencias relevantes para la pregunta, menciónalas todas.\n\n"
                "Contexto del CV:\n{contexto}\n"
            ),

            MessagesPlaceholder(variable_name="historial_chat"),
            HumanMessagePromptTemplate.from_template("{pregunta_usuario}"),
        ]
    )

    memory = ConversationBufferWindowMemory(
        k=mem_length,
        memory_key="historial_chat",
        input_key="pregunta_usuario",
        return_messages=True,
    )

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=False,
    )
    return chain


# ==========================
# Aplicación Streamlit
# ==========================

def main():
    st.set_page_config(page_title="Chatbot RAG sobre CV", page_icon="💼")

    st.title("💼 Chatbot RAG sobre el CV del alumno")

    st.markdown(
        """
        Este chatbot utiliza **RAG (Retrieval-Augmented Generation)** para
        responder preguntas **únicamente** sobre el CV del alumno.

        1. La pregunta del usuario se convierte en un embedding.
        2. Se buscan los fragmentos más relevantes del CV en Pinecone.
        3. Esos fragmentos se usan como **contexto** para el modelo de Groq.
        """
    )

    # ----- Sidebar: configuración -----
    with st.sidebar:
        st.header("⚙️ Configuración")

        default_system_prompt = (
            "Eres un asistente experto en recursos humanos que responde "
            "preguntas sobre la experiencia laboral, educación, habilidades "
            "y proyectos del alumno basándote únicamente en su CV."
        )

        system_prompt = st.text_area(
            "Prompt del sistema:",
            value=default_system_prompt,
            height=160,
        )

        model_name = st.selectbox(
            "Modelo de Groq:",
            options=[
                "llama-3.1-8b-instant",
                "llama-3.1-70b-versatile",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
            ],
            index=0,
        )

        mem_length = st.slider(
            "Longitud de memoria (turnos recordados):",
            min_value=1,
            max_value=20,
            value=5,
        )

        mostrar_contexto = st.checkbox(
            "Mostrar el contexto recuperado del CV (debug)",
            value=False,
        )

        if st.button("🔄 Reiniciar conversación"):
            st.session_state.pop("chain", None)
            st.session_state.pop("config_chain", None)
            st.session_state.pop("mensajes_pantalla", None)
            st.success("Conversación reiniciada.")

    # ----- Inicializar o reconstruir la chain según la config -----

    config_actual = {
        "model_name": model_name,
        "system_prompt": system_prompt,
        "mem_length": mem_length,
    }

    if "chain" not in st.session_state or "config_chain" not in st.session_state:
        st.session_state.chain = construir_cadena_llm(
            model_name=model_name,
            system_prompt=system_prompt,
            mem_length=mem_length,
        )
        st.session_state.config_chain = config_actual
        st.session_state.mensajes_pantalla = []
    else:
        if st.session_state.config_chain != config_actual:
            st.session_state.chain = construir_cadena_llm(
                model_name=model_name,
                system_prompt=system_prompt,
                mem_length=mem_length,
            )
            st.session_state.config_chain = config_actual
            st.session_state.mensajes_pantalla = []

    # ----- Input del usuario -----

    st.subheader("💬 Conversación")

    pregunta_usuario = st.text_input(
        "Escribe tu pregunta sobre el CV:",
        value="",
        key="input_pregunta",
    )

    if st.button("Enviar") and pregunta_usuario.strip():
        # 1) Recuperar contexto del CV (RAG)
        with st.spinner("Buscando información relevante en el CV..."):
            contexto = retrieve_cv_context(pregunta_usuario, top_k=8)

        if mostrar_contexto:
            with st.expander("📄 Contexto recuperado del CV"):
                st.write(contexto)

        # 2) Obtener respuesta del LLM usando el contexto
        with st.spinner("Generando respuesta con el modelo de Groq..."):
            respuesta = st.session_state.chain.predict(
                system_prompt=system_prompt,
                contexto=contexto,
                pregunta_usuario=pregunta_usuario,
            )

        # 3) Guardar en el historial visible
        st.session_state.mensajes_pantalla.append(
            {"role": "user", "content": pregunta_usuario}
        )
        st.session_state.mensajes_pantalla.append(
            {"role": "assistant", "content": respuesta}
        )

    # ----- Mostrar historial de chat -----

    if "mensajes_pantalla" in st.session_state:
        for msg in st.session_state.mensajes_pantalla:
            if msg["role"] == "user":
                st.markdown(f"🧑 **Tú:** {msg['content']}")
            else:
                st.markdown(f"🤖 **Bot:** {msg['content']}")


if __name__ == "__main__":
    main()