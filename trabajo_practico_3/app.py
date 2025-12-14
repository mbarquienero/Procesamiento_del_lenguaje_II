import os
from typing import Dict, List

from dotenv import load_dotenv
import streamlit as st

from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from retriever_cv import retrieve_cv_context
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

# Índices de Pinecone por persona
AGENTES_CONFIG: Dict[str, Dict] = {
    "mauro": {
        "nombre_mostrado": "Mauro Barquinero",
        "index_name": os.getenv("PINECONE_INDEX_MAURO"),
        "alias": ["mauro", "mauro barquinero"],
        "es_alumno": True,
    },
    "ana": {
        "nombre_mostrado": "Ana Sofía López",
        "index_name": os.getenv("PINECONE_INDEX_ANA"),
        "alias": ["ana", "ana sofía lópez"],
        "es_alumno": False,
    },
    "juan": {
        "nombre_mostrado": "Juan Pablo González",
        "index_name": os.getenv("PINECONE_INDEX_JUAN"),
        "alias": ["juan", "juan pablo gonzález", "juan pablo"],
        "es_alumno": False,
    },
    "pedro": {
        "nombre_mostrado": "Pedro Luis Ramírez",
        "index_name": os.getenv("PINECONE_INDEX_PEDRO"),
        "alias": ["pedro", "pedro ramirez", "pedro luis", "pedro luis ramírez"],
        "es_alumno": False,
    }
}

# Filtramos agentes que tengan índice configurado.
AGENTES_CONFIG = {
    k: v for k, v in AGENTES_CONFIG.items() if v.get("index_name")
}

if not AGENTES_CONFIG:
    raise RuntimeError(
        "No hay agentes configurados con índice de Pinecone.\n"
        "Configura al menos PINECONE_INDEX_MAURO en tu .env."
    )


# ==================================
# Construcción de la LLMChain (RAG)
# ==================================

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
            "Eres un asistente especializado en responder preguntas sobre perfiles "
            "profesionales (CVs).\n\n"
            "Tu única fuente de verdad es el siguiente contexto extraído del CV "
            "de esta persona. Úsalo para responder de la forma más completa, "
            "natural y realista posible sobre su profesión, experiencia, "
            "estudios, habilidades y proyectos.\n\n"
            "Reglas importantes:\n"
            "1) No inventes nombres de empresas, puestos, certificaciones ni rangos "
            "de fechas que no aparezcan explícitamente en el contexto.\n"
            "2) Puedes inferir años que caen dentro de un rango explícitamente "
            "mencionado.\n"
            "3) Puedes identificar la primera experiencia laboral del CV tomando "
            "la fecha más antigua que aparezca en el contexto, incluso si esa "
            "experiencia aparece fragmentada en varios períodos.\n"
            "4) Si la información es incompleta o contradictoria, indícalo "
            "explícitamente.\n"
            "5) Si hay varias experiencias relevantes para la pregunta, "
            "menciónalas todas.\n"
            "6) Si la pregunta requiere comparar a esta persona con otra y no "
            "existe información de la otra persona en el contexto, debes:\n"
            "   a) Indicar claramente que la comparación no es posible.\n"
            "   b) Explicar brevemente qué información falta.\n"
            "   c) Responder SIEMPRE con la información disponible de esta persona, "
            "      describiendo su experiencia profesional, aunque la comparación "
            "      no sea posible.\n"
            "   d) Mantener la misma estructura, nivel de detalle y estilo de "
            "      respuesta independientemente de la persona cuyo CV se esté "
            "      utilizando como contexto.\n"
            "7) Si no hay información suficiente para responder una pregunta, "
            "indícalo explícitamente y no intentes completar la respuesta con "
            "suposiciones.\n\n"
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
# Clase AgenteCV
# ==========================

class AgenteCV:
    """
    Representa a un agente asociado al CV de una persona.
    Cada agente tiene:
      - Un índice de Pinecone propio.
      - Una LLMChain con memoria de conversación.
    """

    def __init__(
        self,
        agente_id: str,
        nombre_mostrado: str,
        index_name: str,
        chain: LLMChain,
        system_prompt: str,
    ):
        """
        Constructor del agente.

        :param agente_id: Identificador interno del agente.
        :param nombre_mostrado: Nombre que se mostrará en la interfaz.
        :param index_name: Nombre del índice de Pinecone asociado al CV de la persona.
        :param chain: Cadena de LangChain que maneja el diálogo.
        :param system_prompt: Prompt de sistema específico de la persona.
        """
        self.agente_id = agente_id
        self.nombre_mostrado = nombre_mostrado
        self.index_name = index_name
        self.chain = chain
        self.system_prompt = system_prompt

    def responder(self, pregunta: str, top_k: int = 8) -> dict:
        """
        Ejecuta el flujo para el agente:
          1) Recupera contexto del CV correspondiente.
          2) Llama al modelo de lenguaje con dicho contexto.
          3) Devuelve la respuesta y el contexto utilizado.
        """
        contexto = retrieve_cv_context(
            consulta=pregunta,
            top_k=top_k,
            index_name=self.index_name,
        )

        respuesta = self.chain.predict(
            system_prompt=self.system_prompt,
            contexto=contexto,
            pregunta_usuario=pregunta,
        )

        return {
            "respuesta": respuesta,
            "contexto": contexto,
        }

# ==========================
# Router de agentes
# ==========================

def detectar_personas_en_pregunta(
    pregunta: str,
    agentes_config: Dict[str, Dict],
) -> List[str]:
    """
    Detecta qué personas se mencionan en la pregunta en base a los alias
    definidos para cada agente.

    :param pregunta: Texto de la consulta del usuario.
    :param agentes_config: Diccionario de configuración de agentes.
    :return: Lista de ids de agentes a activar.
    """
    pregunta_lower = pregunta.lower()

    agentes_detectados: List[str] = []
    for agente_id, cfg in agentes_config.items():
        alias = cfg.get("alias", [])
        for nombre in alias:
            if nombre.lower() in pregunta_lower:
                agentes_detectados.append(agente_id)
                break

    if agentes_detectados:
        return agentes_detectados

    # Si no se detectó a nadie, usamos el agente del alumno
    for agente_id, cfg in agentes_config.items():
        if cfg.get("es_alumno"):
            return [agente_id]

    return [list(agentes_config.keys())[0]]


# ==========================
# Aplicación Streamlit
# ==========================

def inicializar_agentes(
    model_name: str,
    mem_length: int,
    system_prompt_base: str,
) -> Dict[str, AgenteCV]:
    """
    Crea un AgenteCV por persona configurada en AGENTES_CONFIG.
    """
    agentes: Dict[str, AgenteCV] = {}

    for agente_id, cfg in AGENTES_CONFIG.items():
        nombre = cfg["nombre_mostrado"]
        index_name = cfg["index_name"]

        # Prompt específico por persona
        system_prompt_persona = (
            f"{system_prompt_base}\n\n"
            f"Estás respondiendo exclusivamente sobre el CV de {nombre}. "
            f"Responde como si fueras {nombre} hablando en primera persona "
            f"cuando tenga sentido (por ejemplo, 'yo tengo experiencia en...')."
        )

        chain = construir_cadena_llm(
            model_name=model_name,
            system_prompt=system_prompt_persona,
            mem_length=mem_length,
        )

        agentes[agente_id] = AgenteCV(
            agente_id=agente_id,
            nombre_mostrado=nombre,
            index_name=index_name,
            chain=chain,
            system_prompt=system_prompt_persona,
        )

    return agentes


def main():
    st.set_page_config(page_title="Sistema de agentes sobre CVs", page_icon="🤖")

    st.title("Sistema de agentes sobre CVs del equipo")

    st.markdown(
        """
        Este sistema utiliza **un agente por persona** para responder preguntas
        sobre los CVs de los integrantes del equipo.

        Flujo general:
        1. El sistema detecta de qué persona(s) se está hablando en la consulta.
        2. Activa el/los agente(s) correspondientes (cada uno con su CV).
        3. Si no se menciona a nadie, se usa por defecto el **agente del alumno**.
        4. Si se pregunta por más de un CV, se consultan varios agentes y se
           muestran las respuestas de cada uno.
        """
    )

    # ----- Sidebar: configuración -----
    with st.sidebar:
        st.header("Configuración")

        default_system_prompt = (
            "Eres un asistente experto en recursos humanos que responde "
            "preguntas sobre la experiencia laboral, educación, habilidades "
            "y proyectos de la persona basándote únicamente en su CV."
        )

        system_prompt_base = st.text_area(
            "Prompt base del sistema (se personaliza por agente):",
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
            "Longitud de memoria por agente (turnos recordados):",
            min_value=1,
            max_value=20,
            value=5,
        )

        mostrar_contexto = st.checkbox(
            "Mostrar el contexto recuperado del CV (debug por agente)",
            value=False,
        )

        if st.button("Reiniciar conversación"):
            st.session_state.pop("agentes", None)
            st.session_state.pop("config_chain", None)
            st.session_state.pop("mensajes_pantalla", None)
            st.success("Conversación reiniciada.")

    config_actual = {
        "model_name": model_name,
        "system_prompt_base": system_prompt_base,
        "mem_length": mem_length,
    }

    if "agentes" not in st.session_state or "config_chain" not in st.session_state:
        st.session_state.agentes = inicializar_agentes(
            model_name=model_name,
            mem_length=mem_length,
            system_prompt_base=system_prompt_base,
        )
        st.session_state.config_chain = config_actual
        st.session_state.mensajes_pantalla = []
    else:
        if st.session_state.config_chain != config_actual:
            st.session_state.agentes = inicializar_agentes(
                model_name=model_name,
                mem_length=mem_length,
                system_prompt_base=system_prompt_base,
            )
            st.session_state.config_chain = config_actual
            st.session_state.mensajes_pantalla = []

    # ----- Input del usuario -----

    st.subheader("Conversación")

    pregunta_usuario = st.text_input(
        "Escribe tu pregunta sobre los CVs (puedes nombrar a una o varias personas):",
        value="",
        key="input_pregunta",
    )

    if st.button("Enviar") and pregunta_usuario.strip():
        # 1) Detectar qué agentes se deben activar
        ids_agentes = detectar_personas_en_pregunta(
            pregunta=pregunta_usuario,
            agentes_config=AGENTES_CONFIG,
        )

        nombres_agentes = [
            st.session_state.agentes[a].nombre_mostrado for a in ids_agentes
        ]

        st.info(
            "Agentes activados para esta consulta: "
            + ", ".join(nombres_agentes)
        )

        respuestas_por_agente = []

        # 2) Ejecutar cada agente de forma independiente
        for agente_id in ids_agentes:
            agente = st.session_state.agentes[agente_id]

            with st.spinner(
                f"Buscando información en el CV de {agente.nombre_mostrado}..."
            ):
                resultado = agente.responder(pregunta=pregunta_usuario, top_k=8)

            respuestas_por_agente.append(
                {
                    "nombre": agente.nombre_mostrado,
                    "respuesta": resultado["respuesta"],
                    "contexto": resultado["contexto"],
                }
            )

        # 3) Armar salida visible (una sección por agente)
        texto_respuesta_final = ""
        for r in respuestas_por_agente:
            texto_respuesta_final += (
                f"### Respuesta de {r['nombre']}\n\n{r['respuesta']}\n\n"
            )

        # Guardar en historial para mostrar en pantalla
        st.session_state.mensajes_pantalla.append(
            {"role": "user", "content": pregunta_usuario}
        )
        st.session_state.mensajes_pantalla.append(
            {"role": "assistant", "content": texto_respuesta_final}
        )

        # Mostrar contexto opcional (debug) por agente
        if mostrar_contexto:
            for r in respuestas_por_agente:
                with st.expander(f"📄 Contexto de {r['nombre']}"):
                    st.write(r["contexto"])

    # ----- Mostrar historial de chat -----

    if "mensajes_pantalla" in st.session_state:
        for msg in st.session_state.mensajes_pantalla:
            if msg["role"] == "user":
                st.markdown(f"🧑 **Tú:** {msg['content']}")
            else:
                st.markdown(f"🤖 **Bot:**\n\n{msg['content']}")


if __name__ == "__main__":
    main()