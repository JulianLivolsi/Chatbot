import streamlit as st
import pandas as pd
from io import StringIO
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.llms import Ollama

# --- CARGA DE DATOS ---
@st.cache_data
def cargar_datos():
    df_clientes = pd.read_csv("clientes.csv")
    df_ventas = pd.read_csv("ventas.csv")
    return df_clientes, df_ventas

clientes, ventas = cargar_datos()

# --- CONTEXTO PARA LOS AGENTES ---
contexto_base = """
Sos un asistente virtual exclusivo para la empresa Card SA. Solamente podés responder preguntas basándote en dos conjuntos de datos CSV:

clientes.csv con columnas:
- ClienteID
- Nombre
- Email
- País
- FechaRegistro

ventas.csv con columnas:
- VentaID
- ClienteID
- FechaVenta
- Monto
- Producto

NO respondas con información general o externa. Solo usá estos datos y responde en consecuencia.

Si la pregunta no puede ser respondida con esta información, indicá que no tenés datos para responder.
"""


# --- CREAR AGENTES CON CONTEXTO ---
llm = Ollama(model="mistral")

agente_clientes = create_pandas_dataframe_agent(
    llm,
    clientes,
    verbose=False,
    allow_dangerous_code=True,
    handle_parsing_errors=True,
    prefix=contexto_base
)

agente_ventas = create_pandas_dataframe_agent(
    llm,
    ventas,
    verbose=False,
    allow_dangerous_code=True,
    handle_parsing_errors=True,
    prefix=contexto_base
)

# --- FUNCION GENERAL PARA DETERMINAR SI USAR DATOS ---
def necesita_datos(prompt):
    palabras_clave_directas = ["cuántos", "filtrar", "listar", "total", "sumar", "exportar", "generar", "descargar"]
    entidades = ["cliente", "clientes", "venta", "ventas", "producto", "monto", "país", "csv", "excel"]
    return any(p in prompt.lower() for p in palabras_clave_directas) and any(e in prompt.lower() for e in entidades)

# --- FUNCION PARA DETECTAR SI SE SOLICITA UN ARCHIVO ---
def solicitar_archivo(prompt):
    return "descargar" in prompt.lower() or "archivo" in prompt.lower() or "csv" in prompt.lower() or "excel" in prompt.lower()

# --- INTERFAZ STREAMLIT ---
st.title("🤖 Asistente de Card SA")
st.markdown("Escribí tus preguntas sobre clientes, ventas o lo que necesites. También podés pedir archivos.")

# --- REINICIAR CHAT ---
if st.button("🔄 Reiniciar chat"):
    st.session_state.clear()
    st.rerun()

if "historial" not in st.session_state or len(st.session_state.historial) == 0:
    mensaje_inicial = "¡Hola! Soy el asistente de Card SA. Podés preguntarme sobre ventas, clientes o cualquier otra cosa. ¿En qué puedo ayudarte?"
    st.session_state.historial = [{"role": "assistant", "content": mensaje_inicial}]

for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])

prompt = st.chat_input("¿En qué puedo ayudarte hoy?")

if prompt:
    st.session_state.historial.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        respuesta_placeholder = st.empty()
        respuesta_generada = ""

        try:
            if necesita_datos(prompt):
                if "cliente" in prompt.lower():
                    resultado = agente_clientes.run(prompt)
                elif "venta" in prompt.lower() or "producto" in prompt.lower() or "monto" in prompt.lower():
                    resultado = agente_ventas.run(prompt)
                else:
                    resultado = "Puedo ayudarte con clientes o ventas, pero no entendí bien tu pedido."
                respuesta_generada = resultado
                respuesta_placeholder.markdown(respuesta_generada)

                # --- OPCION DE DESCARGA ---
                if solicitar_archivo(prompt):
                    if "cliente" in prompt.lower():
                        df_filtrado = clientes
                    else:
                        df_filtrado = ventas

                    csv_buffer = StringIO()
                    df_filtrado.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Descargar CSV",
                        data=csv_buffer.getvalue(),
                        file_name="resultado.csv",
                        mime="text/csv"
                    )
            else:
                for chunk in llm.stream(prompt):
                    respuesta_generada += chunk
                    respuesta_placeholder.markdown(respuesta_generada)
        except Exception as e:
            respuesta_generada = f"Ocurrió un error al procesar tu pregunta: {e}"
            respuesta_placeholder.markdown(respuesta_generada)

        st.session_state.historial.append({"role": "assistant", "content": respuesta_generada})
