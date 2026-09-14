"""Aplicación Streamlit para predicción online de admisiones."""

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from joblib import load

# ---------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------
st.set_page_config(
    page_title="Predicción de admisión",
    page_icon="🎓",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "ridge_optimized_pipeline.joblib"


@st.cache_resource
def load_model() -> Any:
    """Carga y almacena en caché el modelo entrenado."""
    if not MODEL_PATH.exists():
        msg = f"No se encontró el modelo en: {MODEL_PATH}"
        raise FileNotFoundError(msg)

    return load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError as error:
    st.error(
        "No fue posible cargar el modelo entrenado. Verifique que el archivo del modelo exista."
    )
    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# ESTILOS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8f4ff 0%, #eee4ff 100%);
    }

    .block-container {
        max-width: 1050px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    h1 {
        color: #3f237c;
        font-weight: 800;
    }

    h3 {
        color: #6544a5;
    }

    p {
        color: #665a7c;
        font-size: 17px;
    }

    div[data-testid="stNumberInput"] input {
        background-color: #fbf9ff;
        border-radius: 10px;
    }

    div[data-testid="stSelectbox"] > div > div {
        background-color: #fbf9ff;
        border-radius: 10px;
    }

    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #9b6de3, #7440c7);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem;
        font-size: 18px;
        font-weight: 700;
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #8757d2, #6331b5);
        color: white;
        border: none;
    }

    .hero {
        background: rgba(255, 255, 255, 0.88);
        padding: 30px 35px;
        border-radius: 22px;
        margin-bottom: 25px;
        box-shadow: 0 8px 30px rgba(93, 61, 145, 0.08);
    }

    .hero-label {
        color: #8b69c6;
        font-weight: 700;
        letter-spacing: 3px;
        font-size: 13px;
    }

    .result-card {
        background: linear-gradient(135deg, #f1e7ff, #faf7ff);
        border: 1px solid #e2d2fa;
        border-radius: 18px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
    }

    .result-title {
        color: #7152a6;
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 2px;
    }

    .result-number {
        color: #7541c8;
        font-size: 42px;
        font-weight: 800;
    }

    .instructions {
        background: rgba(255, 255, 255, 0.65);
        border-radius: 14px;
        padding: 15px 20px;
        margin-bottom: 25px;
    }

    .footer-text {
        text-align: center;
        color: #9275bf;
        font-style: italic;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# ENCABEZADO
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-label">ANÁLISIS DE DATOS PARA LA EDUCACIÓN</div>
        <h1>🎓 Predicción de probabilidad de admisión</h1>
        <p>
        Ingrese la información académica del aspirante para estimar
        su probabilidad de admisión mediante el modelo de aprendizaje
        automático entrenado.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="instructions">
        <strong>¿Cómo utilizar la aplicación?</strong><br>
        1. Ingrese la información académica del aspirante.<br>
        2. Indique si cuenta con experiencia en investigación.<br>
        3. Presione <strong>Realizar predicción</strong>.<br>
        4. Consulte la probabilidad estimada de admisión.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Información del aspirante")


# ---------------------------------------------------------
# FORMULARIO
# ---------------------------------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    gre_score = st.number_input(
        "Puntuación GRE",
        min_value=260,
        max_value=340,
        value=320,
        help="Puntuación del examen GRE entre 260 y 340.",
    )

    university_rating = st.number_input(
        "Clasificación universitaria",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=1.0,
        help="Clasificación de la universidad entre 1 y 5.",
    )

    lor = st.number_input(
        "LOR",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.5,
        help="Valoración de las cartas de recomendación entre 1 y 5.",
    )

    research = st.selectbox(
        "Experiencia en investigación",
        options=[False, True],
        format_func=lambda value: "Sí" if value else "No",
    )

with col2:
    toefl_score = st.number_input(
        "Puntuación TOEFL",
        min_value=0,
        max_value=120,
        value=110,
        help="Puntuación del examen TOEFL entre 0 y 120.",
    )

    sop = st.number_input(
        "SOP",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.5,
        help="Valoración de la declaración de propósito entre 1 y 5.",
    )

    cgpa = st.number_input(
        "Promedio general de calificaciones (CGPA)",
        min_value=0.0,
        max_value=10.0,
        value=8.5,
        step=0.1,
        help="Promedio académico acumulado entre 0 y 10.",
    )


# ---------------------------------------------------------
# PREDICCIÓN
# ---------------------------------------------------------
st.write("")

if st.button("✨ Realizar predicción", use_container_width=True):
    input_data = pd.DataFrame(
        [
            {
                "GRE Score": gre_score,
                "TOEFL Score": toefl_score,
                "University Rating": university_rating,
                "SOP": sop,
                "LOR": lor,
                "CGPA": cgpa,
                "Research": research,
            }
        ]
    )

    input_data["Research"] = input_data["Research"].astype("object")

    try:
        prediction = float(model.predict(input_data)[0])

        # La variable objetivo del modelo representa una probabilidad.
        prediction = max(0.0, min(1.0, prediction))

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">RESULTADO</div>
                <div>Probabilidad estimada de admisión</div>
                <div class="result-number">{prediction:.2%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except (TypeError, ValueError) as error:
        st.error("No fue posible generar la predicción.")
        st.exception(error)


# ---------------------------------------------------------
# PIE DE PÁGINA
# ---------------------------------------------------------
st.markdown(
    """
    <div class="footer-text">
        Modelo de predicción de admisiones · Ciencia de Datos en Producción 🎓
    </div>
    """,
    unsafe_allow_html=True,
)
