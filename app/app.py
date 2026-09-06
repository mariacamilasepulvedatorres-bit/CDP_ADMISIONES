from pathlib import Path

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

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "ridge_optimized_pipeline.joblib"
)

model = load(MODEL_PATH)


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
        Ingrese la información del aspirante para estimar su
        probabilidad de admisión mediante el modelo de aprendizaje
        automático entrenado.
        </p>
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
        "GRE Score",
        min_value=0,
        max_value=400,
        value=320,
    )

    university_rating = st.number_input(
        "University Rating",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=1.0,
    )

    lor = st.number_input(
        "LOR",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.5,
    )

    research = st.selectbox(
        "Research",
        options=[False, True],
        format_func=lambda value: "Sí" if value else "No",
    )


with col2:
    toefl_score = st.number_input(
        "TOEFL Score",
        min_value=0,
        max_value=150,
        value=110,
    )

    sop = st.number_input(
        "SOP",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.5,
    )

    cgpa = st.number_input(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        value=8.5,
        step=0.1,
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

    prediction = model.predict(input_data)[0]

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


st.markdown(
    """
    <div class="footer-text">
        La educación transforma vidas 💜
    </div>
    """,
    unsafe_allow_html=True,
)