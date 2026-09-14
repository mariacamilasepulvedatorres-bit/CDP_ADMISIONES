"""Aplicación Streamlit para predicción online y batch de admisiones."""

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

FEATURE_COLUMNS = [
    "GRE Score",
    "TOEFL Score",
    "University Rating",
    "SOP",
    "LOR",
    "CGPA",
    "Research",
]

PREDICTION_COLUMN = "Predicted Chance of Admit"


@st.cache_resource
def load_model() -> Any:
    """Carga y almacena en caché el modelo entrenado."""
    if not MODEL_PATH.exists():
        msg = f"No se encontró el modelo en: {MODEL_PATH}"
        raise FileNotFoundError(msg)

    return load(MODEL_PATH)


def validate_batch_data(data: pd.DataFrame) -> list[str]:
    """Valida las columnas requeridas para realizar inferencia batch."""
    return [column for column in FEATURE_COLUMNS if column not in data.columns]


def prepare_batch_data(data: pd.DataFrame) -> pd.DataFrame:
    """Prepara las variables requeridas por el modelo."""
    features = data[FEATURE_COLUMNS].copy()
    features["Research"] = features["Research"].astype("object")
    return features


def generate_batch_predictions(
    data: pd.DataFrame,
    prediction_model: Any,
) -> pd.DataFrame:
    """Genera predicciones para múltiples registros."""
    features = prepare_batch_data(data)
    predictions = prediction_model.predict(features)

    output = data.copy()
    output[PREDICTION_COLUMN] = predictions
    return output


def dataframe_to_csv(data: pd.DataFrame) -> bytes:
    """Convierte un DataFrame a CSV para descarga."""
    csv_content: str = data.to_csv(index=False)
    return csv_content.encode("utf-8")


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

    h2, h3 {
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
        Utilice el modelo de aprendizaje automático para realizar
        predicciones individuales o procesar múltiples aspirantes
        mediante un archivo CSV.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# PESTAÑAS
# ---------------------------------------------------------
online_tab, batch_tab = st.tabs(
    [
        "🎓 Predicción individual",
        "📂 Procesamiento batch",
    ]
)


# ---------------------------------------------------------
# PREDICCIÓN INDIVIDUAL
# ---------------------------------------------------------
with online_tab:
    st.markdown(
        """
        <div class="instructions">
            <strong>¿Cómo utilizar la predicción individual?</strong><br>
            1. Ingrese la información académica del aspirante.<br>
            2. Indique si cuenta con experiencia en investigación.<br>
            3. Presione <strong>Realizar predicción</strong>.<br>
            4. Consulte la probabilidad estimada de admisión.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Información del aspirante")

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
# PROCESAMIENTO BATCH
# ---------------------------------------------------------
with batch_tab:
    st.markdown(
        """
        <div class="instructions">
            <strong>¿Cómo utilizar el procesamiento batch?</strong><br>
            1. Prepare un archivo CSV con las siete variables requeridas.<br>
            2. Cargue el archivo utilizando el selector inferior.<br>
            3. Revise la vista previa de los datos.<br>
            4. Presione <strong>Generar predicciones</strong>.<br>
            5. Visualice y descargue los resultados en formato CSV.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📂 Cargar archivo de aspirantes")

    st.caption(
        "Columnas requeridas: GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA y Research."
    )

    example_data = pd.DataFrame(
        [
            {
                "GRE Score": 325,
                "TOEFL Score": 112,
                "University Rating": 4,
                "SOP": 4.0,
                "LOR": 4.5,
                "CGPA": 9.1,
                "Research": 1,
            },
            {
                "GRE Score": 310,
                "TOEFL Score": 105,
                "University Rating": 3,
                "SOP": 3.5,
                "LOR": 3.5,
                "CGPA": 8.4,
                "Research": 0,
            },
            {
                "GRE Score": 335,
                "TOEFL Score": 118,
                "University Rating": 5,
                "SOP": 4.5,
                "LOR": 4.5,
                "CGPA": 9.6,
                "Research": 1,
            },
            {
                "GRE Score": 300,
                "TOEFL Score": 100,
                "University Rating": 2,
                "SOP": 3.0,
                "LOR": 3.0,
                "CGPA": 8.0,
                "Research": 0,
            },
            {
                "GRE Score": 320,
                "TOEFL Score": 110,
                "University Rating": 4,
                "SOP": 4.0,
                "LOR": 4.0,
                "CGPA": 8.9,
                "Research": 1,
            },
        ]
    )

    st.download_button(
        label="⬇️ Descargar archivo de ejemplo",
        data=dataframe_to_csv(example_data),
        file_name="admission_batch_example.csv",
        mime="text/csv",
        use_container_width=True,
    )

    uploaded_file = st.file_uploader(
        "Seleccione un archivo CSV",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            batch_data.columns = batch_data.columns.str.strip()

            st.markdown("#### Vista previa de los datos")
            st.dataframe(batch_data, use_container_width=True)

            missing_columns = validate_batch_data(batch_data)

            if missing_columns:
                st.error(
                    "El archivo no contiene todas las columnas requeridas. "
                    f"Faltan: {', '.join(missing_columns)}"
                )

            elif batch_data.empty:
                st.error("El archivo cargado no contiene registros.")

            elif st.button(
                "🔮 Generar predicciones batch",
                use_container_width=True,
            ):
                results = generate_batch_predictions(batch_data, model)

                st.success(f"Se generaron correctamente {len(results)} predicciones.")

                st.markdown("#### Resultados")
                st.dataframe(results, use_container_width=True)

                st.download_button(
                    label="⬇️ Descargar predicciones",
                    data=dataframe_to_csv(results),
                    file_name="admission_batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except (TypeError, ValueError, pd.errors.ParserError) as error:
            st.error(
                "No fue posible procesar el archivo. "
                "Verifique que sea un CSV válido y que los datos tengan "
                "el formato esperado."
            )
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
