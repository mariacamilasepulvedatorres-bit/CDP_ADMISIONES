"""Inference pipeline for the Admissions prediction model."""

from pathlib import Path
from typing import Any, Protocol, cast

import joblib
import pandas as pd
from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_ROOT / "models" / "ridge_optimized_pipeline.joblib"
INPUT_PATH = PROJECT_ROOT / "data" / "08_inference_input" / "admission_new_data.csv"
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "09_inference_output" / "admission_predictions.csv"
)

FEATURE_COLUMNS = [
    "GRE Score",
    "TOEFL Score",
    "University Rating",
    "SOP",
    "LOR",
    "CGPA",
    "Research",
]

TARGET_COLUMN = "Chance of Admit"
PREDICTION_COLUMN = "Predicted Chance of Admit"


class PredictionModel(Protocol):
    """Protocol for models that implement the predict method."""

    def predict(self, features: pd.DataFrame) -> Any:
        """Generate predictions from input features."""


class InferenceValidationError(ValueError):
    """Raised when inference data does not satisfy required conditions."""


def load_model(model_path: Path = MODEL_PATH) -> PredictionModel:
    """Load trained model from project storage."""
    logger.info(f"Loading trained model from: {model_path}")

    if not model_path.exists():
        msg = f"Trained model not found: {model_path}"
        raise FileNotFoundError(msg)

    model = cast(PredictionModel, joblib.load(model_path))

    logger.info("Model loaded successfully")

    return model


def load_inference_data(input_path: Path = INPUT_PATH) -> pd.DataFrame:
    """Read new observations from CSV or Parquet."""
    logger.info(f"Reading inference data from: {input_path}")

    if not input_path.exists():
        msg = f"Inference data file not found: {input_path}"
        raise FileNotFoundError(msg)

    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        data = pd.read_csv(input_path)
    elif suffix == ".parquet":
        data = pd.read_parquet(input_path)
    else:
        msg = (
            "Unsupported inference file format. "
            "Only CSV and Parquet files are supported."
        )
        raise InferenceValidationError(msg)

    data.columns = data.columns.str.strip()

    logger.debug(f"Inference data shape: {data.shape}")
    logger.debug(f"Inference data columns: {data.columns.tolist()}")

    return data


def validate_inference_data(data: pd.DataFrame) -> None:
    """Validate that inference data contains all required features."""
    logger.info("Validating inference data")

    missing_columns = [
        column for column in FEATURE_COLUMNS if column not in data.columns
    ]

    if missing_columns:
        msg = f"Missing required inference columns: {missing_columns}"
        raise InferenceValidationError(msg)

    if data.empty:
        msg = "Inference data is empty"
        raise InferenceValidationError(msg)

    logger.info("Inference data validation passed")


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Select model features using the same schema used during training."""
    logger.info("Preparing features for inference")

    validate_inference_data(data)

    features = data[FEATURE_COLUMNS].copy()

    logger.debug(f"Prepared inference features shape: {features.shape}")

    return features


def generate_predictions(
    model: PredictionModel,
    features: pd.DataFrame,
) -> pd.Series:
    """Generate admission probability predictions."""
    logger.info("Generating admission predictions")

    predictions = model.predict(features)

    prediction_series = pd.Series(
        predictions,
        index=features.index,
        name=PREDICTION_COLUMN,
        dtype="float64",
    )

    logger.info(f"Generated {len(prediction_series)} predictions")
    logger.debug(
        "Prediction range: "
        f"{prediction_series.min():.4f} - {prediction_series.max():.4f}"
    )

    return prediction_series


def build_prediction_output(
    data: pd.DataFrame,
    predictions: pd.Series,
) -> pd.DataFrame:
    """Combine inference observations and model predictions."""
    logger.info("Building prediction output")

    output_data = data.copy()

    if TARGET_COLUMN in output_data.columns:
        output_data = output_data.drop(columns=[TARGET_COLUMN])

    output_data[PREDICTION_COLUMN] = predictions

    return output_data


def save_predictions(
    predictions: pd.DataFrame,
    output_path: Path = OUTPUT_PATH,
) -> None:
    """Store inference results as CSV."""
    logger.info(f"Saving predictions to: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)

    logger.info("Predictions stored successfully")


def run_inference_pipeline(
    model_path: Path = MODEL_PATH,
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Execute the complete Admissions inference pipeline."""
    logger.info("Start Admissions Inference Pipeline")

    model = load_model(model_path)

    data = load_inference_data(input_path)

    features = prepare_features(data)

    predictions = generate_predictions(
        model,
        features,
    )

    prediction_output = build_prediction_output(
        data,
        predictions,
    )

    save_predictions(
        prediction_output,
        output_path,
    )

    logger.info("Admissions Inference Pipeline completed successfully")

    return prediction_output


if __name__ == "__main__":
    run_inference_pipeline()