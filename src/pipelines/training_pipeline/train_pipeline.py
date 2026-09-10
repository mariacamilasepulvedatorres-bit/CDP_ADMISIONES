"""Training pipeline for the Admissions Prediction project."""

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Add the project root to sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Project paths
FEATURE_DATA_PATH = (
    _PROJECT_ROOT / "data" / "04_feature" / "admission_features.parquet"
)
MODEL_PATH = _PROJECT_ROOT / "models" / "ridge_optimized_pipeline.joblib"
METRICS_PATH = (
    _PROJECT_ROOT / "data" / "07_model_output" / "training_metrics.json"
)

# Model variables defined during the POC
NUMERIC_COLUMNS = [
    "GRE Score",
    "TOEFL Score",
    "University Rating",
    "SOP",
    "LOR",
    "CGPA",
]

CATEGORICAL_COLUMNS = ["Research"]

TARGET_COLUMN = "Chance of Admit"

TEST_SIZE = 0.2
RANDOM_STATE = 42
RIDGE_ALPHA = 10.0


def load_features(filepath: Path = FEATURE_DATA_PATH) -> pd.DataFrame:
    """Load processed features from the feature pipeline."""
    logger.info(f"Reading processed features from: {filepath}")

    data = pd.read_parquet(filepath)

    logger.debug(f"Feature data shape: {data.shape}")
    logger.debug(f"Feature data dtypes:\n{data.dtypes}")

    return data


def split_features_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictor variables from the target variable."""
    logger.info("Separate features and target")

    x_features = data.drop(columns=[TARGET_COLUMN]).copy()
    y_target = data[TARGET_COLUMN].copy()

    return x_features, y_target


def split_train_test(
    x_features: pd.DataFrame,
    y_target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and test sets."""
    logger.info("Split data into train and test sets")

    x_train, x_test, y_train, y_test = train_test_split(
        x_features,
        y_target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    logger.debug(f"Training features shape: {x_train.shape}")
    logger.debug(f"Test features shape: {x_test.shape}")

    return x_train, x_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing pipeline used during the POC."""
    logger.info("Build preprocessing pipeline")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            (
                "categoric ordinal",
                categorical_pipeline,
                CATEGORICAL_COLUMNS,
            ),
        ]
    )


def build_model() -> Pipeline:
    """Build the optimized Ridge model selected during the POC."""
    logger.info("Build Ridge regression model")

    preprocessor = build_preprocessor()

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )


def train_model(
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Train the admissions prediction model."""
    logger.info("Train Ridge regression model")

    model.fit(x_train, y_train)

    logger.info("Model training completed successfully")

    return model


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate regression evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def evaluate_model(
    model: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, dict[str, float]]:
    """Evaluate the model on training and test data."""
    logger.info("Evaluate model performance")

    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)

    train_metrics = calculate_metrics(y_train, train_predictions)
    test_metrics = calculate_metrics(y_test, test_predictions)

    logger.info(f"Train metrics: {train_metrics}")
    logger.info(f"Test metrics: {test_metrics}")

    return {
        "train": train_metrics,
        "test": test_metrics,
    }


def save_model(
    model: Pipeline,
    filepath: Path = MODEL_PATH,
) -> None:
    """Persist the trained model."""
    logger.info(f"Saving trained model to: {filepath}")

    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)

    logger.info("Model stored successfully")


def save_metrics(
    metrics: dict[str, dict[str, float]],
    filepath: Path = METRICS_PATH,
) -> None:
    """Persist model evaluation metrics as JSON."""
    logger.info(f"Saving evaluation metrics to: {filepath}")

    filepath.parent.mkdir(parents=True, exist_ok=True)

    with filepath.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    logger.info("Evaluation metrics stored successfully")


def run_training_pipeline() -> None:
    """Run the complete Admissions training pipeline."""
    logger.info("Start Admissions Training Pipeline")

    data = load_features()

    x_features, y_target = split_features_target(data)

    x_train, x_test, y_train, y_test = split_train_test(
        x_features,
        y_target,
    )

    model = build_model()
    trained_model = train_model(model, x_train, y_train)

    metrics = evaluate_model(
        trained_model,
        x_train,
        x_test,
        y_train,
        y_test,
    )

    save_model(trained_model)
    save_metrics(metrics)

    logger.info("Admissions Training Pipeline completed successfully")


if __name__ == "__main__":
    run_training_pipeline()