"""Feature pipeline for the Admissions Prediction project."""

import os
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

# Add the 'src' directory to sys.path.
# This follows the project structure used for executable pipeline scripts.
_SRCDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRCDIR not in sys.path:
    sys.path.insert(0, _SRCDIR)

# Project paths
_PROJECT_DIR = Path(_SRCDIR).parent
RAW_DATA_PATH = _PROJECT_DIR / "data" / "01_raw" / "Admission_Predict.csv"
FEATURE_DATA_PATH = _PROJECT_DIR / "data" / "04_feature" / "admission_features.parquet"

# Features used by the admissions prediction model
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


def load_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the original admissions dataset."""
    logger.info(f"Reading original data from: {filepath}")

    data = pd.read_csv(filepath)

    logger.debug(f"Raw data shape: {data.shape}")
    logger.debug(f"Raw data dtypes:\n{data.dtypes}")

    return data


def transform_data(data: pd.DataFrame) -> pd.DataFrame:
    """Transform the admissions data into model-ready features."""
    logger.info("Transform admissions data into features")

    transformed_data = data.copy()

    # Remove leading and trailing spaces from column names
    transformed_data.columns = transformed_data.columns.str.strip()

    # Remove duplicate records
    transformed_data = transformed_data.drop_duplicates().reset_index(drop=True)

    # Set the data types defined during the POC
    transformed_data["GRE Score"] = transformed_data["GRE Score"].astype("Int16")
    transformed_data["TOEFL Score"] = transformed_data["TOEFL Score"].astype("Int16")
    transformed_data["Research"] = transformed_data["Research"].astype("boolean")

    # Keep the model features and target
    transformed_data = transformed_data[[*FEATURE_COLUMNS, TARGET_COLUMN]]

    logger.debug(f"Processed data shape: {transformed_data.shape}")
    logger.debug(f"Processed data dtypes:\n{transformed_data.dtypes}")

    return transformed_data


def save_features(
    data: pd.DataFrame,
    filepath: Path = FEATURE_DATA_PATH,
) -> None:
    """Persist the processed features to a Parquet file."""
    logger.info(f"Store processed features in: {filepath}")

    filepath.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(filepath, index=False)

    logger.info("Features stored successfully")


def run_feature_pipeline() -> None:
    """Run the complete feature pipeline."""
    logger.info("Start Admissions Feature Pipeline")

    data = load_data()
    features = transform_data(data)
    save_features(features)

    logger.info("Admissions Feature Pipeline completed successfully")


if __name__ == "__main__":
    run_feature_pipeline()
