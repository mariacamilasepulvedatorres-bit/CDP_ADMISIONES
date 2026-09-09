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

REQUIRED_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]

# Maximum percentage of missing values allowed per column
MAX_NULL_PERCENTAGE = 10.0

# Valid numeric ranges according to the admissions dataset
VALID_RANGES = {
    "GRE Score": (260, 340),
    "TOEFL Score": (0, 120),
    "University Rating": (1, 5),
    "SOP": (1, 5),
    "LOR": (1, 5),
    "CGPA": (0, 10),
    TARGET_COLUMN: (0, 1),
}


class DataValidationError(ValueError):
    """Raised when admissions data does not satisfy validation rules."""


def load_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the original admissions dataset."""
    logger.info(f"Reading original data from: {filepath}")

    data = pd.read_csv(filepath)

    logger.debug(f"Raw data shape: {data.shape}")
    logger.debug(f"Raw data dtypes:\n{data.dtypes}")

    return data


def validate_required_columns(data: pd.DataFrame) -> list[str]:
    """Validate that all required columns are present."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]

    if missing_columns:
        return [f"Missing required columns: {missing_columns}"]

    return []


def validate_data_types(data: pd.DataFrame) -> list[str]:
    """Validate expected data types."""
    errors = []

    numeric_columns = [
        "GRE Score",
        "TOEFL Score",
        "University Rating",
        "SOP",
        "LOR",
        "CGPA",
        TARGET_COLUMN,
    ]

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            errors.append(f"Column '{column}' must contain numeric values")

    if not pd.api.types.is_numeric_dtype(data["Research"]):
        errors.append("Column 'Research' must contain numeric values 0 or 1")

    return errors


def validate_nulls(data: pd.DataFrame) -> list[str]:
    """Validate the maximum percentage of missing values."""
    errors = []

    for column in REQUIRED_COLUMNS:
        null_percentage = data[column].isna().mean() * 100

        if null_percentage > MAX_NULL_PERCENTAGE:
            errors.append(
                f"Column '{column}' has {null_percentage:.2f}% missing values; "
                f"maximum allowed is {MAX_NULL_PERCENTAGE:.2f}%"
            )

    return errors


def validate_ranges(data: pd.DataFrame) -> list[str]:
    """Validate numeric values against expected ranges."""
    errors = []

    for column, (minimum, maximum) in VALID_RANGES.items():
        valid_values = data[column].dropna()

        if not valid_values.between(minimum, maximum).all():
            errors.append(
                f"Column '{column}' contains values outside the valid range [{minimum}, {maximum}]"
            )

    return errors


def validate_research_categories(data: pd.DataFrame) -> list[str]:
    """Validate allowed categories for the Research variable."""
    research_values = set(data["Research"].dropna().unique())
    valid_research_values = {0, 1}

    if not research_values.issubset(valid_research_values):
        return ["Column 'Research' contains invalid categories. Only values 0 and 1 are allowed"]

    return []


def validate_duplicates(data: pd.DataFrame) -> list[str]:
    """Validate record uniqueness."""
    if data.duplicated().any():
        return ["Dataset contains duplicate records"]

    return []


def validate_data(data: pd.DataFrame) -> None:
    """Validate admissions data quality, format and integrity."""
    logger.info("Validate admissions data")

    errors = validate_required_columns(data)

    if errors:
        message = "Data validation failed:\n- " + "\n- ".join(errors)
        logger.error(message)
        raise DataValidationError(message)

    errors.extend(validate_data_types(data))
    errors.extend(validate_nulls(data))
    errors.extend(validate_ranges(data))
    errors.extend(validate_research_categories(data))
    errors.extend(validate_duplicates(data))

    if errors:
        message = "Data validation failed:\n- " + "\n- ".join(errors)
        logger.error(message)
        raise DataValidationError(message)

    logger.info("Admissions data validation completed successfully")


def transform_data(data: pd.DataFrame) -> pd.DataFrame:
    """Transform the admissions data into model-ready features."""
    logger.info("Transform admissions data into features")

    transformed_data = data.copy()

    # Remove leading and trailing spaces from column names
    transformed_data.columns = transformed_data.columns.str.strip()

    # Remove duplicate records
    transformed_data = transformed_data.drop_duplicates().reset_index(drop=True)

    # Keep the model features and target
    transformed_data = transformed_data[REQUIRED_COLUMNS]

    # Validate data before type conversion and persistence
    validate_data(transformed_data)

    # Set the data types defined during the POC
    transformed_data["GRE Score"] = transformed_data["GRE Score"].astype("Int16")
    transformed_data["TOEFL Score"] = transformed_data["TOEFL Score"].astype("Int16")
    transformed_data["Research"] = transformed_data["Research"].astype("boolean")

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
