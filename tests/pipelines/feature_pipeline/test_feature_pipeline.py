"""Unit tests for the Admissions feature pipeline."""

from pathlib import Path

import pandas as pd
import pytest

from src.pipelines.feature_pipeline.feature_pipeline import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    DataValidationError,
    load_data,
    save_features,
    transform_data,
    validate_data,
)


def create_valid_data() -> pd.DataFrame:
    """Create a valid synthetic admissions dataset."""
    return pd.DataFrame(
        {
            "GRE Score": [320.0, 310.0],
            "TOEFL Score": [110.0, 105.0],
            "University Rating": [4.0, 3.0],
            "SOP": [4.0, 3.5],
            "LOR": [4.5, 3.5],
            "CGPA": [9.0, 8.5],
            "Research": [1.0, 0.0],
            "Chance of Admit": [0.85, 0.75],
        }
    )


def test_load_data(tmp_path: Path) -> None:
    """Test that the original data can be loaded from a CSV file."""
    input_file = tmp_path / "admissions.csv"

    expected_data = create_valid_data()
    expected_data.columns = [
        "GRE Score",
        "TOEFL Score",
        "University Rating",
        "SOP",
        "LOR ",
        "CGPA",
        "Research",
        "Chance of Admit ",
    ]
    expected_data.to_csv(input_file, index=False)

    loaded_data = load_data(input_file)

    assert loaded_data.shape == (2, 8)
    pd.testing.assert_frame_equal(loaded_data, expected_data)


def test_validate_data_with_valid_data() -> None:
    """Test that valid admissions data passes validation."""
    data = create_valid_data()

    validate_data(data)


def test_validate_data_invalid_range() -> None:
    """Test that values outside valid ranges raise an error."""
    data = create_valid_data()
    data.loc[0, "GRE Score"] = 400.0

    with pytest.raises(DataValidationError, match="GRE Score"):
        validate_data(data)


def test_validate_data_invalid_research_category() -> None:
    """Test that invalid Research categories raise an error."""
    data = create_valid_data()
    data.loc[0, "Research"] = 2.0

    with pytest.raises(DataValidationError, match="Research"):
        validate_data(data)


def test_validate_data_excessive_nulls() -> None:
    """Test that excessive missing values raise an error."""
    data = create_valid_data()
    data.loc[0, "CGPA"] = None

    with pytest.raises(DataValidationError, match="CGPA"):
        validate_data(data)


def test_validate_data_duplicate_records() -> None:
    """Test that duplicate records raise an error."""
    data = create_valid_data()
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)

    with pytest.raises(DataValidationError, match="duplicate"):
        validate_data(data)


def test_transform_data() -> None:
    """Test feature transformations with valid raw data."""
    input_data = create_valid_data()

    input_data.columns = [
        "GRE Score",
        "TOEFL Score",
        "University Rating",
        "SOP",
        "LOR ",
        "CGPA",
        "Research",
        "Chance of Admit ",
    ]

    transformed_data = transform_data(input_data)

    assert transformed_data.shape == (2, 8)
    assert list(transformed_data.columns) == [*FEATURE_COLUMNS, TARGET_COLUMN]
    assert str(transformed_data["GRE Score"].dtype) == "Int16"
    assert str(transformed_data["TOEFL Score"].dtype) == "Int16"
    assert str(transformed_data["Research"].dtype) == "boolean"
    assert not transformed_data.duplicated().any()


def test_save_features(tmp_path: Path) -> None:
    """Test that processed features are stored as a Parquet file."""
    output_file = tmp_path / "admission_features.parquet"

    data = create_valid_data()
    data["GRE Score"] = data["GRE Score"].astype("Int16")
    data["TOEFL Score"] = data["TOEFL Score"].astype("Int16")
    data["Research"] = data["Research"].astype("boolean")

    save_features(data, output_file)

    assert output_file.exists()

    stored_data = pd.read_parquet(output_file)
    pd.testing.assert_frame_equal(stored_data, data)
