"""Unit tests for the Admissions feature pipeline."""

from pathlib import Path

import pandas as pd

from src.pipelines.feature_pipeline.feature_pipeline import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_data,
    save_features,
    transform_data,
)


def test_load_data(tmp_path: Path) -> None:
    """Test that the original data can be loaded from a CSV file."""
    input_file = tmp_path / "admissions.csv"

    expected_data = pd.DataFrame(
        {
            "GRE Score": [320.0, 310.0],
            "TOEFL Score": [110.0, 105.0],
            "University Rating": [4.0, 3.0],
            "SOP": [4.0, 3.5],
            "LOR ": [4.5, 3.5],
            "CGPA": [9.0, 8.5],
            "Research": [1.0, 0.0],
            "Chance of Admit ": [0.85, 0.75],
        }
    )
    expected_data.to_csv(input_file, index=False)

    loaded_data = load_data(input_file)

    assert loaded_data.shape == (2, 8)
    pd.testing.assert_frame_equal(loaded_data, expected_data)


def test_transform_data() -> None:
    """Test feature transformations and duplicate removal."""
    input_data = pd.DataFrame(
        {
            "GRE Score": [320.0, 310.0, 320.0],
            "TOEFL Score": [110.0, 105.0, 110.0],
            "University Rating": [4.0, 3.0, 4.0],
            "SOP": [4.0, 3.5, 4.0],
            "LOR ": [4.5, 3.5, 4.5],
            "CGPA": [9.0, 8.5, 9.0],
            "Research": [1.0, 0.0, 1.0],
            "Chance of Admit ": [0.85, 0.75, 0.85],
        }
    )

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

    data = pd.DataFrame(
        {
            "GRE Score": pd.Series([320, 310], dtype="Int16"),
            "TOEFL Score": pd.Series([110, 105], dtype="Int16"),
            "University Rating": [4.0, 3.0],
            "SOP": [4.0, 3.5],
            "LOR": [4.5, 3.5],
            "CGPA": [9.0, 8.5],
            "Research": pd.Series([True, False], dtype="boolean"),
            "Chance of Admit": [0.85, 0.75],
        }
    )

    save_features(data, output_file)

    assert output_file.exists()

    stored_data = pd.read_parquet(output_file)
    pd.testing.assert_frame_equal(stored_data, data)
