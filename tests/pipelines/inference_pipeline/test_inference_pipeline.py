"""Unit tests for the Admissions inference pipeline."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.pipelines.inference_pipeline.inference_pipeline import (
    FEATURE_COLUMNS,
    PREDICTION_COLUMN,
    InferenceValidationError,
    build_prediction_output,
    generate_predictions,
    load_inference_data,
    load_model,
    prepare_features,
    run_inference_pipeline,
    save_predictions,
    validate_inference_data,
)

DUMMY_PREDICTION = 0.80


class DummyAdmissionModel:
    """Dummy model used to test the inference pipeline."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return deterministic synthetic predictions."""
        return np.full(len(features), DUMMY_PREDICTION)


def create_sample_inference_data() -> pd.DataFrame:
    """Create synthetic admissions observations for inference tests."""
    return pd.DataFrame(
        {
            "GRE Score": [325, 310, 335],
            "TOEFL Score": [112, 105, 118],
            "University Rating": [4, 3, 5],
            "SOP": [4.0, 3.5, 4.5],
            "LOR": [4.5, 3.5, 4.5],
            "CGPA": [9.1, 8.4, 9.6],
            "Research": [1, 0, 1],
        }
    )


def test_load_model(tmp_path: Path) -> None:
    """Test loading a serialized prediction model."""
    model_path = tmp_path / "dummy_model.joblib"
    dummy_model = DummyAdmissionModel()

    joblib.dump(dummy_model, model_path)

    loaded_model = load_model(model_path)

    data = create_sample_inference_data()
    predictions = loaded_model.predict(data)

    assert len(predictions) == len(data)


def test_load_model_missing_file(tmp_path: Path) -> None:
    """Test error when the trained model does not exist."""
    model_path = tmp_path / "missing_model.joblib"

    with pytest.raises(
        FileNotFoundError,
        match="Trained model not found",
    ):
        load_model(model_path)


def test_load_inference_data_csv(tmp_path: Path) -> None:
    """Test reading new inference observations from CSV."""
    input_path = tmp_path / "new_data.csv"
    expected_data = create_sample_inference_data()

    expected_data.to_csv(input_path, index=False)

    loaded_data = load_inference_data(input_path)

    pd.testing.assert_frame_equal(loaded_data, expected_data)


def test_load_inference_data_parquet(tmp_path: Path) -> None:
    """Test reading new inference observations from Parquet."""
    input_path = tmp_path / "new_data.parquet"
    expected_data = create_sample_inference_data()

    expected_data.to_parquet(input_path, index=False)

    loaded_data = load_inference_data(input_path)

    pd.testing.assert_frame_equal(loaded_data, expected_data)


def test_load_inference_data_missing_file(tmp_path: Path) -> None:
    """Test error when the inference file does not exist."""
    input_path = tmp_path / "missing_data.csv"

    with pytest.raises(
        FileNotFoundError,
        match="Inference data file not found",
    ):
        load_inference_data(input_path)


def test_validate_inference_data() -> None:
    """Test validation of correct inference data."""
    data = create_sample_inference_data()

    validate_inference_data(data)


def test_validate_inference_data_missing_column() -> None:
    """Test detection of a missing required feature."""
    data = create_sample_inference_data().drop(columns=["CGPA"])

    with pytest.raises(
        InferenceValidationError,
        match="Missing required inference columns",
    ):
        validate_inference_data(data)


def test_validate_inference_data_empty() -> None:
    """Test detection of an empty inference dataset."""
    data = pd.DataFrame(columns=FEATURE_COLUMNS)

    with pytest.raises(
        InferenceValidationError,
        match="Inference data is empty",
    ):
        validate_inference_data(data)


def test_prepare_features() -> None:
    """Test preparation of model input features."""
    data = create_sample_inference_data()

    features = prepare_features(data)

    assert features.columns.tolist() == FEATURE_COLUMNS
    assert len(features) == len(data)


def test_generate_predictions() -> None:
    """Test prediction generation with a dummy model."""
    data = create_sample_inference_data()
    model = DummyAdmissionModel()

    predictions = generate_predictions(model, data)

    assert len(predictions) == len(data)
    assert predictions.name == PREDICTION_COLUMN
    assert (predictions == DUMMY_PREDICTION).all()


def test_build_prediction_output() -> None:
    """Test combination of observations and predictions."""
    data = create_sample_inference_data()
    predictions = pd.Series(
        [0.80, 0.80, 0.80],
        name=PREDICTION_COLUMN,
    )

    output = build_prediction_output(
        data,
        predictions,
    )

    assert PREDICTION_COLUMN in output.columns
    assert len(output) == len(data)
    assert (output[PREDICTION_COLUMN] == DUMMY_PREDICTION).all()


def test_save_predictions(tmp_path: Path) -> None:
    """Test persistence of inference predictions."""
    output_path = tmp_path / "predictions.csv"

    data = create_sample_inference_data()
    data[PREDICTION_COLUMN] = [0.80, 0.80, 0.80]

    save_predictions(
        data,
        output_path,
    )

    assert output_path.exists()

    saved_data = pd.read_csv(output_path)

    assert PREDICTION_COLUMN in saved_data.columns
    assert len(saved_data) == len(data)


def test_run_inference_pipeline(tmp_path: Path) -> None:
    """Test autonomous execution of the complete inference pipeline."""
    model_path = tmp_path / "dummy_model.joblib"
    input_path = tmp_path / "new_data.csv"
    output_path = tmp_path / "predictions.csv"

    dummy_model = DummyAdmissionModel()
    joblib.dump(dummy_model, model_path)

    input_data = create_sample_inference_data()
    input_data.to_csv(input_path, index=False)

    result = run_inference_pipeline(
        model_path=model_path,
        input_path=input_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert PREDICTION_COLUMN in result.columns
    assert len(result) == len(input_data)
    assert (result[PREDICTION_COLUMN] == DUMMY_PREDICTION).all()
