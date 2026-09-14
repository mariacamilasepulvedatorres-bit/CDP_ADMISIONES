"""Unit tests for the Admissions training pipeline."""

from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.pipelines.training_pipeline.train_pipeline import (
    CV_FOLDS,
    TARGET_COLUMN,
    TrainTestValidationError,
    analyze_generalization,
    build_model,
    calculate_metrics,
    cross_validate_model,
    load_features,
    plot_cross_validation_folds,
    plot_metric_comparison,
    save_metrics,
    save_model,
    split_features_target,
    split_train_test,
    train_model,
    validate_train_test_split,
)


def create_sample_data() -> pd.DataFrame:
    """Create synthetic admissions data for testing."""
    return pd.DataFrame(
        {
            "GRE Score": [320, 310, 330, 300, 315, 325, 305, 318, 312, 328],
            "TOEFL Score": [110, 105, 115, 100, 108, 112, 102, 109, 106, 114],
            "University Rating": [4, 3, 5, 2, 3, 4, 2, 4, 3, 5],
            "SOP": [4.0, 3.5, 4.5, 3.0, 3.5, 4.0, 3.0, 4.0, 3.5, 4.5],
            "LOR": [4.5, 3.5, 4.5, 3.0, 3.5, 4.0, 3.0, 4.0, 3.5, 4.5],
            "CGPA": [9.0, 8.5, 9.5, 8.0, 8.7, 9.2, 8.2, 8.9, 8.6, 9.4],
            "Research": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            TARGET_COLUMN: [
                0.85,
                0.75,
                0.92,
                0.65,
                0.78,
                0.88,
                0.68,
                0.82,
                0.76,
                0.90,
            ],
        }
    )


def test_load_features(tmp_path: Path) -> None:
    """Test reading processed features from Parquet."""
    input_file = tmp_path / "features.parquet"
    expected_data = create_sample_data()
    expected_data.to_parquet(input_file, index=False)

    loaded_data = load_features(input_file)

    pd.testing.assert_frame_equal(loaded_data, expected_data)


def test_split_features_target() -> None:
    """Test separation of predictors and target."""
    data = create_sample_data()

    x_features, y_target = split_features_target(data)

    assert TARGET_COLUMN not in x_features.columns
    assert len(x_features) == len(data)
    assert len(y_target) == len(data)
    pd.testing.assert_series_equal(y_target, data[TARGET_COLUMN])


def test_split_train_test() -> None:
    """Test train and test data separation."""
    data = create_sample_data()
    x_features, y_target = split_features_target(data)

    x_train, x_test, y_train, y_test = split_train_test(
        x_features,
        y_target,
    )

    expected_train_size = 8
    expected_test_size = 2

    assert len(x_train) == expected_train_size
    assert len(x_test) == expected_test_size
    assert len(y_train) == expected_train_size
    assert len(y_test) == expected_test_size


def test_validate_train_test_split_valid() -> None:
    """Test validation of a correct train/test split."""
    data = create_sample_data()
    x_features, y_target = split_features_target(data)

    x_train, x_test, y_train, y_test = split_train_test(
        x_features,
        y_target,
    )

    results = validate_train_test_split(
        x_train,
        x_test,
        y_train,
        y_test,
    )

    dataset_size = results["dataset_size"]
    index_leakage = results["index_leakage"]
    sample_mix = results["sample_mix"]

    assert isinstance(dataset_size, dict)
    assert isinstance(index_leakage, dict)
    assert isinstance(sample_mix, dict)

    assert dataset_size["passed"] is True
    assert index_leakage["passed"] is True
    assert sample_mix["passed"] is True
    assert index_leakage["shared_indices"] == 0
    assert sample_mix["shared_samples"] == 0


def test_validate_train_test_split_detects_index_leakage() -> None:
    """Test detection of shared indices between train and test."""
    data = create_sample_data()
    x_features, y_target = split_features_target(data)

    x_train = x_features.iloc[:8].copy()
    y_train = y_target.iloc[:8].copy()

    x_test = x_features.iloc[7:].copy()
    y_test = y_target.iloc[7:].copy()

    with pytest.raises(
        TrainTestValidationError,
        match="index leakage",
    ):
        validate_train_test_split(
            x_train,
            x_test,
            y_train,
            y_test,
        )


def test_validate_train_test_split_detects_sample_mix() -> None:
    """Test detection of identical samples between train and test."""
    data = create_sample_data()
    x_features, y_target = split_features_target(data)

    x_train = x_features.iloc[:8].copy()
    y_train = y_target.iloc[:8].copy()

    x_test = x_features.iloc[8:].copy()
    y_test = y_target.iloc[8:].copy()

    duplicated_sample = x_train.iloc[[0]].copy()
    duplicated_target = y_train.iloc[[0]].copy()

    duplicated_sample.index = [100]
    duplicated_target.index = [100]

    x_test = pd.concat([x_test, duplicated_sample])
    y_test = pd.concat([y_test, duplicated_target])

    with pytest.raises(
        TrainTestValidationError,
        match="sample mixing",
    ):
        validate_train_test_split(
            x_train,
            x_test,
            y_train,
            y_test,
        )


def test_train_model() -> None:
    """Test model training."""
    data = create_sample_data()
    x_features, y_target = split_features_target(data)

    model = build_model()
    trained_model = train_model(model, x_features, y_target)

    predictions = trained_model.predict(x_features)

    assert len(predictions) == len(data)
    assert hasattr(trained_model.named_steps["model"], "coef_")


def test_calculate_metrics() -> None:
    """Test generation of regression metrics."""
    y_true = pd.Series([0.70, 0.80, 0.90])
    y_pred = pd.Series([0.72, 0.78, 0.88]).to_numpy()

    metrics = calculate_metrics(y_true, y_pred)

    assert set(metrics) == {"MAE", "RMSE", "R2"}
    assert all(isinstance(value, float) for value in metrics.values())


def test_cross_validate_model() -> None:
    """Test K-Fold cross-validation metrics."""
    data = create_sample_data()
    x_features, y_target = split_features_target(data)

    model = build_model()

    cv_metrics, fold_metrics = cross_validate_model(
        model,
        x_features,
        y_target,
    )

    assert set(cv_metrics) == {"MAE", "RMSE", "R2"}
    assert set(fold_metrics) == {"MAE", "RMSE", "R2"}

    assert all(isinstance(value, float) for value in cv_metrics.values())

    assert len(fold_metrics["MAE"]) == CV_FOLDS
    assert len(fold_metrics["RMSE"]) == CV_FOLDS
    assert len(fold_metrics["R2"]) == CV_FOLDS


def test_analyze_generalization_adequate() -> None:
    """Test detection of adequate model generalization."""
    train_metrics = {
        "MAE": 0.045,
        "RMSE": 0.063,
        "R2": 0.81,
    }
    cv_metrics = {
        "MAE": 0.047,
        "RMSE": 0.064,
        "R2": 0.79,
    }
    test_metrics = {
        "MAE": 0.052,
        "RMSE": 0.070,
        "R2": 0.78,
    }

    result = analyze_generalization(
        train_metrics,
        cv_metrics,
        test_metrics,
    )

    assert result["diagnosis"] == "adequate_generalization"


def test_analyze_generalization_detects_overfitting() -> None:
    """Test detection of possible overfitting."""
    train_metrics = {
        "MAE": 0.01,
        "RMSE": 0.02,
        "R2": 0.95,
    }
    cv_metrics = {
        "MAE": 0.08,
        "RMSE": 0.10,
        "R2": 0.70,
    }
    test_metrics = {
        "MAE": 0.09,
        "RMSE": 0.11,
        "R2": 0.68,
    }

    result = analyze_generalization(
        train_metrics,
        cv_metrics,
        test_metrics,
    )

    assert result["diagnosis"] == "possible_overfitting"


def test_analyze_generalization_detects_underfitting() -> None:
    """Test detection of possible underfitting."""
    train_metrics = {
        "MAE": 0.15,
        "RMSE": 0.18,
        "R2": 0.40,
    }
    cv_metrics = {
        "MAE": 0.16,
        "RMSE": 0.19,
        "R2": 0.35,
    }
    test_metrics = {
        "MAE": 0.17,
        "RMSE": 0.20,
        "R2": 0.32,
    }

    result = analyze_generalization(
        train_metrics,
        cv_metrics,
        test_metrics,
    )

    assert result["diagnosis"] == "possible_underfitting"


def test_plot_metric_comparison(tmp_path: Path) -> None:
    """Test generation of model validation comparison plot."""
    output_file = tmp_path / "model_validation_metrics.png"

    train_metrics = {
        "MAE": 0.045,
        "RMSE": 0.063,
        "R2": 0.81,
    }
    cv_metrics = {
        "MAE": 0.047,
        "RMSE": 0.064,
        "R2": 0.79,
    }
    test_metrics = {
        "MAE": 0.052,
        "RMSE": 0.070,
        "R2": 0.78,
    }

    plot_metric_comparison(
        train_metrics,
        cv_metrics,
        test_metrics,
        output_file,
    )

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_plot_cross_validation_folds(tmp_path: Path) -> None:
    """Test generation of K-Fold validation plot."""
    output_file = tmp_path / "cross_validation_folds.png"

    fold_metrics = {
        "MAE": [0.04, 0.041, 0.05, 0.049, 0.053],
        "RMSE": [0.055, 0.057, 0.071, 0.069, 0.071],
        "R2": [0.81, 0.86, 0.80, 0.77, 0.71],
    }

    plot_cross_validation_folds(
        fold_metrics,
        output_file,
    )

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_save_model(tmp_path: Path) -> None:
    """Test trained model persistence."""
    data = create_sample_data()
    x_features, y_target = split_features_target(data)

    model = build_model()
    trained_model = train_model(model, x_features, y_target)

    output_file = tmp_path / "model.joblib"
    save_model(trained_model, output_file)

    assert output_file.exists()

    loaded_model = joblib.load(output_file)
    predictions = loaded_model.predict(x_features)

    assert len(predictions) == len(data)


def test_save_metrics(tmp_path: Path) -> None:
    """Test evaluation and validation metrics persistence."""
    output_file = tmp_path / "metrics.json"

    metrics = {
        "train": {
            "MAE": 0.045,
            "RMSE": 0.063,
            "R2": 0.81,
        },
        "cross_validation": {
            "MAE": 0.047,
            "RMSE": 0.064,
            "R2": 0.79,
        },
        "test": {
            "MAE": 0.052,
            "RMSE": 0.070,
            "R2": 0.78,
        },
        "generalization_analysis": {
            "diagnosis": "adequate_generalization",
        },
    }

    save_metrics(metrics, output_file)

    assert output_file.exists()

    content = output_file.read_text(encoding="utf-8")

    assert '"train"' in content
    assert '"cross_validation"' in content
    assert '"test"' in content
    assert '"generalization_analysis"' in content
    assert '"MAE"' in content
    assert '"RMSE"' in content
    assert '"R2"' in content
