"""Training pipeline for the Admissions Prediction project."""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Add the project root to sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Project paths
FEATURE_DATA_PATH = _PROJECT_ROOT / "data" / "04_feature" / "admission_features.parquet"
MODEL_PATH = _PROJECT_ROOT / "models" / "ridge_optimized_pipeline.joblib"
OUTPUT_PATH = _PROJECT_ROOT / "data" / "07_model_output"
METRICS_PATH = OUTPUT_PATH / "training_metrics.json"
METRICS_COMPARISON_PLOT_PATH = OUTPUT_PATH / "model_validation_metrics.png"
CV_FOLDS_PLOT_PATH = OUTPUT_PATH / "cross_validation_folds.png"

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

# Train/Test validation thresholds
MIN_TEST_TRAIN_RATIO = 0.01
DRIFT_P_VALUE_THRESHOLD = 0.05
MAX_CATEGORY_PROPORTION_DIFFERENCE = 0.20

# Model validation configuration
CV_FOLDS = 5
OVERFITTING_R2_GAP = 0.10
UNDERFITTING_R2_THRESHOLD = 0.50


class TrainTestValidationError(ValueError):
    """Raised when a critical train/test validation check fails."""


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


def _validate_dataset_sizes(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> dict[str, float | bool]:
    """Validate the relative size of train and test datasets."""
    if x_train.empty or x_test.empty:
        message = "Train and test datasets must not be empty."
        raise TrainTestValidationError(message)

    test_train_ratio = len(x_test) / len(x_train)
    passed = test_train_ratio > MIN_TEST_TRAIN_RATIO

    if not passed:
        message = (
            f"Test dataset is too small compared with train dataset. Ratio: {test_train_ratio:.4f}"
        )
        raise TrainTestValidationError(message)

    return {
        "passed": True,
        "test_train_ratio": float(test_train_ratio),
    }


def _validate_index_leakage(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> dict[str, int | bool]:
    """Check whether train and test share dataframe indices."""
    leaking_indices = x_train.index.intersection(x_test.index)

    if len(leaking_indices) > 0:
        message = f"Train-test index leakage detected. Shared indices: {len(leaking_indices)}"
        raise TrainTestValidationError(message)

    return {
        "passed": True,
        "shared_indices": 0,
    }


def _validate_sample_mix(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> dict[str, int | bool]:
    """Check whether identical feature rows appear in train and test."""
    train_rows = {
        tuple(row) for row in x_train.astype(object).where(pd.notna(x_train), None).to_numpy()
    }
    test_rows = {
        tuple(row) for row in x_test.astype(object).where(pd.notna(x_test), None).to_numpy()
    }

    mixed_samples = train_rows.intersection(test_rows)

    if mixed_samples:
        message = f"Train-test sample mixing detected. Shared feature rows: {len(mixed_samples)}"
        raise TrainTestValidationError(message)

    return {
        "passed": True,
        "shared_samples": 0,
    }


def _check_numeric_drift(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> dict[str, dict[str, float | bool]]:
    """Compare numeric feature distributions using Kolmogorov-Smirnov."""
    results = {}

    for column in NUMERIC_COLUMNS:
        train_values = x_train[column].dropna().astype(float)
        test_values = x_test[column].dropna().astype(float)

        statistic, p_value = ks_2samp(train_values, test_values)
        passed = p_value >= DRIFT_P_VALUE_THRESHOLD

        results[column] = {
            "passed": bool(passed),
            "ks_statistic": float(statistic),
            "p_value": float(p_value),
        }

        if not passed:
            logger.warning(
                f"Possible distribution drift detected in '{column}': "
                f"KS={statistic:.4f}, p-value={p_value:.4f}"
            )

    return results


def _check_categorical_drift(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> dict[str, dict[str, float | bool]]:
    """Compare category proportions between train and test."""
    results = {}

    for column in CATEGORICAL_COLUMNS:
        train_proportions = x_train[column].value_counts(
            normalize=True,
            dropna=False,
        )
        test_proportions = x_test[column].value_counts(
            normalize=True,
            dropna=False,
        )

        categories = train_proportions.index.union(test_proportions.index)

        max_difference = max(
            abs(
                float(train_proportions.get(category, 0.0))
                - float(test_proportions.get(category, 0.0))
            )
            for category in categories
        )

        passed = max_difference <= MAX_CATEGORY_PROPORTION_DIFFERENCE

        results[column] = {
            "passed": bool(passed),
            "max_proportion_difference": float(max_difference),
        }

        if not passed:
            logger.warning(
                f"Possible categorical drift detected in '{column}': "
                f"maximum proportion difference={max_difference:.4f}"
            )

    return results


def _check_target_drift(
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float | bool]:
    """Compare target distributions using Kolmogorov-Smirnov."""
    train_values = y_train.dropna().astype(float)
    test_values = y_test.dropna().astype(float)

    statistic, p_value = ks_2samp(train_values, test_values)
    passed = p_value >= DRIFT_P_VALUE_THRESHOLD

    if not passed:
        logger.warning(f"Possible target drift detected: KS={statistic:.4f}, p-value={p_value:.4f}")

    return {
        "passed": bool(passed),
        "ks_statistic": float(statistic),
        "p_value": float(p_value),
    }


def validate_train_test_split(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, object]:
    """Validate train/test separation for leakage and distribution changes."""
    logger.info("Validate train/test split")

    size_result = _validate_dataset_sizes(x_train, x_test)
    index_result = _validate_index_leakage(x_train, x_test)
    sample_mix_result = _validate_sample_mix(x_train, x_test)

    numeric_drift = _check_numeric_drift(x_train, x_test)
    categorical_drift = _check_categorical_drift(x_train, x_test)
    target_drift = _check_target_drift(y_train, y_test)

    distribution_checks_passed = (
        all(result["passed"] for result in numeric_drift.values())
        and all(result["passed"] for result in categorical_drift.values())
        and bool(target_drift["passed"])
    )

    validation_results = {
        "dataset_size": size_result,
        "index_leakage": index_result,
        "sample_mix": sample_mix_result,
        "numeric_feature_drift": numeric_drift,
        "categorical_feature_drift": categorical_drift,
        "target_drift": target_drift,
        "distribution_checks_passed": distribution_checks_passed,
    }

    if distribution_checks_passed:
        logger.info("Train/test distribution checks passed")
    else:
        logger.warning(
            "Train/test validation detected distribution differences. "
            "Review the reported drift checks."
        )

    logger.info("Critical train/test leakage checks passed")

    return validation_results


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


def cross_validate_model(
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Evaluate model generalization using K-Fold cross-validation."""
    logger.info(f"Run {CV_FOLDS}-Fold cross-validation on training data")

    kfold = KFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }

    cv_results = cross_validate(
        model,
        x_train,
        y_train,
        cv=kfold,
        scoring=scoring,
        return_train_score=False,
    )

    fold_metrics = {
        "MAE": (-cv_results["test_MAE"]).tolist(),
        "RMSE": (-cv_results["test_RMSE"]).tolist(),
        "R2": cv_results["test_R2"].tolist(),
    }

    cv_metrics = {
        "MAE": float(np.mean(fold_metrics["MAE"])),
        "RMSE": float(np.mean(fold_metrics["RMSE"])),
        "R2": float(np.mean(fold_metrics["R2"])),
    }

    logger.info(f"Cross-validation mean metrics: {cv_metrics}")
    logger.info(f"Cross-validation fold metrics: {fold_metrics}")

    return cv_metrics, fold_metrics


def analyze_generalization(
    train_metrics: dict[str, float],
    cv_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> dict[str, str | float]:
    """Analyze underfitting, overfitting, and model generalization."""
    train_r2 = train_metrics["R2"]
    cv_r2 = cv_metrics["R2"]
    test_r2 = test_metrics["R2"]

    train_cv_gap = train_r2 - cv_r2
    train_test_gap = train_r2 - test_r2

    if (
        train_r2 < UNDERFITTING_R2_THRESHOLD
        and cv_r2 < UNDERFITTING_R2_THRESHOLD
        and test_r2 < UNDERFITTING_R2_THRESHOLD
    ):
        diagnosis = "possible_underfitting"
        recommendation = "Review feature engineering, model complexity, and Ridge regularization."
        logger.warning("Possible underfitting detected: train, CV and test R2 are low.")
    elif train_cv_gap > OVERFITTING_R2_GAP or train_test_gap > OVERFITTING_R2_GAP:
        diagnosis = "possible_overfitting"
        recommendation = "Review regularization, feature selection, and model complexity."
        logger.warning(
            "Possible overfitting detected: training performance is "
            "considerably higher than validation or test performance."
        )
    else:
        diagnosis = "adequate_generalization"
        recommendation = "Model performance is stable across train, cross-validation, and test."
        logger.info("Model shows adequate generalization.")

    return {
        "diagnosis": diagnosis,
        "recommendation": recommendation,
        "train_cv_r2_gap": float(train_cv_gap),
        "train_test_r2_gap": float(train_test_gap),
    }


def plot_metric_comparison(
    train_metrics: dict[str, float],
    cv_metrics: dict[str, float],
    test_metrics: dict[str, float],
    filepath: Path = METRICS_COMPARISON_PLOT_PATH,
) -> None:
    """Save comparison plots for train, cross-validation, and test metrics."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Train", "Cross-validation", "Test"]
    metric_sets = [train_metrics, cv_metrics, test_metrics]

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    for axis, metric in zip(axes, ["MAE", "RMSE", "R2"], strict=True):
        values = [metrics[metric] for metrics in metric_sets]
        axis.bar(labels, values)
        axis.set_title(metric)
        axis.set_ylabel("Score")
        axis.grid(axis="y", alpha=0.3)

    figure.suptitle("Model Validation: Train vs Cross-validation vs Test")
    figure.tight_layout()
    figure.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(figure)

    logger.info(f"Model validation metrics plot saved to: {filepath}")


def plot_cross_validation_folds(
    fold_metrics: dict[str, list[float]],
    filepath: Path = CV_FOLDS_PLOT_PATH,
) -> None:
    """Save cross-validation metric results for every fold."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    folds = np.arange(1, CV_FOLDS + 1)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    for axis, metric in zip(axes, ["MAE", "RMSE", "R2"], strict=True):
        axis.plot(folds, fold_metrics[metric], marker="o")
        axis.set_title(metric)
        axis.set_xlabel("Fold")
        axis.set_ylabel("Score")
        axis.set_xticks(folds)
        axis.grid(alpha=0.3)

    figure.suptitle("K-Fold Cross-validation Results")
    figure.tight_layout()
    figure.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(figure)

    logger.info(f"Cross-validation folds plot saved to: {filepath}")


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
    metrics: dict[str, object],
    filepath: Path = METRICS_PATH,
) -> None:
    """Persist model evaluation and validation metrics as JSON."""
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

    validation_results = validate_train_test_split(
        x_train,
        x_test,
        y_train,
        y_test,
    )
    logger.info(f"Train/test validation results: {validation_results}")

    model = build_model()

    cv_metrics, fold_metrics = cross_validate_model(
        model,
        x_train,
        y_train,
    )

    trained_model = train_model(model, x_train, y_train)

    evaluation_metrics = evaluate_model(
        trained_model,
        x_train,
        x_test,
        y_train,
        y_test,
    )

    generalization_analysis = analyze_generalization(
        evaluation_metrics["train"],
        cv_metrics,
        evaluation_metrics["test"],
    )

    metrics = {
        "train": evaluation_metrics["train"],
        "cross_validation": cv_metrics,
        "cross_validation_folds": fold_metrics,
        "test": evaluation_metrics["test"],
        "generalization_analysis": generalization_analysis,
    }

    plot_metric_comparison(
        evaluation_metrics["train"],
        cv_metrics,
        evaluation_metrics["test"],
    )
    plot_cross_validation_folds(fold_metrics)

    save_model(trained_model)
    save_metrics(metrics)

    logger.info("Admissions Training Pipeline completed successfully")


if __name__ == "__main__":
    run_training_pipeline()
