"""
Model definition, training pipeline, and serialization module for Crop Yield Prediction.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, mean_squared_error

# Safe encoding for Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.preprocessing import (
    build_preprocessor,
    get_train_test_data,
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    validate_input_dict,
)


def get_default_model_path() -> str:
    """Returns the default path for the serialized model artifact."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "models", "crop_yield_model.joblib")


def get_default_metrics_path() -> str:
    """Returns the default path for the evaluation metrics JSON artifact."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "models", "model_metrics.json")


def create_model_pipeline(
    model_type: str = "random_forest",
    random_state: int = 42,
    **kwargs
) -> Pipeline:
    """
    Creates a complete end-to-end Pipeline with preprocessing and regressor.
    """
    preprocessor = build_preprocessor()

    if model_type == "random_forest":
        regressor = RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 28),
            min_samples_split=kwargs.get("min_samples_split", 2),
            min_samples_leaf=kwargs.get("min_samples_leaf", 1),
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "xgboost" and HAS_XGBOOST:
        regressor = XGBRegressor(
            n_estimators=kwargs.get("n_estimators", 600),
            max_depth=kwargs.get("max_depth", 12),
            learning_rate=kwargs.get("learning_rate", 0.08),
            subsample=kwargs.get("subsample", 0.85),
            colsample_bytree=kwargs.get("colsample_bytree", 0.85),
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "extra_trees":
        regressor = ExtraTreesRegressor(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 26),
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        regressor = GradientBoostingRegressor(
            n_estimators=kwargs.get("n_estimators", 250),
            learning_rate=kwargs.get("learning_rate", 0.1),
            max_depth=kwargs.get("max_depth", 8),
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])
    return pipeline


def train_and_evaluate_model(
    model_type: str = "random_forest",
    data_path: str = None,
    model_save_path: str = None,
    metrics_save_path: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
    compress_level: int = 5,
    **kwargs
) -> dict:
    """
    Trains the chosen model pipeline, calculates genuine evaluation metrics,
    and serializes both the model and metadata artifacts.
    """
    if model_save_path is None:
        model_save_path = get_default_model_path()
    if metrics_save_path is None:
        metrics_save_path = get_default_metrics_path()

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)

    print(f"Loading and splitting dataset (test_size={test_size}, random_state={random_state})...")
    X_train, X_test, y_train, y_test, full_df = get_train_test_data(
        file_path=data_path, test_size=test_size, random_state=random_state
    )

    print(f"Building pipeline for model: {model_type}...")
    pipeline = create_model_pipeline(
        model_type=model_type, random_state=random_state, **kwargs
    )

    print("Fitting model pipeline on training data...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on train and held-out test sets...")
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    # Calculate real metrics
    r2_train = float(r2_score(y_train, y_pred_train))
    r2_test = float(r2_score(y_test, y_pred_test))
    mae_test = float(mean_absolute_error(y_test, y_pred_test))
    mse_test = float(mean_squared_error(y_test, y_pred_test))
    rmse_test = float(root_mean_squared_error(y_test, y_pred_test))

    # Mean Absolute Percentage Error
    non_zero_mask = y_test > 0
    mape_test = float(
        np.mean(
            np.abs((y_test[non_zero_mask] - y_pred_test[non_zero_mask]) / y_test[non_zero_mask])
        )
        * 100
    )

    # Save model pipeline with compression
    joblib.dump(pipeline, model_save_path, compress=compress_level)
    model_size_mb = os.path.getsize(model_save_path) / (1024 * 1024)

    # Unique metadata
    unique_countries = sorted(full_df["Area"].unique().tolist())
    unique_crops = sorted(full_df["Item"].unique().tolist())

    metrics_data = {
        "model_name": model_type.replace("_", " ").title(),
        "model_type": model_type,
        "algorithm": type(pipeline.named_steps["regressor"]).__name__,
        "r2_test": round(r2_test, 4),
        "r2_train": round(r2_train, 4),
        "mae": round(mae_test, 2),
        "rmse": round(rmse_test, 2),
        "mse": round(mse_test, 2),
        "mape_percent": round(mape_test, 2),
        "total_records": len(full_df),
        "train_records": len(X_train),
        "test_records": len(X_test),
        "target_column": TARGET_COLUMN,
        "target_unit": "hg/ha (Hectograms per hectare) [10,000 hg/ha = 1 tonne/ha]",
        "feature_columns": ALL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "total_countries": len(unique_countries),
        "total_crops": len(unique_crops),
        "unique_countries": unique_countries,
        "unique_crops": unique_crops,
        "model_size_mb": round(model_size_mb, 2),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "random_state": random_state,
        "test_size": test_size,
    }

    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)

    print("\n================ MODEL TRAINING COMPLETE ================")
    print(f"Model Algorithm      : {metrics_data['algorithm']}")
    print(f"R2 Score (Test)      : {metrics_data['r2_test']:.4f}")
    print(f"R2 Score (Train)     : {metrics_data['r2_train']:.4f}")
    print(f"MAE (Test)           : {metrics_data['mae']:.2f} hg/ha ({metrics_data['mae']/10000:.3f} tonnes/ha)")
    print(f"RMSE (Test)          : {metrics_data['rmse']:.2f} hg/ha ({metrics_data['rmse']/10000:.3f} tonnes/ha)")
    print(f"Model Artifact Size  : {model_size_mb:.2f} MB")
    print(f"Saved Model File     : {model_save_path}")
    print(f"Saved Metrics File   : {metrics_save_path}")
    print("=========================================================\n")

    return metrics_data


def load_trained_model(model_path: str = None) -> Pipeline:
    """
    Loads the serialized model pipeline from disk.
    If the model does not exist, raises FileNotFoundError with helpful instructions.
    """
    if model_path is None:
        model_path = get_default_model_path()

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model artifact not found at '{model_path}'. "
            f"Please run 'python train_model.py' first to train and generate the model."
        )

    pipeline = joblib.load(model_path)
    return pipeline


def load_model_metrics(metrics_path: str = None) -> dict:
    """
    Loads model evaluation metrics from disk.
    """
    if metrics_path is None:
        metrics_path = get_default_metrics_path()

    if not os.path.exists(metrics_path):
        return None

    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_yield(
    input_data: dict, model_path: str = None
) -> dict:
    """
    Performs real machine learning inference for user input features.
    Returns dictionary with predicted yield in hg/ha and tonnes/ha, plus metadata.
    """
    pipeline = load_trained_model(model_path)
    input_df = validate_input_dict(input_data)

    pred_hg_ha = float(pipeline.predict(input_df)[0])
    # Yield cannot be negative in real agriculture
    pred_hg_ha = max(0.0, pred_hg_ha)
    pred_tonnes_ha = pred_hg_ha / 10000.0

    return {
        "predicted_yield_hg_ha": round(pred_hg_ha, 2),
        "predicted_yield_tonnes_ha": round(pred_tonnes_ha, 2),
        "input_features": input_data,
        "model_type": type(pipeline.named_steps["regressor"]).__name__,
    }
