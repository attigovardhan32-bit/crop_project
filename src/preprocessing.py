"""
Preprocessing module for Crop Yield Prediction.
Handles dataset loading, cleaning, validation, and ColumnTransformer pipeline creation.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Standard feature names
categorical_features = ["Area", "Item", "Season"]
NUMERICAL_FEATURES = [
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET_COLUMN = "hg/ha_yield"


def get_default_data_path() -> str:
    """Returns the absolute or relative path to the processed dataset."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_path = os.path.join(base_dir, "data", "processed", "cleaned_crop_yield.csv")
    if os.path.exists(processed_path):
        return processed_path
    raw_path = os.path.join(base_dir, "data", "raw", "yield_df.csv")
    return raw_path


def load_dataset(file_path: str = None) -> pd.DataFrame:
    """
    Loads and cleans the crop yield dataset.
    Removes Unnamed index columns and duplicates.
    """
    if file_path is None:
        file_path = get_default_data_path()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    df = pd.read_csv(file_path)

    # Drop unnamed index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Ensure required columns exist
    required_cols = ALL_FEATURES + [TARGET_COLUMN]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataset.")

    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Clean numeric types
    for num_col in NUMERICAL_FEATURES + [TARGET_COLUMN]:
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    # Drop any rows with NaN in essential columns
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer for feature preprocessing:
    - Categorical features (Area, Item): OneHotEncoder with handle_unknown='ignore'
    - Numerical features (Year, Rainfall, Pesticides, Temperature): StandardScaler
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), NUMERICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_train_test_data(
    file_path: str = None, test_size: float = 0.2, random_state: int = 42
):
    """
    Loads dataset and performs a reproducible train/test split.
    Returns: X_train, X_test, y_train, y_test
    """
    df = load_dataset(file_path)
    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, df


def validate_input_dict(data: dict) -> pd.DataFrame:
    """
    Validates a dictionary of user input features and converts to a DataFrame.
    """
    area = str(data.get("Area", "")).strip()
    item = str(data.get("Item", "")).strip()

    if not area:
        raise ValueError("Country/Area cannot be empty.")
    if not item:
        raise ValueError("Crop/Item cannot be empty.")

    try:
        year = int(data.get("Year", 2024))
    except (ValueError, TypeError):
        raise ValueError("Year must be a valid integer.")

    try:
        rainfall = float(data.get("average_rain_fall_mm_per_year", 0.0))
        if rainfall < 0:
            raise ValueError("Rainfall cannot be negative.")
    except (ValueError, TypeError):
        raise ValueError("Average rainfall must be a valid number.")

    try:
        pesticides = float(data.get("pesticides_tonnes", 0.0))
        if pesticides < 0:
            raise ValueError("Pesticides used cannot be negative.")
    except (ValueError, TypeError):
        raise ValueError("Pesticides used must be a valid number.")

    try:
        temp = float(data.get("avg_temp", 25.0))
        if temp < -50 or temp > 60:
            raise ValueError("Temperature must be within a realistic range (-50°C to 60°C).")
    except (ValueError, TypeError):
        raise ValueError("Average temperature must be a valid number.")

    input_df = pd.DataFrame([
        {
            "Area": area,
            "Item": item,
            "Year": year,
            "average_rain_fall_mm_per_year": rainfall,
            "pesticides_tonnes": pesticides,
            "avg_temp": temp,
        }
    ])
    return input_df
