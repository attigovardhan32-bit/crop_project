"""
Automated Test Suite for Crop Yield Prediction & Smart Farm Analytics.
Verifies data loading, model inference, dynamic responsiveness, input validation,
tree calculator mathematics, and advisory systems.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Ensure project root is in python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing import (
    load_dataset,
    get_train_test_data,
    build_preprocessor,
    validate_input_dict,
    ALL_FEATURES,
    TARGET_COLUMN,
)
from src.model import (
    load_trained_model,
    predict_yield,
    get_default_model_path,
    load_model_metrics,
)
from src.utils import (
    calculate_tree_planting_capacity,
    generate_farm_advisory,
    get_country_and_crop_lists,
)


class TestCropYieldPipeline(unittest.TestCase):
    """Test suite verifying end-to-end functionality of the Crop Yield system."""

    def test_01_dataset_loading_and_integrity(self):
        """Test dataset loads cleanly with expected columns and non-null rows."""
        df = load_dataset()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 20000, "Dataset should have over 20,000 observations.")
        
        for col in ALL_FEATURES + [TARGET_COLUMN]:
            self.assertIn(col, df.columns, f"Column '{col}' must be present in dataset.")
            self.assertEqual(df[col].isnull().sum(), 0, f"Column '{col}' must not contain nulls.")

    def test_02_preprocessing_transformer(self):
        """Test scikit-learn ColumnTransformer processes categorical & numerical features."""
        preprocessor = build_preprocessor()
        df = load_dataset()
        X = df[ALL_FEATURES].head(50)
        
        transformed = preprocessor.fit_transform(X)
        self.assertIsInstance(transformed, np.ndarray)
        self.assertEqual(transformed.shape[0], 50)
        self.assertGreater(transformed.shape[1], len(ALL_FEATURES))

    def test_03_model_loading_and_metrics(self):
        """Test trained model pipeline and evaluation metrics artifacts."""
        model_path = get_default_model_path()
        self.assertTrue(os.path.exists(model_path), f"Model file must exist at '{model_path}'.")

        pipeline = load_trained_model()
        self.assertIsNotNone(pipeline)

        metrics = load_model_metrics()
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.get("r2_test", 0), 0.95, "Test R² score must be >= 0.95.")
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)

    def test_04_prediction_inference(self):
        """Test single inference prediction returning valid positive yield numbers."""
        test_payload = {
            "Area": "India",
            "Item": "Rice, paddy",
            "Year": 2024,
            "average_rain_fall_mm_per_year": 1080.0,
            "pesticides_tonnes": 550.0,
            "avg_temp": 26.5,
        }
        res = predict_yield(test_payload)
        self.assertIn("predicted_yield_hg_ha", res)
        self.assertIn("predicted_yield_tonnes_ha", res)
        self.assertGreater(res["predicted_yield_hg_ha"], 0)
        self.assertAlmostEqual(
            res["predicted_yield_hg_ha"] / 10000.0,
            res["predicted_yield_tonnes_ha"],
            places=2,
        )

    def test_05_prediction_changes_with_inputs(self):
        """Crucial test: Changing inputs must genuinely change prediction output."""
        base_payload = {
            "Area": "India",
            "Item": "Wheat",
            "Year": 2024,
            "average_rain_fall_mm_per_year": 600.0,
            "pesticides_tonnes": 200.0,
            "avg_temp": 20.0,
        }
        pred_base = predict_yield(base_payload)["predicted_yield_hg_ha"]

        # 1. Change rainfall
        high_rain_payload = base_payload.copy()
        high_rain_payload["average_rain_fall_mm_per_year"] = 1800.0
        pred_rain = predict_yield(high_rain_payload)["predicted_yield_hg_ha"]
        self.assertNotEqual(pred_base, pred_rain, "Changing rainfall must change prediction.")

        # 2. Change crop
        crop_payload = base_payload.copy()
        crop_payload["Item"] = "Potatoes"
        pred_crop = predict_yield(crop_payload)["predicted_yield_hg_ha"]
        self.assertNotEqual(pred_base, pred_crop, "Changing crop must change prediction.")

        # 3. Change country
        country_payload = base_payload.copy()
        country_payload["Area"] = "Spain"
        pred_country = predict_yield(country_payload)["predicted_yield_hg_ha"]
        self.assertNotEqual(pred_base, pred_country, "Changing country must change prediction.")

    def test_06_input_validation(self):
        """Test validation catches invalid inputs and prevents crashes."""
        # Empty country
        with self.assertRaises(ValueError):
            validate_input_dict({"Area": "", "Item": "Wheat"})

        # Negative rainfall
        with self.assertRaises(ValueError):
            validate_input_dict({
                "Area": "India",
                "Item": "Wheat",
                "average_rain_fall_mm_per_year": -50.0,
            })

        # Extreme invalid temperature
        with self.assertRaises(ValueError):
            validate_input_dict({
                "Area": "India",
                "Item": "Wheat",
                "avg_temp": 120.0,
            })

    def test_07_tree_calculator_mathematics(self):
        """Test Tree Planting Calculator formulas and conversions."""
        # 2 hectares, 4m x 3m spacing, 90% usable
        # 2 ha = 20,000 m2. Usable = 18,000 m2. Area per tree = 12 m2. Expected = 1,500 trees.
        res = calculate_tree_planting_capacity(
            area_value=2.0,
            area_unit="Hectares",
            row_spacing_m=4.0,
            tree_spacing_m=3.0,
            usable_percentage=90.0,
        )
        self.assertEqual(res["total_area_m2"], 20000.0)
        self.assertEqual(res["usable_area_m2"], 18000.0)
        self.assertEqual(res["area_per_tree_m2"], 12.0)
        self.assertEqual(res["estimated_tree_count"], 1500)
        self.assertGreater(res["annual_co2_tonnes"], 0)

    def test_08_farm_advisory(self):
        """Test Farm Advisory recommendations engine."""
        adv = generate_farm_advisory(
            crop="Rice, paddy",
            country="India",
            rainfall_mm=1200.0,
            temp_c=28.0,
            soil_type="Clay Loam",
        )
        self.assertIn("rain_status", adv)
        self.assertIn("temp_status", adv)
        self.assertIn("npk_advice", adv)


if __name__ == "__main__":
    unittest.main()
