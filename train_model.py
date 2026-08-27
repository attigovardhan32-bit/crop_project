"""
Standalone Model Training Script for Crop Yield Prediction.
Executes reproducible data loading, train/test splitting, pipeline fitting,
genuine evaluation metrics calculation (R², MAE, RMSE), and artifact serialization.
"""

import sys
import os
import argparse

# Safe UTF-8 encoding for Windows terminals
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.model import train_and_evaluate_model


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Crop Yield Prediction machine learning pipeline."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["random_forest", "xgboost", "extra_trees", "gradient_boosting"],
        help="Machine learning regression algorithm to train (default: random_forest)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split (default: 0.2)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of decision trees / boosting stages (default: 100)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=28,
        help="Maximum tree depth for the regressor (default: 28)",
    )

    args = parser.parse_args()

    print("=========================================================")
    print("[TRAIN] CROP YIELD PREDICTION - MACHINE LEARNING PIPELINE")
    print("=========================================================")
    print(f"Selected Model Type  : {args.model}")
    print(f"Test Split Ratio     : {args.test_size * 100:.0f}%")
    print(f"Random Seed State    : {args.random_state}")
    print(f"Estimator Trees      : {args.n_estimators}")
    print(f"Max Tree Depth       : {args.max_depth}")
    print("---------------------------------------------------------")

    metrics = train_and_evaluate_model(
        model_type=args.model,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )

    print("[SUCCESS] Model training and evaluation completed!")
    print(f"Verified Test R2 Score : {metrics['r2_test']:.4f}")
    print(f"Test MAE               : {metrics['mae']:.2f} hg/ha ({metrics['mae']/10000:.3f} tonnes/ha)")
    print(f"Test RMSE              : {metrics['rmse']:.2f} hg/ha ({metrics['rmse']/10000:.3f} tonnes/ha)")
    print("Artifacts saved to 'models/' directory.")


if __name__ == "__main__":
    main()
