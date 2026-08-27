"""
Inference and CLI Prediction Script for Crop Yield Prediction.
Supports both interactive command line input and scripted argument flags.
"""

import sys
import os
import argparse
import json

# Safe UTF-8 encoding for Windows console
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

from src.model import predict_yield, load_trained_model, get_default_model_path


def main():
    parser = argparse.ArgumentParser(
        description="Predict Crop Yield using the trained Machine Learning Pipeline."
    )
    parser.add_argument(
        "--country",
        type=str,
        default="India",
        help="Country or geographical area name (e.g. India, United States, Spain)",
    )
    parser.add_argument(
        "--crop",
        type=str,
        default="Rice, paddy",
        help="Crop variety name (e.g. 'Rice, paddy', Wheat, Maize, Potatoes)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Harvest / prediction year (e.g. 2024)",
    )
    parser.add_argument(
        "--rainfall",
        type=float,
        default=1080.0,
        help="Average annual rainfall in mm/year (e.g. 1080.0)",
    )
    parser.add_argument(
        "--pesticides",
        type=float,
        default=550.0,
        help="Pesticides applied in metric tonnes (e.g. 550.0)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=26.5,
        help="Average annual temperature in Celsius (e.g. 26.5)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result strictly in JSON format",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to serialized model artifact (.joblib)",
    )

    args = parser.parse_args()

    input_payload = {
        "Area": args.country,
        "Item": args.crop,
        "Year": args.year,
        "average_rain_fall_mm_per_year": args.rainfall,
        "pesticides_tonnes": args.pesticides,
        "avg_temp": args.temperature,
    }

    try:
        result = predict_yield(input_payload, model_path=args.model_path)
    except FileNotFoundError as fnf_err:
        print(f"[ERROR] {fnf_err}", file=sys.stderr)
        sys.exit(1)
    except ValueError as val_err:
        print(f"[INPUT ERROR] {val_err}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[RUNTIME ERROR] Failed to generate prediction: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("=========================================================")
    print("🌾 CROP YIELD PREDICTION RESULT")
    print("=========================================================")
    print(f"Country / Area       : {args.country}")
    print(f"Crop Variety         : {args.crop}")
    print(f"Target Year          : {args.year}")
    print(f"Annual Rainfall      : {args.rainfall:.1f} mm/year")
    print(f"Pesticides Applied   : {args.pesticides:.1f} tonnes")
    print(f"Average Temperature  : {args.temperature:.1f} °C")
    print("---------------------------------------------------------")
    print(f"PREDICTED YIELD      : {result['predicted_yield_hg_ha']:,.2f} hg/ha")
    print(f"EQUIVALENT YIELD     : {result['predicted_yield_tonnes_ha']:,.2f} tonnes/hectare")
    print(f"Model Pipeline       : {result['model_type']}")
    print("=========================================================")


if __name__ == "__main__":
    main()
