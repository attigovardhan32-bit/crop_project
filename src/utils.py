"""
Utility and helper functions for Crop Yield Prediction & Smart Farm Analytics.
Includes Tree Planting Calculator, Farm Advisory heuristics, and dataset lookup utilities.
"""

import pandas as pd
from src.preprocessing import load_dataset


def get_country_and_crop_lists(df: pd.DataFrame = None):
    """
    Returns sorted lists of unique countries and crops from the dataset,
    along with a country-to-crops dictionary.
    """
    if df is None:
        df = load_dataset()

    countries = sorted(df["Area"].unique().tolist())
    crops = sorted(df["Item"].unique().tolist())

    country_crop_map = {}
    for country in countries:
        c_crops = sorted(df[df["Area"] == country]["Item"].unique().tolist())
        country_crop_map[country] = c_crops

    return countries, crops, country_crop_map


def get_country_defaults(country: str, crop: str = None, df: pd.DataFrame = None) -> dict:
    """
    Returns realistic average historical values for rainfall, temperature, and pesticides
    for a given country and crop to pre-populate form inputs.
    """
    if df is None:
        df = load_dataset()

    subset = df[df["Area"] == country]
    if crop and not subset.empty and crop in subset["Item"].values:
        subset = subset[subset["Item"] == crop]

    if subset.empty:
        # Global fallback defaults
        return {
            "rainfall": 1000.0,
            "temperature": 25.0,
            "pesticides": 500.0,
            "year": 2024,
        }

    return {
        "rainfall": round(float(subset["average_rain_fall_mm_per_year"].mean()), 1),
        "temperature": round(float(subset["avg_temp"].mean()), 1),
        "pesticides": round(float(subset["pesticides_tonnes"].mean()), 1),
        "year": 2024,
    }


def calculate_tree_planting_capacity(
    area_value: float,
    area_unit: str = "hectares",
    row_spacing_m: float = 4.0,
    tree_spacing_m: float = 3.0,
    usable_percentage: float = 90.0,
    tree_species: str = "Fruit / Timber Trees",
) -> dict:
    """
    Calculates tree planting capacity, land utilization, and estimated carbon offset.
    Units supported: hectares, acres, m2 (square meters).
    """
    if area_value <= 0:
        raise ValueError("Field area must be greater than 0.")
    if row_spacing_m <= 0 or tree_spacing_m <= 0:
        raise ValueError("Grid spacing must be greater than 0 meters.")
    if not (1 <= usable_percentage <= 100):
        raise ValueError("Usable field percentage must be between 1% and 100%.")

    # Convert area to square meters
    unit_lower = area_unit.lower()
    if "hectare" in unit_lower:
        total_area_m2 = area_value * 10000.0
    elif "acre" in unit_lower:
        total_area_m2 = area_value * 4046.856
    elif "m2" in unit_lower or "sq" in unit_lower:
        total_area_m2 = area_value
    else:
        total_area_m2 = area_value * 10000.0  # default to hectares

    usable_area_m2 = total_area_m2 * (usable_percentage / 100.0)
    area_per_tree_m2 = row_spacing_m * tree_spacing_m
    estimated_trees = int(usable_area_m2 / area_per_tree_m2)

    # Approximate field perimeter assuming near-square layout
    approx_side_m = total_area_m2 ** 0.5
    perimeter_m = approx_side_m * 4
    boundary_tree_count = int(perimeter_m / max(tree_spacing_m, 2.0))

    # Carbon sequestration estimate (~21.77 kg CO2/year per mature tree)
    annual_co2_kg = estimated_trees * 21.8
    annual_co2_tonnes = annual_co2_kg / 1000.0

    return {
        "input_area": area_value,
        "area_unit": area_unit,
        "total_area_m2": round(total_area_m2, 2),
        "usable_area_m2": round(usable_area_m2, 2),
        "usable_percentage": usable_percentage,
        "row_spacing_m": row_spacing_m,
        "tree_spacing_m": tree_spacing_m,
        "area_per_tree_m2": round(area_per_tree_m2, 2),
        "tree_density_per_ha": round(10000.0 / area_per_tree_m2, 0),
        "estimated_tree_count": estimated_trees,
        "boundary_tree_count": boundary_tree_count,
        "annual_co2_kg": round(annual_co2_kg, 1),
        "annual_co2_tonnes": round(annual_co2_tonnes, 2),
        "tree_species": tree_species,
    }


def generate_farm_advisory(
    crop: str,
    country: str,
    rainfall_mm: float,
    temp_c: float,
    soil_type: str = "Loamy Soil",
    nitrogen_n: float = 80.0,
    phosphorus_p: float = 40.0,
    potassium_k: float = 40.0,
) -> dict:
    """
    Generates agronomic advice based on crop biological requirements and environmental inputs.
    """
    # Crop biological profile reference
    crop_profiles = {
        "Rice, paddy": {"opt_rain": (1000, 2500), "opt_temp": (20, 35), "opt_soil": "Clayey / Alluvial", "npk": (100, 50, 50)},
        "Wheat": {"opt_rain": (400, 900), "opt_temp": (12, 25), "opt_soil": "Loamy / Alluvial", "npk": (120, 60, 40)},
        "Maize": {"opt_rain": (500, 1100), "opt_temp": (18, 30), "opt_soil": "Well-drained Loam", "npk": (120, 60, 50)},
        "Potatoes": {"opt_rain": (500, 800), "opt_temp": (15, 22), "opt_soil": "Sandy Loam / Loose Soil", "npk": (150, 80, 100)},
        "Soybeans": {"opt_rain": (600, 1000), "opt_temp": (20, 30), "opt_soil": "Fertile Loam", "npk": (30, 80, 80)},
        "Cassava": {"opt_rain": (800, 1600), "opt_temp": (25, 32), "opt_soil": "Sandy Clay Loam", "npk": (80, 40, 80)},
        "Sorghum": {"opt_rain": (350, 700), "opt_temp": (22, 35), "opt_soil": "Clay Loam / Alluvial", "npk": (80, 40, 40)},
        "Sweet potatoes": {"opt_rain": (750, 1200), "opt_temp": (21, 28), "opt_soil": "Sandy Loam", "npk": (60, 50, 90)},
        "Yams": {"opt_rain": (1000, 1500), "opt_temp": (25, 30), "opt_soil": "Deep Rich Loam", "npk": (80, 60, 100)},
        "Plantains and others": {"opt_rain": (1200, 2200), "opt_temp": (26, 32), "opt_soil": "Deep Alluvial Loam", "npk": (150, 60, 180)},
    }

    profile = crop_profiles.get(
        crop,
        {"opt_rain": (600, 1200), "opt_temp": (18, 30), "opt_soil": "Loamy Soil", "npk": (100, 50, 50)}
    )

    rain_min, rain_max = profile["opt_rain"]
    temp_min, temp_max = profile["opt_temp"]

    # Rainfall assessment
    if rainfall_mm < rain_min:
        rain_status = f"Deficit (Recommended: {rain_min}–{rain_max} mm/yr). Supplemental drip or furrow irrigation is strongly advised."
        water_alert = "warning"
    elif rainfall_mm > rain_max:
        rain_status = f"Surplus (Recommended: {rain_min}–{rain_max} mm/yr). Ensure effective drainage channels to prevent waterlogging and root rot."
        water_alert = "info"
    else:
        rain_status = f"Optimal ({rainfall_mm:.0f} mm/yr is within ideal range {rain_min}–{rain_max} mm/yr)."
        water_alert = "success"

    # Temperature assessment
    if temp_c < temp_min:
        temp_status = f"Lower than optimal ({temp_min}–{temp_max}°C). Germination and vegetative growth may be slower."
    elif temp_c > temp_max:
        temp_status = f"Higher than optimal ({temp_min}–{temp_max}°C). Consider mulch cover and adequate soil moisture to mitigate thermal stress."
    else:
        temp_status = f"Optimal ({temp_c:.1f}°C is within ideal growing range {temp_min}–{temp_max}°C)."

    # Nutrient balance
    rec_n, rec_p, rec_k = profile["npk"]
    npk_advice = (
        f"Target N-P-K recommendation for {crop}: {rec_n}:{rec_p}:{rec_k} kg/ha. "
        f"Your current baseline ({nitrogen_n}:{phosphorus_p}:{potassium_k}) "
    )
    if nitrogen_n < rec_n:
        npk_advice += "indicates potential Nitrogen deficiency; consider organic compost or nitrogenous top-dressing."
    else:
        npk_advice += "shows adequate nitrogen balance."

    return {
        "crop": crop,
        "country": country,
        "soil_type": soil_type,
        "ideal_soil": profile["opt_soil"],
        "rain_status": rain_status,
        "water_alert": water_alert,
        "temp_status": temp_status,
        "npk_advice": npk_advice,
        "recommended_npk": f"{rec_n}:{rec_p}:{rec_k} kg/ha",
    }
