import os
import requests
import streamlit as st

USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"

NUTRIENT_IDS = {
    1008: "Energy (kcal)",
    1003: "Protein (g)",
    1004: "Total Fat (g)",
    1005: "Carbohydrates (g)",
    1079: "Fiber (g)",
    2000: "Sugars (g)",
    1093: "Sodium (mg)",
    1258: "Saturated Fat (g)",
    1253: "Cholesterol (mg)",
    1087: "Calcium (mg)",
    1089: "Iron (mg)",
    1092: "Potassium (mg)",
    1162: "Vitamin C (mg)",
}


def search_food(query, page_size=10):
    """
    Search for foods in the USDA FoodData Central database by query text.
    Returns a list of simplified food item records.
    """
    api_key = None
    try:
        if hasattr(st, "secrets") and "USDA_API_KEY" in st.secrets:
            api_key = st.secrets["USDA_API_KEY"]
    except Exception:
        pass

    if not api_key:
        api_key = os.environ.get("USDA_API_KEY", "")

    if not api_key or api_key == "PASTE_YOUR_USDA_API_KEY_HERE":
        raise ValueError(
            "USDA API key not found. Please set USDA_API_KEY in .streamlit/secrets.toml."
        )

    url = f"{USDA_BASE_URL}/foods/search"
    params = {
        "api_key": api_key,
        "query": query,
        "pageSize": page_size
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    results = []
    for food in data.get("foods", []):
        results.append({
            "fdcId": food.get("fdcId"),
            "description": food.get("description"),
            "dataType": food.get("dataType"),
            "brand": food.get("brandOwner"),
            "nutrients": food.get("foodNutrients", [])
        })
    return results


def get_nutrients_per_100g(food):
    """
    Extract selected nutrient values from a USDA food record.
    Standard USDA food-composition records are generally reported
    per 100 g. The returned values are used as a per-100-g basis
    for the serving calculation.
    """
    nutrient_values = {}
    for item in food.get("nutrients", []):
        nutrient_id = item.get("nutrientId")
        if nutrient_id in NUTRIENT_IDS:
            value = item.get("value")
            if value is not None:
                nutrient_values[
                    NUTRIENT_IDS[nutrient_id]
                ] = float(value)
    return nutrient_values


def calculate_serving_nutrition(nutrients_per_100g, serving_grams):
    """Scale nutrient values from 100 g to the selected portion."""
    if serving_grams <= 0:
        raise ValueError("Serving size must be greater than zero.")
    factor = serving_grams / 100.0
    return {
        nutrient: round(value * factor, 2)
        for nutrient, value in nutrients_per_100g.items()
    }

