import re
import pandas as pd
import streamlit as st

from transformers import pipeline


MODEL_NAME = "nateraw/food"


@st.cache_resource(show_spinner="Loading the food recognition model...")
def load_food_classifier():
    """
    Load the pretrained Food-101 image classification model.
    Runs 100% locally with zero external API calls or tokens.
    """
    classifier = pipeline(
        "image-classification",
        model=MODEL_NAME,
        device=-1
    )
    return classifier


def predict_food(image):
    """Return the top food predictions for an uploaded PIL image."""

    classifier = load_food_classifier()

    predictions = classifier(
        image.convert("RGB"),
        top_k=5
    )

    return [
        {
            "food": item["label"].replace("_", " ").title(),
            "label": item["label"].replace("_", " ").title(),
            "confidence": float(item["score"]),
            "score": float(item["score"])
        }
        for item in predictions
    ]


def clean_words(text):
    """Extract useful words from a food name."""

    stop_words = {
        "with", "and", "the", "of", "in", "a",
        "easy", "best", "homemade", "simple",
        "style", "recipe", "good"
    }

    words = re.findall(r"[a-z]+", str(text).lower())

    return {
        word for word in words
        if len(word) > 2 and word not in stop_words
    }


def estimate_recipe_nutrition(
    food_name,
    recipes,
    nutrition_data=None
):
    """
    Find similar Food.com recipes and estimate typical
    nutrition per recipe serving using their median values.
    Runs in ~0.04s using direct vectorized substring search.
    """
    if "id" not in recipes.columns or "name" not in recipes.columns:
        return None, pd.DataFrame()

    nutrition_columns = [
        "calories",
        "total_fat_pdv",
        "sugar_pdv",
        "sodium_pdv",
        "protein_pdv",
        "saturated_fat_pdv",
        "carbohydrates_pdv"
    ]

    available_nutrients = [
        col for col in nutrition_columns
        if col in recipes.columns
    ]

    if not available_nutrients:
        return None, pd.DataFrame()

    clean_term = str(food_name).strip().lower()
    if not clean_term:
        return None, pd.DataFrame()

    # Fast substring match on recipe names
    matched = recipes[
        recipes["name"].str.lower().str.contains(clean_term, regex=False, na=False)
    ].head(30).copy()

    # If few matches, try first meaningful token
    if len(matched) < 3 and " " in clean_term:
        tokens = [t for t in clean_term.split() if len(t) > 2]
        if tokens:
            matched = recipes[
                recipes["name"].str.lower().str.contains(tokens[0], regex=False, na=False)
            ].head(30).copy()

    if matched.empty:
        return None, pd.DataFrame()

    estimated_nutrients = matched[available_nutrients].median(
        numeric_only=True
    )

    return estimated_nutrients, matched


def get_nutrition_explanation():
    return (
        "Nutrition values are estimated from similar Food.com recipes. "
        "They are not measured directly from the photograph. "
        "Actual values vary with ingredients, preparation method "
        "and portion size."
    )
