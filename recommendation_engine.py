# ============================================================================
# PERSONALIZED NUTRITION-AWARE FOOD RECOMMENDATION SYSTEM
# PHASE 5.8: HYBRID RECOMMENDATION ENGINE
# ============================================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# 1. CONFIGURATION & FILE PATHS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

RECIPES_GZ_PATH = os.path.join(MODEL_DIR, "recipes.csv.gz")
RECIPES_RAW_PATH = os.path.join(MODEL_DIR, "recipes.csv")
RECIPES_PATH = RECIPES_GZ_PATH if os.path.exists(RECIPES_GZ_PATH) else RECIPES_RAW_PATH
MATRIX_PATH = os.path.join(MODEL_DIR, "tfidf_matrix.npz")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
MAPPING_PATH = os.path.join(MODEL_DIR, "recipe_id_to_index.json")
NUTRITION_PATH = os.path.join(MODEL_DIR, "nutrition_scores.csv")

required_files = [
    RECIPES_PATH,
    MATRIX_PATH,
    VECTORIZER_PATH,
    MAPPING_PATH,
    NUTRITION_PATH
]

for file_path in required_files:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required model file not found: {file_path}")

print("=" * 75)
print("LOADING FOOD RECOMMENDATION MODEL & ASSETS")
print("=" * 75)

# ============================================================================
# 2. LOAD RECIPE DATASET
# ============================================================================

available_columns = pd.read_csv(RECIPES_PATH, nrows=0).columns.tolist()

preferred_columns = [
    "id",
    "name",
    "minutes",
    "minutes_clean",
    "time_category",
    "calories",
    "total_fat_pdv",
    "sugar_pdv",
    "sodium_pdv",
    "protein_pdv",
    "saturated_fat_pdv",
    "carbohydrates_pdv",
    "ingredients",
    "ingredients_text",
    "tags",
    "tags_text",
    "description",
    "description_clean",
    "steps"
]

selected_columns = [c for c in preferred_columns if c in available_columns]

recipes = pd.read_csv(
    RECIPES_PATH,
    usecols=selected_columns,
    low_memory=False
)

recipes["id"] = recipes["id"].astype(str)
recipes["name"] = recipes["name"].fillna("Untitled Recipe").astype(str)
recipes["ingredients_text"] = recipes["ingredients_text"].fillna("").astype(str).str.lower()
recipes["tags_text"] = recipes["tags_text"].fillna("").astype(str).str.lower()
recipes["description"] = recipes["description"].fillna("").astype(str)
recipes["description_clean"] = recipes["description_clean"].fillna("").astype(str) if "description_clean" in recipes.columns else recipes["description"]
recipes["steps"] = recipes["steps"].fillna("").astype(str) if "steps" in recipes.columns else ""

# Ensure numeric columns are clean
numeric_cols = [
    "minutes_clean", "calories", "total_fat_pdv", "sugar_pdv",
    "sodium_pdv", "protein_pdv", "saturated_fat_pdv", "carbohydrates_pdv"
]
for col in numeric_cols:
    if col in recipes.columns:
        recipes[col] = pd.to_numeric(recipes[col], errors="coerce")

print(f"Recipes loaded: {len(recipes):,}")

# ============================================================================
# 3. LOAD TF-IDF MATRIX & VECTORIZER
# ============================================================================

tfidf_matrix = sp.load_npz(MATRIX_PATH)
tfidf_vectorizer = joblib.load(VECTORIZER_PATH)

print(f"TF-IDF matrix loaded: {tfidf_matrix.shape}")
print(f"TF-IDF vectorizer vocabulary: {len(tfidf_vectorizer.vocabulary_):,}")

# ============================================================================
# 4. LOAD ID MAPPING
# ============================================================================

with open(MAPPING_PATH, "r") as f:
    recipe_id_to_index = json.load(f)

# Normalize ID mapping keys to string
recipe_id_to_index = {str(k): int(v) for k, v in recipe_id_to_index.items()}
print(f"Recipe ID mappings: {len(recipe_id_to_index):,}")

# ============================================================================
# 5. LOAD NUTRITION DATASET & PRE-CALCULATE NUTRITION SCORE MAP
# ============================================================================

nutrition_data = pd.read_csv(NUTRITION_PATH, low_memory=False)
nutrition_score_dict = dict(
    zip(
        nutrition_data["id"].astype(str),
        pd.to_numeric(nutrition_data["nutrition_score"], errors="coerce")
    )
)

# Map precomputed nutrition score to all recipes; fallback to 0.65 for any unmapped
recipes["nutrition_score"] = recipes["id"].map(nutrition_score_dict).fillna(0.65).astype(float)
print(f"Nutrition dataset loaded: {nutrition_data.shape}, scores mapped: {len(recipes):,}")

# ============================================================================
# 6. DIETARY RESTRICTION DEFINITIONS
# ============================================================================

MEAT_WORDS = [
    "chicken", "beef", "pork", "lamb", "mutton", "turkey", "bacon",
    "ham", "sausage", "meat", "fish", "salmon", "tuna", "shrimp",
    "prawn", "crab", "lobster", "anchovy", "oyster", "clam", "duck", "veal"
]

ANIMAL_PRODUCT_WORDS = [
    "milk", "butter", "cheese", "cream", "yogurt", "egg", "honey", "ghee", "whey", "mayonnaise"
]

GLUTEN_WORDS = [
    "wheat", "flour", "bread", "pasta", "barley", "rye", "breadcrumbs", "semolina", "couscous"
]

DAIRY_WORDS = [
    "milk", "butter", "cheese", "cream", "yogurt", "ghee", "whey"
]


# ============================================================================
# 7. SEARCH RECIPES BY NAME
# ============================================================================

def search_recipe(search_text: str, limit: int = 10) -> pd.DataFrame:
    """Search recipes by substring match in title."""
    search_text = str(search_text).strip().lower()
    if not search_text:
        return recipes.head(0).copy()

    results = recipes[
        recipes["name"].str.lower().str.contains(search_text, regex=False, na=False)
    ]
    return results.head(limit).copy().reset_index(drop=True)


# ============================================================================
# 8. RECOMMEND SIMILAR RECIPES BY ID
# ============================================================================

def recommend_similar_recipes(recipe_id: str, top_n: int = 10) -> pd.DataFrame:
    """Find recipes similar to a given recipe ID using TF-IDF cosine similarity."""
    recipe_id = str(recipe_id)
    if recipe_id not in recipe_id_to_index:
        raise ValueError(f"Recipe ID {recipe_id} does not exist.")
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    recipe_index = recipe_id_to_index[recipe_id]
    similarity_scores = cosine_similarity(
        tfidf_matrix[recipe_index],
        tfidf_matrix
    ).flatten()

    similarity_scores[recipe_index] = -1
    recommended_indices = np.argsort(similarity_scores)[::-1][:top_n]

    recommendations = recipes.iloc[recommended_indices].copy()
    recommendations["similarity_score"] = similarity_scores[recommended_indices]
    recommendations = recommendations[recommendations["similarity_score"] >= 0]
    return recommendations.reset_index(drop=True)


# ============================================================================
# 9. FILTER RECIPES BY USER PROFILE CONSTRAINTS
# ============================================================================

def filter_recipes_by_profile(data: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """Filter recipes based on dietary restrictions, allergies, max calories, and max time."""
    mask = pd.Series(True, index=data.index)

    diet = profile.get("diet", "No restriction")
    if diet == "Vegetarian":
        diet_match = (
            data["tags_text"].str.contains("vegetarian", regex=False, na=False) |
            data["tags_text"].str.contains("vegan", regex=False, na=False)
        )
        for w in MEAT_WORDS:
            diet_match = diet_match & (~data["ingredients_text"].str.contains(w, regex=False, na=False))
        mask = mask & diet_match

    elif diet == "Vegan":
        diet_match = data["tags_text"].str.contains("vegan", regex=False, na=False)
        for w in MEAT_WORDS + ANIMAL_PRODUCT_WORDS:
            diet_match = diet_match & (~data["ingredients_text"].str.contains(w, regex=False, na=False))
        mask = mask & diet_match

    elif diet == "Gluten-Free":
        diet_match = data["tags_text"].str.contains("gluten-free", regex=False, na=False)
        for w in GLUTEN_WORDS:
            diet_match = diet_match & (~data["ingredients_text"].str.contains(w, regex=False, na=False))
        mask = mask & diet_match

    elif diet == "Dairy-Free":
        diet_match = data["tags_text"].str.contains("dairy-free", regex=False, na=False)
        for w in DAIRY_WORDS:
            diet_match = diet_match & (~data["ingredients_text"].str.contains(w, regex=False, na=False))
        mask = mask & diet_match

    elif diet == "Low-Carb":
        carb_match = (data["carbohydrates_pdv"].fillna(100) <= 15) | data["tags_text"].str.contains("low-carb", regex=False, na=False)
        mask = mask & carb_match

    # Exclude avoided ingredients and allergies
    excluded = list(set([
        item.strip().lower() for item in profile.get("avoid", []) + profile.get("allergies", [])
        if item.strip()
    ]))
    for exc in excluded:
        mask = mask & (~data["ingredients_text"].str.contains(exc, regex=False, na=False))

    # Preparation time filter
    max_time = profile.get("max_time", 0)
    if max_time and max_time > 0:
        mask = mask & data["minutes_clean"].notna() & (data["minutes_clean"] <= max_time)

    # Maximum calories filter
    max_calories = profile.get("max_calories", 0)
    if max_calories and max_calories > 0:
        mask = mask & data["calories"].notna() & (data["calories"] <= max_calories)

    return data[mask].copy()


# ============================================================================
# 10. GENERATE EXPLANATION FOR A RECOMMENDATION
# ============================================================================

def generate_recommendation_reason(row: pd.Series, profile: dict) -> list[str]:
    """Generate concise, user-friendly bullet points explaining why this recipe was selected."""
    reasons = []

    diet = profile.get("diet", "No restriction")
    if diet != "No restriction":
        reasons.append(f"Complies with your {diet} dietary preference.")

    # Matched preferred ingredients
    preferred = [p.strip().lower() for p in profile.get("preferred", []) if p.strip()]
    if preferred:
        matched = [p for p in preferred if p in str(row.get("ingredients_text", ""))]
        if matched:
            reasons.append(f"Includes preferred ingredients: {', '.join(matched)}.")

    # Calorie compliance
    cal = row.get("calories", 0)
    max_cal = profile.get("max_calories", 600)
    if pd.notna(cal) and max_cal:
        if cal <= max_cal:
            reasons.append(f"Under your calorie limit ({cal:.0f} kcal / target: <= {max_cal:.0f} kcal).")

    # Time compliance
    mins = row.get("minutes_clean", 0)
    max_time = profile.get("max_time", 60)
    if pd.notna(mins) and max_time:
        if mins <= max_time:
            reasons.append(f"Ready in {mins:.0f} minutes (limit: <= {max_time:.0f} min).")

    # Nutrition score
    nutr = row.get("nutrition_score", 0.65)
    if pd.notna(nutr):
        reasons.append(f"High nutritional balance score of {nutr * 100:.1f}%.")

    return reasons


# ============================================================================
# 11. HYBRID RECOMMENDATION ENGINE
# ============================================================================

def recommend_hybrid(
    profile: dict,
    query_text: str = None,
    seed_recipe_id: str = None,
    top_n: int = 10,
    weights: tuple = (0.3, 0.3, 0.4)
) -> pd.DataFrame:
    """
    Generate ranked recipe recommendations using the hybrid formula:
      hybrid_score = 0.3 * content_score + 0.3 * preference_score + 0.4 * nutrition_score
    """
    w_content, w_pref, w_nutr = weights

    # Step 1: Filter candidate recipes using profile restrictions
    candidates = filter_recipes_by_profile(recipes, profile)

    if candidates.empty:
        return pd.DataFrame()

    cand_indices = [recipe_id_to_index[rid] for rid in candidates["id"] if rid in recipe_id_to_index]
    if not cand_indices:
        return pd.DataFrame()

    cand_subset = candidates[candidates["id"].isin(recipe_id_to_index)].copy()

    # Step 2: Calculate Content Similarity Score
    if seed_recipe_id and str(seed_recipe_id) in recipe_id_to_index:
        seed_idx = recipe_id_to_index[str(seed_recipe_id)]
        seed_vec = tfidf_matrix[seed_idx]
        sim_scores = seed_vec.dot(tfidf_matrix.T).toarray().flatten()
    elif query_text and query_text.strip():
        q_vec = tfidf_vectorizer.transform([query_text.strip().lower()])
        sim_scores = q_vec.dot(tfidf_matrix.T).toarray().flatten()
    elif profile.get("preferred"):
        preferred_str = " ".join(profile.get("preferred", []))
        q_vec = tfidf_vectorizer.transform([preferred_str])
        sim_scores = q_vec.dot(tfidf_matrix.T).toarray().flatten()
    else:
        # Default baseline content score
        sim_scores = np.full(tfidf_matrix.shape[0], 0.5, dtype=np.float32)

    # Map content scores to candidate subset
    cand_subset["content_score"] = [sim_scores[recipe_id_to_index[rid]] for rid in cand_subset["id"]]

    # Normalize content_score to [0, 1] if max > min
    max_c = cand_subset["content_score"].max()
    min_c = cand_subset["content_score"].min()
    if max_c > min_c:
        cand_subset["content_score"] = (cand_subset["content_score"] - min_c) / (max_c - min_c)
    else:
        cand_subset["content_score"] = 0.5

    # Step 3: Calculate Ingredient Preference Score
    preferred = [p.strip().lower() for p in profile.get("preferred", []) if p.strip()]
    if preferred:
        matches = np.zeros(len(cand_subset), dtype=np.float32)
        for p in preferred:
            matches += cand_subset["ingredients_text"].str.contains(p, regex=False, na=False).to_numpy()
        cand_subset["preference_score"] = matches / len(preferred)
    else:
        cand_subset["preference_score"] = 0.5

    # Step 4: Nutrition Score (Pre-calculated / mapped)
    cand_subset["nutrition_score"] = cand_subset["nutrition_score"].fillna(0.65).astype(float)

    # Step 5: Hybrid Score Formulation
    cand_subset["hybrid_score"] = (
        w_content * cand_subset["content_score"] +
        w_pref * cand_subset["preference_score"] +
        w_nutr * cand_subset["nutrition_score"]
    )

    # Exclude seed recipe if provided
    if seed_recipe_id:
        cand_subset = cand_subset[cand_subset["id"] != str(seed_recipe_id)]

    # Step 6: Rank and format results
    top_results = cand_subset.sort_values(by="hybrid_score", ascending=False).head(top_n).copy()

    # Generate explanations
    top_results["explanation"] = [
        generate_recommendation_reason(row, profile)
        for _, row in top_results.iterrows()
    ]

    return top_results.reset_index(drop=True)
