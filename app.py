import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import ast
import io
import datetime
from PIL import Image

from recommendation_engine import (
    recipes,
    nutrition_data,
    recommend_hybrid,
    search_recipe,
    recommend_similar_recipes
)
from food_image_analyzer import (
    predict_food,
    estimate_recipe_nutrition,
    get_nutrition_explanation
)
from nutrition_lookup import (
    search_food,
    get_nutrients_per_100g,
    calculate_serving_nutrition
)
from database.db_manager import (
    init_db,
    create_user,
    authenticate_user,
    save_food_profile,
    get_food_profile,
    add_meal_log,
    get_meal_logs,
    delete_meal_log,
    save_recipe_feedback,
    get_user_saved_recipes
)

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="NutriFood | Smart Nutrition Assistant",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database schema automatically
init_db()

# --------------------------------------------------
# 2. CUSTOM CSS - HIGH-CONTRAST EMERALD THEME
# --------------------------------------------------

st.markdown("""
<style>

/* Root application and typography */
.stApp {
    background-color: #F8FAF9 !important;
    color: #1A2E22 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Universal high-contrast text color defaults */
p, span, label, li, h1, h2, h3, h4, h5, h6 {
    color: #1A2E22 !important;
}

.stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown strong {
    color: #1A2E22 !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #F0F5F2 !important;
    border-right: 1px solid #D2DFD7 !important;
}

[data-testid="stSidebar"] * {
    color: #143522 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #143522 !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label span {
    color: #143522 !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    color: #2F4E3C !important;
    font-weight: 500 !important;
}

/* Headings */
.main-title {
    font-size: 34px !important;
    font-weight: 800 !important;
    color: #0A4325 !important;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
}

.sub-title {
    color: #2B4738 !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    margin-bottom: 20px;
}

.section-title {
    font-size: 22px !important;
    font-weight: 750 !important;
    color: #0F522E !important;
    margin-top: 18px;
    margin-bottom: 12px;
}

/* Metrics and Cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1.5px solid #D8E5DD !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    box-shadow: 0 2px 8px rgba(16, 80, 50, 0.06) !important;
}

[data-testid="stMetricLabel"] * {
    color: #264A35 !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

[data-testid="stMetricValue"] * {
    color: #0A4325 !important;
    font-size: 28px !important;
    font-weight: 800 !important;
}

/* Feature & Recommendation Cards */
.feature-card {
    background-color: #FFFFFF;
    border: 1.5px solid #DCE6E0;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(16, 80, 50, 0.05);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.feature-card:hover {
    border-color: #16734A;
    box-shadow: 0 4px 14px rgba(22, 115, 74, 0.12);
}

.recipe-card-title {
    font-size: 20px;
    font-weight: 750;
    color: #0B4A28 !important;
    margin-bottom: 4px;
}

/* Score Badges */
.badge-pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 16px;
    font-size: 12px;
    font-weight: 700;
    margin-right: 6px;
    margin-bottom: 6px;
}

.badge-hybrid {
    background-color: #E6F4EA;
    color: #0D652D !important;
    border: 1px solid #A8DAB5;
}

.badge-content {
    background-color: #E8F0FE;
    color: #1A73E8 !important;
    border: 1px solid #AECBFA;
}

.badge-pref {
    background-color: #FEF7E0;
    color: #B06000 !important;
    border: 1px solid #FDD663;
}

.badge-nutr {
    background-color: #FCE8E6;
    color: #C5221F !important;
    border: 1px solid #F5A9A4;
}

.badge-meal {
    background-color: #E8F5E9;
    color: #1B5E20 !important;
    border: 1px solid #81C784;
}

/* Reason Box */
.reason-box {
    background-color: #F4F9F5;
    border-left: 4px solid #16734A;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 10px 0;
    font-size: 13.5px;
}

/* Form Inputs */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    background-color: #FFFFFF !important;
    color: #1A2E22 !important;
    border: 1.5px solid #CCD8D0 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #16734A !important;
    box-shadow: 0 0 0 2px rgba(22, 115, 74, 0.2) !important;
}

/* Selectbox */
[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1.5px solid #CCD8D0 !important;
    border-radius: 8px !important;
}

[data-baseweb="select"] * {
    color: #1A2E22 !important;
    font-weight: 600 !important;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #16734A, #105A38) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 9px 20px !important;
    box-shadow: 0 2px 8px rgba(16, 90, 56, 0.25) !important;
    transition: all 0.2s ease !important;
}

.stButton > button *,
.stFormSubmitButton > button * {
    color: #FFFFFF !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #105A38, #0B4228) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 12px rgba(16, 90, 56, 0.35) !important;
    transform: translateY(-1px);
}

/* Expanders */
.streamlit-expanderHeader {
    background-color: #FFFFFF !important;
    border: 1.5px solid #D8E5DD !important;
    border-radius: 10px !important;
}

.streamlit-expanderHeader * {
    color: #0B5731 !important;
    font-weight: 700 !important;
}

.streamlit-expanderContent {
    background-color: #FFFFFF !important;
    border: 1.5px solid #D8E5DD !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    padding: 18px !important;
}

.streamlit-expanderContent * {
    color: #1A2E22 !important;
}

/* Alerts and Information Banners */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: 1.5px solid #CCE0D4 !important;
    background-color: #F2F8F4 !important;
}

[data-testid="stAlert"] * {
    color: #123B23 !important;
}

/* Captions & Footers */
.stCaption, small, .small-note {
    color: #385946 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* Dividers */
hr {
    border-color: #D2DFD7 !important;
    margin: 22px 0 !important;
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
    border: 2px dashed #9EBEAA !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

[data-testid="stFileUploader"] * {
    color: #143522 !important;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. LOAD DATASET CACHE
# --------------------------------------------------

@st.cache_data
def prepare_data():
    data = recipes.copy()
    data["name"] = data["name"].fillna("Untitled Recipe").astype(str)
    data["ingredients_text"] = data["ingredients_text"].fillna("").astype(str).str.lower()
    data["tags_text"] = data["tags_text"].fillna("").astype(str).str.lower()
    data["description"] = data["description"].fillna("").astype(str)
    data["description_clean"] = data["description_clean"].fillna("").astype(str) if "description_clean" in data.columns else data["description"]
    data["steps"] = data["steps"].fillna("").astype(str) if "steps" in data.columns else ""
    data["minutes_clean"] = pd.to_numeric(data["minutes_clean"], errors="coerce")

    numeric_columns = [
        "calories", "total_fat_pdv", "sugar_pdv", "sodium_pdv",
        "protein_pdv", "saturated_fat_pdv", "carbohydrates_pdv", "nutrition_score"
    ]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    return data

df = prepare_data()

# --------------------------------------------------
# 4. HELPER FUNCTIONS
# --------------------------------------------------

def parse_terms(text):
    """Convert comma-separated user input into a clean list of lowercase strings."""
    if not text:
        return []
    return [item.strip().lower() for item in text.split(",") if item.strip()]

def format_steps(steps_raw):
    """Parse raw string or list into structured recipe instruction steps."""
    if not steps_raw or pd.isna(steps_raw):
        return ["Instructions not specified."]
    try:
        if isinstance(steps_raw, str) and steps_raw.strip().startswith("[") and steps_raw.strip().endswith("]"):
            steps_list = ast.literal_eval(steps_raw)
            if isinstance(steps_list, list) and len(steps_list) > 0:
                return [str(s).strip() for s in steps_list if str(s).strip()]
    except Exception:
        pass
    if isinstance(steps_raw, str):
        # Split on sentence boundaries if literal list evaluation was not applicable
        sentences = [s.strip() for s in re.split(r'\. |\n', steps_raw) if len(s.strip()) > 3]
        return sentences if sentences else [steps_raw]
    return [str(steps_raw)]

def format_ingredients(ing_raw):
    """Parse raw string or list into clean ingredient items."""
    if not ing_raw or pd.isna(ing_raw):
        return ["Ingredients not listed."]
    try:
        if isinstance(ing_raw, str) and ing_raw.strip().startswith("[") and ing_raw.strip().endswith("]"):
            ing_list = ast.literal_eval(ing_raw)
            if isinstance(ing_list, list) and len(ing_list) > 0:
                return [str(item).strip() for item in ing_list if str(item).strip()]
    except Exception:
        pass
    if isinstance(ing_raw, str):
        return [item.strip() for item in ing_raw.split(",") if item.strip()]
    return [str(ing_raw)]

# --------------------------------------------------
# 5. USER SESSION & AUTHENTICATION STATE
# --------------------------------------------------

if "user" not in st.session_state:
    st.session_state.user = None

# If user is logged in, ensure profile is synced from SQLite
if st.session_state.user:
    active_user_id = st.session_state.user["id"]
    if "profile" not in st.session_state or st.session_state.get("last_synced_user") != active_user_id:
        st.session_state.profile = get_food_profile(active_user_id)
        st.session_state.last_synced_user = active_user_id
else:
    active_user_id = 1  # Default guest user ID in SQLite
    if "profile" not in st.session_state:
        st.session_state.profile = get_food_profile(1)

# Ensure all default profile keys are present
default_keys = {
    "diet": "No restriction",
    "preferred": [],
    "avoid": [],
    "allergies": [],
    "max_calories": 600.0,
    "max_time": 60,
    "daily_calorie_target": 2000.0
}
for k, v in default_keys.items():
    if k not in st.session_state.profile:
        st.session_state.profile[k] = v

# --------------------------------------------------
# 6. SIDEBAR NAVIGATION & USER AUTHENTICATION
# --------------------------------------------------

st.sidebar.markdown("# 🥗 NutriFood")
st.sidebar.caption("Personalized Nutrition & Dietary Assistant")

# Navigation Menu
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "My Food Profile",
        "Recipe Recommendations",
        "Food Image Analyzer",
        "Nutrition Analytics",
        "Meal Tracker"
    ],
    key="nav_selection_radio"
)

st.sidebar.divider()

# User Account Section in Sidebar
if st.session_state.user:
    full_name = st.session_state.user.get("full_name") or st.session_state.user["username"]
    st.sidebar.markdown(f"**👤 {full_name}**")
    st.sidebar.caption(f"Signed in as @{st.session_state.user['username']}")
    if st.sidebar.button("Sign Out", key="sidebar_sign_out_btn", use_container_width=True):
        st.session_state.user = None
        st.session_state.pop("profile", None)
        st.session_state.pop("last_synced_user", None)
        st.rerun()
else:
    with st.sidebar.expander("🔐 Account Login / Register", expanded=False):
        tab_login, tab_register = st.tabs(["Sign In", "Register"])
        
        with tab_login:
            login_user = st.text_input("Username or Email", key="sidebar_login_user")
            login_pass = st.text_input("Password", type="password", key="sidebar_login_pass")
            if st.button("Sign In", key="sidebar_login_submit", type="primary", use_container_width=True):
                user_record = authenticate_user(login_user, login_pass)
                if user_record:
                    st.session_state.user = user_record
                    st.session_state.profile = get_food_profile(user_record["id"])
                    st.session_state.last_synced_user = user_record["id"]
                    st.success(f"Welcome back, {user_record['username']}!")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")

        with tab_register:
            reg_name = st.text_input("Full Name", key="sidebar_reg_name")
            reg_user = st.text_input("Username", key="sidebar_reg_user")
            reg_email = st.text_input("Email", key="sidebar_reg_email")
            reg_pass = st.text_input("Password (min 6 chars)", type="password", key="sidebar_reg_pass")
            if st.button("Create Account", key="sidebar_reg_submit", type="primary", use_container_width=True):
                success, msg = create_user(reg_user, reg_email, reg_pass, reg_name)
                if success:
                    user_record = authenticate_user(reg_user, reg_pass)
                    st.session_state.user = user_record
                    st.session_state.profile = get_food_profile(user_record["id"])
                    st.session_state.last_synced_user = user_record["id"]
                    st.success("Account created successfully!")
                    st.rerun()
                else:
                    st.error(msg)

st.sidebar.divider()

# Sidebar Food Profile Summary
st.sidebar.markdown("### Your Food Profile")
st.sidebar.write(f"**Diet:** {st.session_state.profile.get('diet', 'No restriction')}")
st.sidebar.write(f"**Max Calories / Serving:** {st.session_state.profile.get('max_calories', 600):.0f} kcal")
st.sidebar.write(f"**Max Prep Time:** {st.session_state.profile.get('max_time', 60)} mins")
st.sidebar.write(f"**Daily Calorie Goal:** {st.session_state.profile.get('daily_calorie_target', 2000):.0f} kcal")


# --------------------------------------------------
# 7. HOME PAGE
# --------------------------------------------------

if page == "Home":

    st.markdown('<div class="main-title">Welcome to NutriFood 🥗</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        'Personalized Nutrition-Aware Food Recommendation and Dietary Analytics System'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Curated Recipes", f"{len(df):,}")
    with col2:
        st.metric("Avg Recipe Calories", f"{df['calories'].mean():.0f} kcal")
    with col3:
        st.metric("Median Prep Time", f"{df['minutes_clean'].median():.0f} mins")
    with col4:
        st.metric("Nutrition Graded", f"{len(nutrition_data):,}")

    st.divider()

    st.markdown('<div class="section-title">Core System Capabilities</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="feature-card">
            <h4>👤 My Food Profile</h4>
            <p>Configure dietary restrictions (Vegetarian, Vegan, Gluten-Free, Dairy-Free, Low-Carb), favorite ingredients, allergens to avoid, and personal daily calorie goals.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h4>🍲 Hybrid Recommendation Engine</h4>
            <p>State-of-the-art hybrid recommendation blending TF-IDF content similarity (30%), user ingredient preference scoring (30%), and balanced nutrition percentile scoring (40%).</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h4>📸 Food Image Analyzer</h4>
            <p>Upload food photographs to classify dishes using a Vision Transformer (ViT) deep learning model, linked directly to verified USDA FoodData Central nutritional facts.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Dietary & Dataset Analytics</h4>
            <p>Interactive Plotly dashboards for both personal daily nutrient tracking and macro balance, plus full statistical distributions across the Food.com recipe corpus.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h4>📅 Persistent Meal Tracker</h4>
            <p>Log daily meals across Breakfast, Lunch, Dinner, and Snacks. Automatically track daily calorie and macronutrient targets with permanent SQLite database storage.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h4>🔐 User Account Persistence</h4>
            <p>Secure salted PBKDF2 authentication ensures your custom food profile, bookmarked recipes, and historical meal logs persist across sessions and devices.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "💡 **Pair Programming Tip:** Get started by setting your dietary restrictions and favorite ingredients in **My Food Profile**, then explore custom matches in **Recipe Recommendations**!"
    )


# --------------------------------------------------
# 8. MY FOOD PROFILE PAGE
# --------------------------------------------------

elif page == "My Food Profile":

    st.markdown('<div class="main-title">My Food Profile 👤</div>', unsafe_allow_html=True)
    st.write("Customize your personal dietary preferences, ingredient choices, allergens, and nutritional goals.")

    current = st.session_state.profile

    diet_options = [
        "No restriction",
        "Vegetarian",
        "Vegan",
        "Gluten-Free",
        "Dairy-Free",
        "Low-Carb"
    ]
    current_diet_index = diet_options.index(current["diet"]) if current["diet"] in diet_options else 0

    with st.form("profile_form"):
        st.markdown("### 1. Dietary Restrictions")
        diet = st.selectbox(
            "Select your dietary lifestyle",
            diet_options,
            index=current_diet_index,
            key="profile_diet_select"
        )

        st.divider()

        st.markdown("### 2. Ingredient Preferences & Allergens")
        preferred_text = st.text_input(
            "Ingredients you love (comma-separated)",
            value=", ".join(current.get("preferred", [])),
            placeholder="Example: garlic, spinach, tomatoes, paneer, olive oil",
            key="profile_pref_input"
        )

        avoid_text = st.text_input(
            "Ingredients to avoid (comma-separated)",
            value=", ".join(current.get("avoid", [])),
            placeholder="Example: mushroom, cilantro, eggplant",
            key="profile_avoid_input"
        )

        allergy_text = st.text_input(
            "Allergies or sensitive ingredients (comma-separated)",
            value=", ".join(current.get("allergies", [])),
            placeholder="Example: peanuts, shellfish, sesame, tree nuts",
            key="profile_allergy_input"
        )

        st.caption(
            "⚠️ Keyword matching filters recipes against recipe text and tags. Always inspect recipe ingredients if you have severe clinical allergies."
        )

        st.divider()

        st.markdown("### 3. Nutrition & Preparation Limits")
        
        col_cal, col_time = st.columns(2)
        with col_cal:
            max_calories = st.slider(
                "Maximum calories per recipe serving (kcal)",
                min_value=100,
                max_value=1500,
                value=int(current.get("max_calories", 600)),
                step=50,
                key="profile_max_cal_slider"
            )

        with col_time:
            max_time = st.slider(
                "Maximum cooking & prep time (minutes)",
                min_value=10,
                max_value=240,
                value=int(current.get("max_time", 60)),
                step=5,
                format="%d min",
                key="profile_max_time_slider"
            )

        daily_target = st.slider(
            "Target daily calorie consumption (kcal/day)",
            min_value=1200,
            max_value=4000,
            value=int(current.get("daily_calorie_target", 2000)),
            step=50,
            key="profile_daily_target_slider"
        )

        save_btn = st.form_submit_button(
            "Save Food Profile",
            type="primary",
            use_container_width=True
        )

        if save_btn:
            new_profile = {
                "diet": diet,
                "preferred": parse_terms(preferred_text),
                "avoid": parse_terms(avoid_text),
                "allergies": parse_terms(allergy_text),
                "max_calories": float(max_calories),
                "max_time": int(max_time),
                "daily_calorie_target": float(daily_target)
            }
            st.session_state.profile = new_profile
            
            # Persist to SQLite
            save_food_profile(
                user_id=active_user_id,
                diet=diet,
                preferred=new_profile["preferred"],
                avoid=new_profile["avoid"],
                allergies=new_profile["allergies"],
                max_calories=float(max_calories),
                max_time=int(max_time),
                daily_calorie_target=float(daily_target)
            )

            if st.session_state.user:
                st.success("✓ Food profile saved permanently to your NutriFood account!")
            else:
                st.success("✓ Food profile saved! (Sign in to sync your profile across multiple devices)")

    st.divider()

    # Saved Recipes Bookmark Section
    st.markdown("### ⭐ Bookmarked Recipes")
    saved_items = get_user_saved_recipes(active_user_id)
    if saved_items:
        st.write(f"You have **{len(saved_items)}** bookmarked recipes:")
        for item in saved_items:
            c1, c2 = st.columns([4, 1])
            with c1:
                st.write(f"• **{item['recipe_name']}** (ID: `{item['recipe_id']}`)")
            with c2:
                st.caption(f"Saved: {item['created_at'][:10]}")
    else:
        st.info("No recipes bookmarked yet. When exploring recommendations, click '⭐ Bookmark Recipe' to save your favorites here!")


# --------------------------------------------------
# 9. RECIPE RECOMMENDATIONS (HYBRID ENGINE)
# --------------------------------------------------

elif page == "Recipe Recommendations":

    st.markdown('<div class="main-title">Recipe Recommendations 🍲</div>', unsafe_allow_html=True)
    st.write(
        "NutriFood uses a **Hybrid Recommendation Engine** combining TF-IDF content similarity, "
        "ingredient preference alignment, and nutritional balance percentile scores."
    )

    profile = st.session_state.profile

    # Search & Tuning Row
    col_search, col_count = st.columns([3, 1])
    with col_search:
        search_query = st.text_input(
            "Search for a dish, craving, or key ingredient",
            placeholder="Example: lentil soup, pasta, grilled chicken, curry",
            key="hybrid_search_query"
        )
    with col_count:
        top_k = st.selectbox(
            "Results count",
            [5, 10, 15, 20],
            index=1,
            key="hybrid_top_k"
        )

    # Advanced Hybrid Weights Tuning Expander
    with st.expander("⚙️ Advanced Model Weights Tuning", expanded=False):
        st.caption(
            "Adjust the relative influence of each signal. Default weights are mathematically optimized "
            "from model evaluation: 30% Content Similarity, 30% Ingredient Preference, 40% Nutritional Quality."
        )
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            w_content = st.slider("Content Match (TF-IDF)", 0.0, 1.0, 0.3, 0.05, key="w_content_slider")
        with col_w2:
            w_pref = st.slider("Ingredient Preference", 0.0, 1.0, 0.3, 0.05, key="w_pref_slider")
        with col_w3:
            w_nutr = st.slider("Nutrition Quality Score", 0.0, 1.0, 0.4, 0.05, key="w_nutr_slider")

        # Normalize weights so they sum to 1.0
        total_w = w_content + w_pref + w_nutr
        if total_w > 0:
            norm_weights = (w_content / total_w, w_pref / total_w, w_nutr / total_w)
        else:
            norm_weights = (0.3, 0.3, 0.4)

    # Compute Recommendations
    with st.spinner("Executing hybrid ranking algorithm across 231,637 recipes..."):
        recommendations = recommend_hybrid(
            profile=profile,
            query_text=search_query,
            top_n=top_k,
            weights=norm_weights
        )

    if recommendations.empty:
        st.warning(
            "No recipes match all your selected constraints. Try increasing your maximum calories or "
            "preparation time limits in **My Food Profile**, or easing ingredient restrictions."
        )
    else:
        st.metric(
            "Matching Recipes Ranked",
            f"{len(recommendations)} recommended recipes",
            f"Top score: {recommendations.iloc[0]['hybrid_score']:.1%}"
        )

        st.divider()

        for idx, recipe in recommendations.iterrows():
            rec_id = str(recipe["id"])
            rec_name = str(recipe["name"]).title()
            cal = recipe.get("calories", 0)
            mins = recipe.get("minutes_clean", 0)
            h_score = recipe.get("hybrid_score", 0)
            c_score = recipe.get("content_score", 0)
            p_score = recipe.get("preference_score", 0)
            n_score = recipe.get("nutrition_score", 0)
            reasons = recipe.get("explanation", [])

            # Recommendation Card Container
            with st.container():
                st.markdown(f"""
                <div class="feature-card">
                    <div class="recipe-card-title">🍽️ #{idx+1}. {rec_name}</div>
                    <div style="margin-top: 6px; margin-bottom: 10px;">
                        <span class="badge-pill badge-hybrid">⭐ Hybrid Score: {h_score:.1%}</span>
                        <span class="badge-pill badge-content">📄 Content Match: {c_score:.1%}</span>
                        <span class="badge-pill badge-pref">❤️ Preference: {p_score:.1%}</span>
                        <span class="badge-pill badge-nutr">🥗 Nutrition: {n_score:.1%}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("Calories", f"{cal:.0f} kcal" if pd.notna(cal) else "N/A")
                with col_m2:
                    st.metric("Prep Time", f"{mins:.0f} min" if pd.notna(mins) else "N/A")
                with col_m3:
                    prot = recipe.get("protein_pdv", 0)
                    st.metric("Protein PDV", f"{prot:.0f}%" if pd.notna(prot) else "N/A")
                with col_m4:
                    carb = recipe.get("carbohydrates_pdv", 0)
                    st.metric("Carbs PDV", f"{carb:.0f}%" if pd.notna(carb) else "N/A")

                # Recommendation Reason Box
                if reasons:
                    reason_html = "<strong>Why NutriFood recommends this:</strong><br>• " + "<br>• ".join(reasons)
                    st.markdown(f'<div class="reason-box">{reason_html}</div>', unsafe_allow_html=True)

                # Expandable details
                with st.expander(f"📖 View Recipe Details, Ingredients & Cooking Steps for '{rec_name}'"):
                    desc = recipe.get("description_clean") or recipe.get("description")
                    if desc and str(desc).strip():
                        st.markdown(f"**Description:** {str(desc).strip()}")

                    col_ing, col_steps = st.columns([1, 1.5])
                    
                    with col_ing:
                        st.markdown("**🛒 Ingredients**")
                        ing_list = format_ingredients(recipe.get("ingredients") or recipe.get("ingredients_text"))
                        for item in ing_list:
                            st.markdown(f"- {item}")

                    with col_steps:
                        st.markdown("**👩‍🍳 Preparation Steps**")
                        steps_list = format_steps(recipe.get("steps"))
                        for s_idx, step in enumerate(steps_list, 1):
                            st.markdown(f"**{s_idx}.** {step}")

                    st.markdown("---")
                    st.markdown("**Nutritional Breakdown (PDV = Percentage Daily Value)**")
                    nutr_cols = st.columns(6)
                    nutr_cols[0].metric("Fat PDV", f"{recipe.get('total_fat_pdv', 0):.0f}%")
                    nutr_cols[1].metric("Sat. Fat PDV", f"{recipe.get('saturated_fat_pdv', 0):.0f}%")
                    nutr_cols[2].metric("Sugar PDV", f"{recipe.get('sugar_pdv', 0):.0f}%")
                    nutr_cols[3].metric("Sodium PDV", f"{recipe.get('sodium_pdv', 0):.0f}%")
                    nutr_cols[4].metric("Protein PDV", f"{recipe.get('protein_pdv', 0):.0f}%")
                    nutr_cols[5].metric("Carbs PDV", f"{recipe.get('carbohydrates_pdv', 0):.0f}%")

                    # Action buttons: Bookmark & Log to Meal Tracker
                    c_act1, c_act2, c_act3 = st.columns([1.5, 1.5, 2])
                    with c_act1:
                        if st.button("⭐ Bookmark Recipe", key=f"bookmark_btn_{rec_id}_{idx}"):
                            save_recipe_feedback(active_user_id, rec_id, rec_name, "saved")
                            st.success(f"Saved '{rec_name}' to your bookmarks!")

                    with c_act2:
                        meal_choice = st.selectbox(
                            "Meal Type",
                            ["Breakfast", "Lunch", "Dinner", "Snack"],
                            key=f"meal_type_sel_{rec_id}_{idx}"
                        )
                    with c_act3:
                        if st.button("➕ Log to Meal Tracker", key=f"log_meal_btn_{rec_id}_{idx}", type="primary"):
                            today_str = datetime.date.today().isoformat()
                            add_meal_log(
                                user_id=active_user_id,
                                date_str=today_str,
                                meal_type=meal_choice,
                                food_name=rec_name,
                                serving_size_g=150.0,
                                calories=float(cal) if pd.notna(cal) else 0.0,
                                protein=float(recipe.get("protein_pdv", 0)) * 0.5,
                                carbs=float(recipe.get("carbohydrates_pdv", 0)) * 2.75,
                                fat=float(recipe.get("total_fat_pdv", 0)) * 0.78,
                                fiber=4.0,
                                sugars=float(recipe.get("sugar_pdv", 0)) * 0.5,
                                sodium=float(recipe.get("sodium_pdv", 0)) * 23.0
                            )
                            st.success(f"✓ Added '{rec_name}' to today's {meal_choice} in Meal Tracker!")

            st.write("")


# --------------------------------------------------
# 10. FOOD IMAGE ANALYZER
# --------------------------------------------------

elif page == "Food Image Analyzer":

    st.markdown('<div class="main-title">Food Image Analyzer 📸</div>', unsafe_allow_html=True)
    st.write(
        "Upload a photograph of your meal. The Vision Transformer (ViT) deep learning model "
        "recognizes the dish locally (using 0 API tokens), and NutriFood retrieves detailed nutritional "
        "facts from the verified USDA FoodData Central database or Food.com recipe corpus."
    )

    st.caption("ℹ️ Automatic image recognition runs 100% locally on your machine with zero token cost. All nutritional values are estimates.")

    uploaded_file = st.file_uploader(
        "Upload a food image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        key="food_image_uploader"
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            col_img1, col_img2 = st.columns([1, 1.2])
            with col_img1:
                try:
                    st.image(image, caption="Uploaded Photograph", use_container_width=True)
                except TypeError:
                    st.image(image, caption="Uploaded Photograph", use_column_width=True)

            with col_img2:
                if st.session_state.get("food_image_bytes") != image_bytes:
                    st.session_state.food_image_bytes = image_bytes
                    with st.spinner("AI model identifying dish in photograph..."):
                        try:
                            preds = predict_food(image)
                        except Exception as e:
                            preds = []
                        st.session_state.food_predictions = preds

                preds = st.session_state.get("food_predictions", [])
                if preds:
                    st.markdown("### Top AI Dish Recognitions")
                    for p in preds:
                        dish_label = str(p.get("food") or p.get("label", "Food Item")).replace('_', ' ').title()
                        conf = float(p.get("confidence") if p.get("confidence") is not None else p.get("score", 0.0))
                        st.write(f"**{dish_label}** — {conf:.1%}")
                        st.progress(min(max(conf, 0.0), 1.0))

                    top_dish = str(preds[0].get("food") or preds[0].get("label", "Healthy Food")).replace('_', ' ')
                else:
                    top_dish = "Healthy Food"

        except Exception as err:
            st.error("Unable to process uploaded image.")
            st.exception(err)
            top_dish = "Healthy Food"

        st.divider()
        st.markdown("### Nutritional Analysis & Verification")
        
        c_q1, c_q2 = st.columns([3, 1])
        with c_q1:
            search_term = st.text_input(
                "Dish Name / Search Term",
                value=top_dish,
                placeholder="Example: pizza, chicken curry, salad, lentil soup",
                key="usda_search_input"
            )
        with c_q2:
            st.write("")
            st.write("")
            do_search = st.button("OK - Analyze Nutrition", key="usda_search_btn", type="primary", use_container_width=True)

        if do_search or "usda_results" in st.session_state or "recipe_est" in st.session_state:
            if do_search:
                with st.spinner(f"Analyzing nutritional data for '{search_term}'..."):
                    # 1. USDA search
                    try:
                        results = search_food(search_term, page_size=5)
                        st.session_state.usda_results = results
                    except Exception as e:
                        st.error(f"Error fetching USDA records: {e}")
                        st.session_state.usda_results = []
                    
                    # 2. Recipe dataset estimate
                    try:
                        est_nutr, matched_recipes = estimate_recipe_nutrition(search_term, recipes, nutrition_data)
                        st.session_state.recipe_est = est_nutr
                        st.session_state.recipe_matched = matched_recipes
                    except Exception as e:
                        st.session_state.recipe_est = None
                        st.session_state.recipe_matched = pd.DataFrame()

            nutr_tab_usda, nutr_tab_recipe = st.tabs([
                "🔬 Verified USDA FoodData Central Record",
                "📖 Food.com Recipe Dataset Median Estimate"
            ])

            # TAB 1: USDA FoodData Central Record
            with nutr_tab_usda:
                results = st.session_state.get("usda_results", [])
                if results:
                    st.success(f"Found {len(results)} matching USDA food records.")
                    food_options = {f"{r.get('description', 'Unknown')} (ID: {r.get('fdcId')})": r for r in results}
                    selected_key = st.selectbox("Select specific USDA record", list(food_options.keys()), key="usda_food_picker")
                    selected_food = food_options[selected_key]

                    # Per 100g base nutrition
                    per_100g = get_nutrients_per_100g(selected_food)

                    serving_grams = st.slider(
                        "Portion / Serving Size (grams)",
                        min_value=25,
                        max_value=500,
                        value=100,
                        step=25,
                        key="serving_size_slider"
                    )

                    serving_nutr = calculate_serving_nutrition(per_100g, serving_grams)

                    col_n1, col_n2, col_n3, col_n4 = st.columns(4)
                    with col_n1:
                        st.metric("Calories", f"{serving_nutr.get('calories', 0):.1f} kcal")
                    with col_n2:
                        st.metric("Protein", f"{serving_nutr.get('protein', 0):.1f} g")
                    with col_n3:
                        st.metric("Carbohydrates", f"{serving_nutr.get('carbohydrates', 0):.1f} g")
                    with col_n4:
                        st.metric("Total Fat", f"{serving_nutr.get('fat', 0):.1f} g")

                    # Macronutrient Split Donut Chart
                    macros = {
                        "Protein": max(0.0, serving_nutr.get("protein", 0)),
                        "Carbohydrates": max(0.0, serving_nutr.get("carbohydrates", 0)),
                        "Fat": max(0.0, serving_nutr.get("fat", 0))
                    }
                    if sum(macros.values()) > 0:
                        fig_donut = px.pie(
                            names=list(macros.keys()),
                            values=list(macros.values()),
                            hole=0.5,
                            title=f"Macronutrient Distribution ({serving_grams}g serving)",
                            color_discrete_sequence=["#16734A", "#40916C", "#95D5B2"]
                        )
                        fig_donut.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
                        st.plotly_chart(fig_donut, use_container_width=True)

                    # Quick Log to Meal Tracker Card
                    st.divider()
                    st.markdown("### ➕ Log This Food to Your Meal Tracker")
                    col_log1, col_log2 = st.columns([2, 1])
                    with col_log1:
                        meal_category = st.selectbox(
                            "Meal Category",
                            ["Breakfast", "Lunch", "Dinner", "Snack"],
                            key="image_meal_cat_select"
                        )
                    with col_log2:
                        st.write("")
                        st.write("")
                        add_to_tracker = st.button("Add to Meal Tracker", key="add_image_food_tracker_btn", type="primary", use_container_width=True)

                    if add_to_tracker:
                        add_meal_log(
                            user_id=active_user_id,
                            date_str=datetime.date.today().isoformat(),
                            meal_type=meal_category,
                            food_name=selected_food.get("description", search_term).title(),
                            serving_size_g=float(serving_grams),
                            calories=float(serving_nutr.get("calories", 0)),
                            protein=float(serving_nutr.get("protein", 0)),
                            carbs=float(serving_nutr.get("carbohydrates", 0)),
                            fat=float(serving_nutr.get("fat", 0)),
                            fiber=float(serving_nutr.get("fiber", 0)),
                            sugars=float(serving_nutr.get("sugars", 0)),
                            sodium=float(serving_nutr.get("sodium", 0))
                        )
                        st.success(f"✓ Successfully added {serving_grams}g '{selected_food.get('description', search_term)}' to today's {meal_category}!")
                else:
                    st.info("No USDA results found. You can switch to the Recipe Dataset tab or try other keywords above.")

            # TAB 2: Food.com Recipe Dataset Estimate
            with nutr_tab_recipe:
                est = st.session_state.get("recipe_est")
                matched_df = st.session_state.get("recipe_matched", pd.DataFrame())
                if est is not None and not matched_df.empty:
                    st.write(f"Estimated from median values of **{len(matched_df)}** similar recipes in the Food.com dataset:")
                    
                    c_r1, c_r2, c_r3, c_r4 = st.columns(4)
                    c_r1.metric("Est. Calories", f"{est.get('calories', 0):.0f} kcal")
                    c_r2.metric("Protein PDV", f"{est.get('protein_pdv', 0):.0f}%")
                    c_r3.metric("Carbs PDV", f"{est.get('carbohydrates_pdv', 0):.0f}%")
                    c_r4.metric("Fat PDV", f"{est.get('total_fat_pdv', 0):.0f}%")

                    with st.expander(f"View {len(matched_df)} matched Food.com recipes"):
                        st.dataframe(matched_df[["id", "name", "minutes_clean", "calories"]], use_container_width=True, hide_index=True)

                    # Quick Log to Meal Tracker from Recipe Estimate
                    col_rl1, col_rl2 = st.columns([2, 1])
                    with col_rl1:
                        meal_rec_cat = st.selectbox(
                            "Meal Category (Recipe Estimate)",
                            ["Breakfast", "Lunch", "Dinner", "Snack"],
                            key="recipe_est_meal_cat_select"
                        )
                    with col_rl2:
                        st.write("")
                        st.write("")
                        add_rec_tracker = st.button("Add Estimate to Meal Tracker", key="add_rec_food_tracker_btn", type="primary", use_container_width=True)

                    if add_rec_tracker:
                        add_meal_log(
                            user_id=active_user_id,
                            date_str=datetime.date.today().isoformat(),
                            meal_type=meal_rec_cat,
                            food_name=search_term.title(),
                            serving_size_g=150.0,
                            calories=float(est.get("calories", 0)),
                            protein=float(est.get("protein_pdv", 0)) * 0.5,
                            carbs=float(est.get("carbohydrates_pdv", 0)) * 2.75,
                            fat=float(est.get("total_fat_pdv", 0)) * 0.78
                        )
                        st.success(f"✓ Successfully added estimated '{search_term.title()}' to today's {meal_rec_cat}!")
                else:
                    st.info("No matching recipes found in Food.com dataset for this query.")



# --------------------------------------------------
# 11. NUTRITION ANALYTICS
# --------------------------------------------------

elif page == "Nutrition Analytics":

    st.markdown('<div class="main-title">Nutrition Analytics Dashboard 📊</div>', unsafe_allow_html=True)
    st.write("Explore personal dietary intake trends and deep statistical distributions across the Food.com recipe corpus.")

    analytics_tab1, analytics_tab2 = st.tabs(["👤 My Dietary Intake Dashboard", "📚 Food.com Dataset Analytics"])

    # TAB 1: PERSONAL INTAKE DASHBOARD
    with analytics_tab1:
        st.markdown("### Personal Nutrition & Intake Tracking")
        
        # Date filter
        intake_filter = st.radio(
            "Timeframe",
            ["Today", "Last 7 Days", "All Logged Meals"],
            horizontal=True,
            key="intake_timeframe_radio"
        )

        all_logs = get_meal_logs(active_user_id)
        today_str = datetime.date.today().isoformat()

        if all_logs:
            logs_df = pd.DataFrame(all_logs)
            logs_df["date"] = pd.to_datetime(logs_df["date"])

            if intake_filter == "Today":
                filtered_logs = logs_df[logs_df["date"].dt.strftime("%Y-%m-%d") == today_str]
            elif intake_filter == "Last 7 Days":
                week_ago = pd.Timestamp.now() - pd.Timedelta(days=7)
                filtered_logs = logs_df[logs_df["date"] >= week_ago]
            else:
                filtered_logs = logs_df
        else:
            filtered_logs = pd.DataFrame()

        daily_target = st.session_state.profile.get("daily_calorie_target", 2000.0)

        if not filtered_logs.empty:
            total_cal = filtered_logs["calories"].sum()
            total_prot = filtered_logs["protein"].sum()
            total_carbs = filtered_logs["carbs"].sum()
            total_fat = filtered_logs["fat"].sum()
            total_fiber = filtered_logs["fiber"].sum()
            total_sodium = filtered_logs["sodium"].sum()
            total_sugars = filtered_logs["sugars"].sum() if "sugars" in filtered_logs.columns else 0.0
            meal_count = len(filtered_logs)

            # Daily Calorie Target Progress
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Total Calories", f"{total_cal:.0f} kcal")
            with col_m2:
                st.metric("Daily Target", f"{daily_target:.0f} kcal")
            with col_m3:
                rem_cal = daily_target - total_cal
                st.metric(
                    "Calorie Balance",
                    f"{abs(rem_cal):.0f} kcal",
                    f"{'Remaining' if rem_cal >= 0 else 'Surplus'}"
                )
            with col_m4:
                st.metric("Meals Logged", f"{meal_count}")

            # Calorie progress bar
            pct = min(1.0, max(0.0, total_cal / daily_target))
            st.progress(pct)
            if total_cal > daily_target:
                st.warning(f"⚠️ You have exceeded your daily calorie target by {total_cal - daily_target:.0f} kcal.")
            else:
                st.success(f"✓ You have {daily_target - total_cal:.0f} kcal remaining today.")

            st.divider()

            col_c1, col_c2 = st.columns(2)

            with col_c1:
                # Macronutrient Donut Chart
                macro_dict = {"Protein (g)": total_prot, "Carbohydrates (g)": total_carbs, "Fat (g)": total_fat}
                fig_m = px.pie(
                    names=list(macro_dict.keys()),
                    values=list(macro_dict.values()),
                    hole=0.45,
                    title="Macronutrient Proportion (Grams)",
                    color_discrete_sequence=["#16734A", "#40916C", "#95D5B2"]
                )
                fig_m.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
                st.plotly_chart(fig_m, use_container_width=True)

            with col_c2:
                # Meal Type Calorie Breakdown
                meal_breakdown = filtered_logs.groupby("meal_type")["calories"].sum().reset_index()
                fig_type = px.bar(
                    meal_breakdown,
                    x="meal_type",
                    y="calories",
                    title="Calories by Meal Category",
                    labels={"meal_type": "Meal Type", "calories": "Calories (kcal)"},
                    color="meal_type",
                    color_discrete_sequence=["#16734A", "#2D6A4F", "#52B788", "#74C69D"]
                )
                fig_type.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", showlegend=False)
                st.plotly_chart(fig_type, use_container_width=True)

            # 7-Day Trend Chart
            st.markdown("### 7-Day Daily Calorie Intake vs. Target Goal")
            daily_intake = logs_df.groupby(logs_df["date"].dt.strftime("%Y-%m-%d"))["calories"].sum().reset_index()
            daily_intake.columns = ["Date", "Calories"]

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Bar(
                x=daily_intake["Date"],
                y=daily_intake["Calories"],
                name="Daily Calories",
                marker_color="#16734A"
            ))
            fig_trend.add_trace(go.Scatter(
                x=daily_intake["Date"],
                y=[daily_target] * len(daily_intake),
                name="Daily Target",
                line=dict(color="#C5221F", width=2, dash="dash")
            ))
            fig_trend.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                title="Daily Calories Logged vs. Target",
                xaxis_title="Date",
                yaxis_title="Calories (kcal)"
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            st.markdown("### Micronutrient Consumption")
            col_u1, col_u2, col_u3 = st.columns(3)
            with col_u1:
                st.metric("Fiber", f"{total_fiber:.1f} g", "Target: ≥ 28g")
            with col_u2:
                st.metric("Sugars", f"{total_sugars:.1f} g", "Limit: <= 50g")
            with col_u3:
                st.metric("Sodium", f"{total_sodium:.0f} mg", "Limit: ≤ 2,300mg")

        else:
            st.info("No meals logged for this timeframe. Go to the **Meal Tracker** tab or click '➕ Log to Meal Tracker' on recommended recipes!")

    # TAB 2: DATASET ANALYTICS
    with analytics_tab2:
        st.markdown("### Food.com Recipe Corpus Visualizations (231,637 Recipes)")

        nutrient_options = {
            "calories": "Calories (kcal)",
            "total_fat_pdv": "Total Fat (% Daily Value)",
            "sugar_pdv": "Sugar (% Daily Value)",
            "sodium_pdv": "Sodium (% Daily Value)",
            "protein_pdv": "Protein (% Daily Value)",
            "saturated_fat_pdv": "Saturated Fat (% Daily Value)",
            "carbohydrates_pdv": "Carbohydrates (% Daily Value)"
        }
        available_nutrients = [k for k in nutrient_options.keys() if k in df.columns]

        tab_dist, tab_macro, tab_scatter, tab_time = st.tabs([
            "📈 Nutrient Distribution",
            "🥗 Macronutrient Balance",
            "🔬 Correlation & Scatter",
            "⏱️ Time vs. Nutrition"
        ])

        with tab_dist:
            selected_n = st.selectbox("Select Nutrient", available_nutrients, format_func=lambda x: nutrient_options[x], key="analytics_dist_select")
            limit = float(df[selected_n].quantile(0.98))
            plot_df = df[df[selected_n].between(0, limit)]

            fig_hist = px.histogram(
                plot_df,
                x=selected_n,
                nbins=50,
                marginal="box",
                title=f"Distribution of {nutrient_options[selected_n]} (Excluding top 2% outliers)",
                labels={selected_n: nutrient_options[selected_n]},
                color_discrete_sequence=["#16734A"]
            )
            fig_hist.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
            st.plotly_chart(fig_hist, use_container_width=True)

        with tab_macro:
            macro_cols = ["protein_pdv", "total_fat_pdv", "carbohydrates_pdv"]
            macro_sample = df[macro_cols].dropna()
            for col in macro_cols:
                macro_sample = macro_sample[macro_sample[col] <= float(macro_sample[col].quantile(0.95))]

            fig_box = px.box(
                macro_sample,
                y=macro_cols,
                title="Macronutrient PDV Distribution Comparison",
                labels={"variable": "Macronutrient", "value": "Percentage of Daily Value (PDV)"},
                color_discrete_sequence=["#16734A", "#40916C", "#74C69D"]
            )
            fig_box.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
            st.plotly_chart(fig_box, use_container_width=True)

        with tab_scatter:
            col_x, col_y = st.columns(2)
            with col_x:
                x_n = st.selectbox("X-Axis Nutrient", available_nutrients, index=0, format_func=lambda x: nutrient_options[x], key="scatter_x")
            with col_y:
                y_n = st.selectbox("Y-Axis Nutrient", available_nutrients, index=4, format_func=lambda x: nutrient_options[x], key="scatter_y")

            sample_cols = list(dict.fromkeys([x_n, y_n, "name", "calories"]))
            sample_df = df[sample_cols].dropna().sample(n=min(3000, len(df)), random_state=42).reset_index(drop=True)
            
            x_limit = float(sample_df[x_n].quantile(0.98))
            y_limit = float(sample_df[y_n].quantile(0.98))
            sample_df = sample_df[(sample_df[x_n] <= x_limit) & (sample_df[y_n] <= y_limit)].reset_index(drop=True)

            fig_sc = px.scatter(
                sample_df,
                x=x_n,
                y=y_n,
                hover_name="name",
                color="calories",
                color_continuous_scale="Mint",
                labels={x_n: nutrient_options[x_n], y_n: nutrient_options[y_n], "calories": "Calories (kcal)"},
                title=f"{nutrient_options[x_n]} vs. {nutrient_options[y_n]}"
            )
            fig_sc.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
            st.plotly_chart(fig_sc, use_container_width=True)

        with tab_time:
            if "time_category" in df.columns:
                t_order = ["Quick", "Medium", "Long"]
                t_df = df[df["time_category"].isin(t_order)].groupby("time_category")[available_nutrients].median().reindex(t_order).reset_index()
                fig_t = px.bar(
                    t_df,
                    x="time_category",
                    y=["calories", "protein_pdv", "total_fat_pdv"],
                    barmode="group",
                    title="Median Nutritional Intake by Preparation Time Category",
                    labels={"time_category": "Preparation Time", "value": "Median Value", "variable": "Nutrient"},
                    color_discrete_sequence=["#16734A", "#40916C", "#74C69D"]
                )
                fig_t.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
                st.plotly_chart(fig_t, use_container_width=True)
                st.caption("Quick: ≤ 30 min | Medium: 30–60 min | Long: > 60 min")
            else:
                st.info("Time category metadata is not available.")

        st.divider()
        st.markdown("### Recipe Corpus Summary Statistics")
        stats_df = df[available_nutrients].describe().T.reset_index()
        stats_df["Nutrient"] = stats_df["index"].map(lambda k: nutrient_options.get(k, k))
        st.dataframe(stats_df, use_container_width=True, hide_index=True)


# --------------------------------------------------
# 12. MEAL TRACKER
# --------------------------------------------------

elif page == "Meal Tracker":

    st.markdown('<div class="main-title">Meal Tracker 📅</div>', unsafe_allow_html=True)
    st.write("Track your daily meals, log calories and macronutrients, and monitor your dietary progress in SQLite.")

    # Date Picker Row
    col_d1, col_d2 = st.columns([2, 2])
    with col_d1:
        selected_date = st.date_input("Select Date", datetime.date.today(), key="tracker_date_picker")
    date_str = selected_date.isoformat()

    daily_goal = st.session_state.profile.get("daily_calorie_target", 2000.0)

    # Fetch meals for chosen date
    day_meals = get_meal_logs(active_user_id, date_str)
    day_calories = sum(m["calories"] for m in day_meals)

    with col_d2:
        rem = daily_goal - day_calories
        st.metric(
            f"Calories Logged for {date_str}",
            f"{day_calories:.0f} / {daily_goal:.0f} kcal",
            f"{rem:.0f} kcal remaining" if rem >= 0 else f"{abs(rem):.0f} kcal surplus"
        )

    # Progress bar
    progress_val = min(1.0, max(0.0, day_calories / daily_goal)) if daily_goal > 0 else 0.0
    st.progress(progress_val)

    st.divider()

    # Log a Meal Form
    with st.expander("➕ Log a New Meal / Food Item", expanded=False):
        with st.form("add_meal_form"):
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                food_name = st.text_input("Food or Meal Name", placeholder="Example: Grilled salmon with brown rice", key="form_food_name")
                meal_type = st.selectbox("Meal Category", ["Breakfast", "Lunch", "Dinner", "Snack"], key="form_meal_type")
                serving_g = st.number_input("Serving Size (grams)", min_value=10.0, max_value=2000.0, value=150.0, step=10.0, key="form_serving")
            with col_f2:
                meal_cal = st.number_input("Calories (kcal)", min_value=0.0, max_value=5000.0, value=350.0, step=10.0, key="form_calories")
                prot_g = st.number_input("Protein (g)", min_value=0.0, max_value=300.0, value=25.0, step=1.0, key="form_protein")
                carbs_g = st.number_input("Carbohydrates (g)", min_value=0.0, max_value=500.0, value=35.0, step=1.0, key="form_carbs")
                fat_g = st.number_input("Total Fat (g)", min_value=0.0, max_value=300.0, value=10.0, step=1.0, key="form_fat")

            submit_meal = st.form_submit_button("Add Meal to Log", type="primary", use_container_width=True)

            if submit_meal:
                if food_name.strip():
                    add_meal_log(
                        user_id=active_user_id,
                        date_str=date_str,
                        meal_type=meal_type,
                        food_name=food_name.strip(),
                        serving_size_g=serving_g,
                        calories=meal_cal,
                        protein=prot_g,
                        carbs=carbs_g,
                        fat=fat_g
                    )
                    st.success(f"✓ Added '{food_name}' to {meal_type} on {date_str}!")
                    st.rerun()
                else:
                    st.error("Please enter a food or meal name.")

    # Display Logged Meals for Selected Date
    st.markdown(f"### Meals Recorded on {date_str}")
    if day_meals:
        for m in day_meals:
            m_id = m["id"]
            with st.container():
                c1, c2, c3, c4, c5, c6 = st.columns([3, 1.5, 1.5, 1.5, 1.5, 1])
                with c1:
                    st.markdown(f"**🍽️ {m['food_name']}**")
                    st.caption(f"{m['serving_size_g']:.0f}g portion")
                with c2:
                    st.markdown(f"<span class='badge-pill badge-meal'>{m['meal_type']}</span>", unsafe_allow_html=True)
                with c3:
                    st.write(f"**{m['calories']:.0f}** kcal")
                with c4:
                    st.write(f"P: {m['protein']:.1f}g")
                with c5:
                    st.write(f"C: {m['carbs']:.1f}g | F: {m['fat']:.1f}g")
                with c6:
                    if st.button("🗑️", key=f"del_meal_{m_id}", help="Delete meal log"):
                        delete_meal_log(m_id, active_user_id)
                        st.success("Meal removed.")
                        st.rerun()
                st.divider()

        # Day totals card
        tot_p = sum(m["protein"] for m in day_meals)
        tot_c = sum(m["carbs"] for m in day_meals)
        tot_f = sum(m["fat"] for m in day_meals)
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        col_t1.metric("Day Calories", f"{day_calories:.0f} kcal")
        col_t2.metric("Day Protein", f"{tot_p:.1f} g")
        col_t3.metric("Day Carbs", f"{tot_c:.1f} g")
        col_t4.metric("Day Fat", f"{tot_f:.1f} g")

    else:
        st.info(f"No meals recorded for {date_str}. Use the form above, or click '➕ Log to Meal Tracker' on recipe recommendations and image analysis results!")

    # Historical Log View Expander
    st.divider()
    with st.expander("📜 All Historical Meal Logs", expanded=False):
        all_user_meals = get_meal_logs(active_user_id)
        if all_user_meals:
            all_df = pd.DataFrame(all_user_meals)
            display_cols = ["date", "meal_type", "food_name", "serving_size_g", "calories", "protein", "carbs", "fat"]
            st.dataframe(all_df[display_cols], use_container_width=True, hide_index=True)
            csv_data = all_df[display_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export Meal History as CSV",
                data=csv_data,
                file_name=f"nutrifood_meal_history_{active_user_id}.csv",
                mime="text/csv"
            )
        else:
            st.write("No historical meals found.")


# --------------------------------------------------
# 13. SIDEBAR FOOTER
# --------------------------------------------------

st.sidebar.divider()
st.sidebar.caption(
    "NutriFood | Personalized Nutrition-Aware Food Recommendation and Dietary Analytics System"
)
