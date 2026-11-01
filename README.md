# 🥗 NutriFood: Personalized Nutrition-Aware Food Recommendation & Dietary Analytics System

NutriFood is an end-to-end intelligent dietary recommendation and analytics platform developed using Python, Scikit-Learn, PyTorch (Transformers), Plotly, SQLite, and Streamlit.

The system addresses modern nutritional challenges by combining natural language processing (NLP), user ingredient preference modeling, multi-objective nutritional scoring, computer vision-based food recognition, and verified USDA nutritional databases.

---

## 🌟 Key Features

### 1. 🍲 Hybrid Recommendation Engine (Phase 5.8)
Combines three distinct recommendation signals into a single unified hybrid objective:
$$\text{Hybrid Score} = 0.30 \times \text{Content Match} + 0.30 \times \text{Ingredient Preference} + 0.40 \times \text{Nutritional Quality}$$
- **Content Similarity (30%):** TF-IDF vectorization and cosine similarity over recipe titles, ingredient tokens, and cooking descriptions across 231,637 recipes.
- **Ingredient Preference Scoring (30%):** Vectorized affinity matching between user-specified favorite ingredients and recipe components.
- **Nutritional Quality Scoring (40%):** Percentile-based nutritional balance heuristic balancing protein intake against sugar, sodium, saturated fat, and calorie constraints:
  $$\text{Nutrition Score} = 0.35 \times \text{Protein} + 0.20 \times (1 - \text{Sugar}) + 0.20 \times (1 - \text{Sodium}) + 0.15 \times (1 - \text{Sat Fat}) + 0.10 \times \text{Calorie}$$
- **Hard Constraints Filtering:** Dietary restrictions (*Vegetarian, Vegan, Gluten-Free, Dairy-Free, Low-Carb*), allergen and ingredient exclusion, maximum preparation time, and serving calorie ceilings.
- **Explainable AI:** Generates bulleted justifications explaining why each dish was matched.

### 2. 📸 Food Image Analyzer with USDA FoodData Central
- **Vision Transformer (ViT):** Uses a pre-trained `nateraw/food` Vision Transformer to classify dishes from uploaded meal photographs.
- **USDA FoodData Central Integration:** Connects predictions directly to the official USDA FoodData Central REST API for accurate micronutrient and macronutrient lookups per 100g.
- **Interactive Portion Scaling:** Dynamically scales calories and macronutrients based on serving size (grams).
- **One-Click Meal Logging:** Add analyzed meals directly into the persistent Meal Tracker with one click.

### 3. 📊 Dual-Mode Nutrition Analytics Dashboard (Phase 5.9)
- **👤 Personal Dietary Intake Dashboard:**
  - Real-time calorie consumption vs. daily target goal.
  - Macronutrient breakdown donut chart (Protein vs. Carbs vs. Fat).
  - 7-Day calorie intake trend chart with target goal line.
  - Micronutrient tracking (Fiber, Sugars, Sodium vs. recommended thresholds).
  - Meal-wise calorie breakdown (Breakfast, Lunch, Dinner, Snack).
- **📚 Food.com Recipe Dataset Analytics:**
  - Interactive distribution histograms with outlier filtering across all 7 nutritional metrics.
  - Macronutrient PDV comparison box plots.
  - Multi-variable scatter plots and correlation analyses.
  - Preparation time impact visualizations (*Quick: ≤30 min, Medium: 30–60 min, Long: >60 min*).

### 4. 📅 Persistent Meal Tracker (Phase 5.10 & 5.11)
- Date picker to view and backfill daily meal records.
- Track Breakfast, Lunch, Dinner, and Snack entries with gram portions, calories, and macronutrients.
- Real-time calorie target progress bar with remaining calories indicator.
- Single-click entry deletion and complete historical CSV export.

### 5. 🔐 Secure SQLite User Persistence (Phase 5.11)
- Salted **PBKDF2-HMAC-SHA256** password hashing with cryptographically secure salts.
- Automatic profile syncing, recipe bookmarking (favorites), and isolated meal log storage.

---

## 🏗️ System Architecture & Database Schema

NutriFood operates with a local SQLite database (`database/nutrifood.db`) configured with Write-Ahead Logging (WAL):

```
┌─────────────────┐       ┌──────────────────────┐
│      users      │1     1│    food_profiles     │
├─────────────────┤───────├──────────────────────┤
│ id (PK)         │       │ user_id (FK, PK)     │
│ username        │       │ diet                 │
│ email           │       │ preferred_ingredients│
│ password_hash   │       │ avoid_ingredients    │
│ salt            │       │ allergies            │
│ full_name       │       │ max_calories         │
│ created_at      │       │ max_time             │
└────────┬────────┘       │ daily_calorie_target │
         │1               └──────────────────────┘
         │
         │* ┌──────────────────────────┐
         ├──┤        meal_logs         │
         │  ├──────────────────────────┤
         │  │ id (PK)                  │
         │  │ user_id (FK)             │
         │  │ date (YYYY-MM-DD)        │
         │  │ meal_type                │
         │  │ food_name, serving_size_g│
         │  │ calories, protein, carbs │
         │  │ fat, fiber, sugars, sod. │
         │  └──────────────────────────┘
         │* ┌──────────────────────────┐
         └──┤      user_feedback       │
            ├──────────────────────────┤
            │ id (PK), user_id (FK)    │
            │ recipe_id, recipe_name   │
            │ action ('saved'), rating │
            └──────────────────────────┘
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11, 3.12, or 3.13
- Virtual environment (`venv`)

### 2. Environment Setup
Clone or navigate to the repository directory:
```powershell
cd B:\MiniProject
```

Activate the virtual environment:
```powershell
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows Command Prompt:
.\venv\Scripts\activate.bat
```

Install required dependencies:
```powershell
pip install -r requirements.txt
```

### 3. Launching the Application
Run the Streamlit application:
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests
NutriFood includes an end-to-end integration test suite validating database authentication, profile upserts, hybrid recommendation scoring, recipe bookmarking, and meal log CRUD operations:

```powershell
.\venv\Scripts\python.exe test_system_integration.py
```

---

## 📂 Project Directory Structure

```
B:\MiniProject\
├── .streamlit/
│   ├── config.toml               # Streamlit server and theme configuration
│   └── secrets.toml              # USDA API key (ignored by git)
├── database/
│   ├── db_manager.py             # SQLite database layer with PBKDF2 auth
│   └── nutrifood.db              # SQLite relational database
├── models/
│   ├── recipes.csv               # Food.com curated recipe metadata
│   ├── tfidf_matrix.npz          # Pre-computed TF-IDF sparse matrix (231,637 x 50,000)
│   ├── tfidf_vectorizer.joblib   # Fitted Scikit-Learn TF-IDF vectorizer
│   ├── recipe_id_to_index.json   # Fast O(1) index mapping
│   ├── nutrition_scores.csv      # Pre-calculated multi-objective nutrition grades
│   └── sample_hybrid_recommendations.csv
├── app.py                        # Main Streamlit web application
├── recommendation_engine.py      # Core hybrid recommendation engine
├── food_image_analyzer.py        # Vision Transformer deep learning classifier
├── nutrition_lookup.py           # USDA FoodData Central API client
├── test_system_integration.py    # End-to-end automated verification suite
├── requirements.txt              # Production dependency specifications
├── .gitignore                    # Git protection rules
└── README.md                     # Academic project documentation
```

