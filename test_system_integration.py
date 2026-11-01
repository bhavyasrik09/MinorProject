# ============================================================================
# NUTRIFOOD END-TO-END SYSTEM INTEGRATION TEST
# ============================================================================

import sys
import datetime
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
from recommendation_engine import (
    recipes,
    recommend_hybrid,
    search_recipe
)

print("=" * 75)
print("RUNNING END-TO-END INTEGRATION TEST SUITE")
print("=" * 75)

# 1. DATABASE & AUTHENTICATION
print("\n[Test 1] Testing User Authentication & Persistence...")
test_user = "test_evaluator"
test_email = "evaluator@test.com"
test_pass = "SecurePass123!"

# Try login or register
auth_result = authenticate_user(test_user, test_pass)
if not auth_result:
    ok, msg = create_user(test_user, test_email, test_pass, full_name="Evaluation User")
    print(f"  User Registration: {ok} ({msg})")
    auth_result = authenticate_user(test_user, test_pass)

assert auth_result is not None, "Authentication failed!"
user_id = auth_result["id"]
print(f"  Authenticated successfully as ID: {user_id}, Name: {auth_result['full_name']}")

# 2. FOOD PROFILE PERSISTENCE
print("\n[Test 2] Testing Food Profile Upsert & Retrieval...")
diet = "Vegetarian"
preferred = ["spinach", "garlic", "tomato"]
avoid = ["mushrooms"]
allergies = ["peanuts"]
max_cal = 550.0
max_time = 45
target = 2100.0

save_ok = save_food_profile(user_id, diet, preferred, avoid, allergies, max_cal, max_time, target)
assert save_ok, "Failed to save food profile"

loaded_profile = get_food_profile(user_id)
assert loaded_profile["diet"] == diet, f"Diet mismatch: {loaded_profile['diet']}"
assert "spinach" in loaded_profile["preferred"], "Preferred ingredient missing"
assert loaded_profile["daily_calorie_target"] == 2100.0, "Calorie target mismatch"
print(f"  Profile saved and loaded accurately: {loaded_profile['diet']}, Max: {loaded_profile['max_calories']} kcal")

# 3. HYBRID RECOMMENDATION ENGINE
print("\n[Test 3] Testing Hybrid Recommendation Engine...")
recs = recommend_hybrid(loaded_profile, query_text="pasta", top_n=5)
assert not recs.empty, "No hybrid recommendations returned!"
assert len(recs) <= 5, "Returned more recipes than top_n!"
for i, r in recs.iterrows():
    # Verify constraints
    assert r["calories"] <= max_cal, f"Recipe exceeded max calories: {r['calories']}"
    assert r["minutes_clean"] <= max_time, f"Recipe exceeded max time: {r['minutes_clean']}"
    assert "content_score" in r, "content_score missing"
    assert "preference_score" in r, "preference_score missing"
    assert "nutrition_score" in r, "nutrition_score missing"
    assert "hybrid_score" in r, "hybrid_score missing"
    assert len(r["explanation"]) > 0, "Recommendation explanation missing"

print(f"  Top recommended recipe: '{recs.iloc[0]['name']}'")
print(f"  Hybrid Score: {recs.iloc[0]['hybrid_score']:.4f} (Content: {recs.iloc[0]['content_score']:.3f}, Pref: {recs.iloc[0]['preference_score']:.3f}, Nutr: {recs.iloc[0]['nutrition_score']:.3f})")
print(f"  Explanations generated: {recs.iloc[0]['explanation']}")

# 4. RECIPE BOOKMARKING
print("\n[Test 4] Testing Recipe Bookmark Saving...")
top_recipe_id = str(recs.iloc[0]["id"])
top_recipe_name = str(recs.iloc[0]["name"])
fb_ok = save_recipe_feedback(user_id, top_recipe_id, top_recipe_name, "saved", 5)
assert fb_ok, "Failed to save feedback"
saved_recipes = get_user_saved_recipes(user_id)
assert any(s["recipe_id"] == top_recipe_id for s in saved_recipes), "Saved recipe not retrieved!"
print(f"  Recipe '{top_recipe_name}' successfully bookmarked and retrieved from SQLite")

# 5. MEAL TRACKING CRUD
print("\n[Test 5] Testing Meal Tracking (Add, Query, Delete)...")
today = datetime.date.today().isoformat()
m1_id = add_meal_log(
    user_id=user_id,
    date_str=today,
    meal_type="Breakfast",
    food_name="Oatmeal with Almonds",
    serving_size_g=120.0,
    calories=320.0,
    protein=12.0,
    carbs=48.0,
    fat=9.0
)
assert m1_id > 0, "Failed to add meal 1"

m2_id = add_meal_log(
    user_id=user_id,
    date_str=today,
    meal_type="Lunch",
    food_name=top_recipe_name,
    serving_size_g=150.0,
    calories=float(recs.iloc[0]["calories"]),
    protein=18.0,
    carbs=42.0,
    fat=11.0
)
assert m2_id > 0, "Failed to add meal 2"

# Query meals
meals = get_meal_logs(user_id, today)
assert len(meals) >= 2, f"Expected at least 2 meals, found {len(meals)}"
day_calories = sum(m["calories"] for m in meals)
print(f"  Logged {len(meals)} meals for {today}. Total Calories: {day_calories:.1f} kcal")

# Delete meal 1
del_ok = delete_meal_log(m1_id, user_id)
assert del_ok, "Failed to delete meal log"
meals_after = get_meal_logs(user_id, today)
assert not any(m["id"] == m1_id for m in meals_after), "Deleted meal still in database!"
print(f"  Successfully deleted meal ID {m1_id}. Remaining meals: {len(meals_after)}")

print("\n" + "=" * 75)
print("ALL END-TO-END TESTS PASSED SUCCESSFULLY! (100% SUCCESS)")
print("=" * 75)

