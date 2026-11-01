import os
import sqlite3
import hashlib
import secrets
import json
from datetime import datetime

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "nutrifood.db")


def get_connection():
    """Create and return a database connection with WAL mode and foreign keys enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def hash_password(password: str, salt: str = None) -> tuple[str, str]:
    """Hash password securely using PBKDF2-HMAC-SHA256 with a unique random salt."""
    if not salt:
        salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        bytes.fromhex(salt),
        100000
    )
    return key.hex(), salt


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    """Verify provided password matches the salted PBKDF2 hash."""
    calc_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(calc_hash, expected_hash)


def init_db():
    """Initialize database tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        full_name TEXT DEFAULT '',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS food_profiles (
        user_id INTEGER PRIMARY KEY,
        diet TEXT DEFAULT 'No restriction',
        preferred_ingredients TEXT DEFAULT '[]',
        avoid_ingredients TEXT DEFAULT '[]',
        allergies TEXT DEFAULT '[]',
        max_calories REAL DEFAULT 600.0,
        max_time INTEGER DEFAULT 60,
        daily_calorie_target REAL DEFAULT 2000.0,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS meal_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        meal_type TEXT NOT NULL,
        food_name TEXT NOT NULL,
        serving_size_g REAL DEFAULT 100.0,
        calories REAL DEFAULT 0.0,
        protein REAL DEFAULT 0.0,
        carbs REAL DEFAULT 0.0,
        fat REAL DEFAULT 0.0,
        fiber REAL DEFAULT 0.0,
        sugars REAL DEFAULT 0.0,
        sodium REAL DEFAULT 0.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS user_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        recipe_id TEXT NOT NULL,
        recipe_name TEXT NOT NULL,
        action TEXT NOT NULL,
        rating INTEGER DEFAULT 5,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS recommendation_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        recipe_id TEXT NOT NULL,
        recipe_name TEXT NOT NULL,
        hybrid_score REAL DEFAULT 0.0,
        recommended_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    );
    """)

    conn.commit()
    conn.close()


# ----------------------------------------------------------------------------
# USER AUTHENTICATION
# ----------------------------------------------------------------------------

def create_user(username: str, email: str, password: str, full_name: str = "") -> tuple[bool, str]:
    """Register a new user account. Returns (success, message)."""
    username = username.strip().lower()
    email = email.strip().lower()

    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters."
    if not email or "@" not in email:
        return False, "Please provide a valid email address."
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters."

    pwd_hash, salt = hash_password(password)

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO users (username, email, password_hash, salt, full_name)
               VALUES (?, ?, ?, ?, ?)""",
            (username, email, pwd_hash, salt, full_name.strip())
        )
        user_id = cursor.lastrowid

        # Initialize default profile
        cursor.execute(
            """INSERT INTO food_profiles (user_id, diet, preferred_ingredients, avoid_ingredients, allergies, max_calories, max_time, daily_calorie_target)
               VALUES (?, 'No restriction', '[]', '[]', '[]', 600, 60, 2000)""",
            (user_id,)
        )
        conn.commit()
        return True, "User registered successfully."
    except sqlite3.IntegrityError as e:
        msg = str(e).lower()
        if "username" in msg:
            return False, f"Username '{username}' is already taken."
        elif "email" in msg:
            return False, f"Email '{email}' is already registered."
        return False, "User registration failed due to duplicate entry."
    except Exception as e:
        return False, f"Error creating user: {e}"
    finally:
        conn.close()


def authenticate_user(username_or_email: str, password: str) -> dict | None:
    """Validate credentials and return user record dict, or None if invalid."""
    ident = username_or_email.strip().lower()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT id, username, email, password_hash, salt, full_name, created_at
               FROM users WHERE username = ? OR email = ?""",
            (ident, ident)
        )
        row = cursor.fetchone()
        if not row:
            return None

        if verify_password(password, row["salt"], row["password_hash"]):
            return {
                "id": row["id"],
                "username": row["username"],
                "email": row["email"],
                "full_name": row["full_name"],
                "created_at": row["created_at"]
            }
        return None
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> dict | None:
    """Retrieve user info by primary key ID."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, email, full_name, created_at FROM users WHERE id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ----------------------------------------------------------------------------
# FOOD PROFILE PERSISTENCE
# ----------------------------------------------------------------------------

def save_food_profile(
    user_id: int,
    diet: str,
    preferred: list,
    avoid: list,
    allergies: list,
    max_calories: float = 600.0,
    max_time: int = 60,
    daily_calorie_target: float = 2000.0
) -> bool:
    """Upsert user's food profile into database."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO food_profiles (
                   user_id, diet, preferred_ingredients, avoid_ingredients,
                   allergies, max_calories, max_time, daily_calorie_target, updated_at
               )
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(user_id) DO UPDATE SET
                   diet = excluded.diet,
                   preferred_ingredients = excluded.preferred_ingredients,
                   avoid_ingredients = excluded.avoid_ingredients,
                   allergies = excluded.allergies,
                   max_calories = excluded.max_calories,
                   max_time = excluded.max_time,
                   daily_calorie_target = excluded.daily_calorie_target,
                   updated_at = CURRENT_TIMESTAMP""",
            (
                user_id,
                diet,
                json.dumps(preferred),
                json.dumps(avoid),
                json.dumps(allergies),
                float(max_calories),
                int(max_time),
                float(daily_calorie_target)
            )
        )
        conn.commit()
        return True
    finally:
        conn.close()


def get_food_profile(user_id: int) -> dict:
    """Retrieve food profile for user. Falls back to default if not saved yet."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT diet, preferred_ingredients, avoid_ingredients, allergies,
                      max_calories, max_time, daily_calorie_target
               FROM food_profiles WHERE user_id = ?""",
            (user_id,)
        )
        row = cursor.fetchone()
        if not row:
            return {
                "diet": "No restriction",
                "preferred": [],
                "avoid": [],
                "allergies": [],
                "max_calories": 600.0,
                "max_time": 60,
                "daily_calorie_target": 2000.0
            }

        def safe_json(val):
            try:
                return json.loads(val) if val else []
            except Exception:
                return []

        return {
            "diet": row["diet"] or "No restriction",
            "preferred": safe_json(row["preferred_ingredients"]),
            "avoid": safe_json(row["avoid_ingredients"]),
            "allergies": safe_json(row["allergies"]),
            "max_calories": float(row["max_calories"]) if row["max_calories"] is not None else 600.0,
            "max_time": int(row["max_time"]) if row["max_time"] is not None else 60,
            "daily_calorie_target": float(row["daily_calorie_target"]) if row["daily_calorie_target"] is not None else 2000.0
        }
    finally:
        conn.close()


# ----------------------------------------------------------------------------
# MEAL TRACKING PERSISTENCE
# ----------------------------------------------------------------------------

def add_meal_log(
    user_id: int,
    date_str: str,
    meal_type: str,
    food_name: str,
    serving_size_g: float = 100.0,
    calories: float = 0.0,
    protein: float = 0.0,
    carbs: float = 0.0,
    fat: float = 0.0,
    fiber: float = 0.0,
    sugars: float = 0.0,
    sodium: float = 0.0
) -> int:
    """Insert a meal record into database. Returns inserted row ID."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO meal_logs (
                   user_id, date, meal_type, food_name, serving_size_g,
                   calories, protein, carbs, fat, fiber, sugars, sodium
               )
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id, date_str, meal_type, food_name,
                float(serving_size_g), float(calories),
                float(protein), float(carbs), float(fat),
                float(fiber), float(sugars), float(sodium)
            )
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_meal_logs(user_id: int, date_str: str = None) -> list[dict]:
    """Retrieve logged meals for a user, optionally filtered by date (YYYY-MM-DD)."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        if date_str:
            cursor.execute(
                """SELECT id, date, meal_type, food_name, serving_size_g,
                          calories, protein, carbs, fat, fiber, sugars, sodium, created_at
                   FROM meal_logs
                   WHERE user_id = ? AND date = ?
                   ORDER BY id ASC""",
                (user_id, date_str)
            )
        else:
            cursor.execute(
                """SELECT id, date, meal_type, food_name, serving_size_g,
                          calories, protein, carbs, fat, fiber, sugars, sodium, created_at
                   FROM meal_logs
                   WHERE user_id = ?
                   ORDER BY date DESC, id DESC""",
                (user_id,)
            )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def delete_meal_log(meal_id: int, user_id: int) -> bool:
    """Delete a meal record belonging to user."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM meal_logs WHERE id = ? AND user_id = ?",
            (meal_id, user_id)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ----------------------------------------------------------------------------
# USER FEEDBACK & RECOMMENDATION AUDIT
# ----------------------------------------------------------------------------

def save_recipe_feedback(user_id: int, recipe_id: str, recipe_name: str, action: str, rating: int = 5) -> bool:
    """Save user action (liked/saved/disliked) on a recipe."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO user_feedback (user_id, recipe_id, recipe_name, action, rating)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, str(recipe_id), recipe_name, action, rating)
        )
        conn.commit()
        return True
    finally:
        conn.close()


def get_user_saved_recipes(user_id: int) -> list[dict]:
    """Retrieve recipes bookmarked/saved by user."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT id, recipe_id, recipe_name, action, rating, created_at
               FROM user_feedback
               WHERE user_id = ? AND action = 'saved'
               ORDER BY id DESC""",
            (user_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


# Initialize database automatically on module import
init_db()

