
import sqlite3
import bcrypt

from src.database import DB_PATH


def register_user(username, password):
    """Register a new user with a securely hashed password."""
    username = username.strip()

    if not username or not password:
        return False, "Username and password are required."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    password_hash = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(),
    ).decode("utf-8")

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO users (username, password_hash)
                VALUES (?, ?)
                """,
                (username, password_hash),
            )

        return True, "Account created successfully."

    except sqlite3.IntegrityError:
        return False, "That username is already registered."


def authenticate_user(username, password):
    """Verify a user's username and password."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT user_id, username, password_hash
            FROM users
            WHERE username = ?
            """,
            (username.strip(),),
        ).fetchone()

    if row is None:
        return None

    user_id, stored_username, stored_hash = row

    password_matches = bcrypt.checkpw(
        password.encode("utf-8"),
        stored_hash.encode("utf-8"),
    )

    if password_matches:
        return {
            "user_id": user_id,
            "username": stored_username,
        }

    return None
