import sqlite3
from pathlib import Path

# Locate the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Location of the SQLite database file
DB_PATH = BASE_DIR / "database" / "expenses.db"


def initialize_database():
    """Create the users and expenses tables safely."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        # User accounts
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Existing expenses table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                receipt_id TEXT PRIMARY KEY,
                source_pdf TEXT,
                page_number INTEGER,
                merchant_name TEXT,
                transaction_date TEXT,
                transaction_time TEXT,
                purchased_items TEXT,
                total_amount REAL,
                currency TEXT,
                expense_category TEXT,
                payment_method TEXT,
                verification_status TEXT DEFAULT 'Pending',
                extraction_notes TEXT,
                predicted_category TEXT
            )
        """)
        
        # Monthly budgets for individual users
        conn.execute("""
            CREATE TABLE IF NOT EXISTS budgets (
                budget_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                budget_month TEXT NOT NULL,
                budget_amount REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                UNIQUE (user_id, budget_month)
            )
        """)

        
        # Store approved examples for improving future predictions
        conn.execute("""
            CREATE TABLE IF NOT EXISTS learning_feedback (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                merchant_name TEXT NOT NULL,
                purchased_items TEXT NOT NULL,
                predicted_category TEXT,
                confirmed_category TEXT NOT NULL,
                share_for_learning INTEGER NOT NULL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)


        # Add user ownership to an existing database without
        # deleting any existing expense records.
        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(expenses)"
            ).fetchall()
        }

        if "user_id" not in columns:
            conn.execute("""
                ALTER TABLE expenses
                ADD COLUMN user_id INTEGER REFERENCES users(user_id)
            """)


def save_learning_feedback(
    user_id,
    merchant_name,
    purchased_items,
    predicted_category,
    confirmed_category,
    share_for_learning=False,
):
    """Save a user-confirmed category for future learning."""

    merchant_name = (merchant_name or "").strip()
    purchased_items = (purchased_items or "").strip()

    if not merchant_name or not purchased_items:
        return False

    if not confirmed_category:
        return False

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO learning_feedback (
                user_id,
                merchant_name,
                purchased_items,
                predicted_category,
                confirmed_category,
                share_for_learning
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                merchant_name,
                purchased_items,
                predicted_category,
                confirmed_category,
                int(share_for_learning),
            ),
        )

    return True

def save_expense(user_id, receipt_id, merchant_name, transaction_date,
                 purchased_items, total_amount, currency,
                 expense_category, predicted_category=None):
    """Save an expense for a specific user."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO expenses (
                receipt_id, merchant_name, transaction_date,
                purchased_items, total_amount, currency,
                expense_category, predicted_category, user_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt_id, merchant_name, transaction_date,
                purchased_items, total_amount, currency,
                expense_category, predicted_category, user_id
            ),
        )


def get_user_expenses(user_id):
    """Retrieve only expenses belonging to a specific user."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT *
            FROM expenses
            WHERE user_id = ?
            ORDER BY transaction_date DESC
            """,
            (user_id,),
        ).fetchall()

    return [dict(row) for row in rows]


def delete_expense(user_id, receipt_id):
    """Delete one expense belonging to the logged-in user."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            DELETE FROM expenses
            WHERE receipt_id = ? AND user_id = ?
            """,
            (receipt_id, user_id),
        )
        return cursor.rowcount > 0



def save_monthly_budget(user_id, budget_month, budget_amount):
    """Save or update a user's budget for a specific month."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO budgets (
                user_id, budget_month, budget_amount
            )
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, budget_month)
            DO UPDATE SET budget_amount = excluded.budget_amount
            """,
            (user_id, budget_month, budget_amount),
        )


def get_monthly_budget(user_id, budget_month):
    """Retrieve a user's budget for a specific month."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT budget_amount
            FROM budgets
            WHERE user_id = ? AND budget_month = ?
            """,
            (user_id, budget_month),
        ).fetchone()

    return row[0] if row else None

def get_shared_merchant_category(merchant_name):
    """Return a category only when shared examples agree."""

    merchant_name = (merchant_name or "").strip().casefold()

    if not merchant_name:
        return None

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT confirmed_category
            FROM learning_feedback
            WHERE share_for_learning = 1
              AND LOWER(TRIM(merchant_name)) = ?
            """,
            (merchant_name,),
        ).fetchall()

    if not rows:
        return None

    categories = {row[0] for row in rows if row[0]}

    # Avoid using a merchant rule if feedback conflicts.
    if len(categories) != 1:
        return None

    return categories.pop()

def get_shared_learning_examples():
    """Return only examples approved for shared learning."""

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT merchant_name, purchased_items, confirmed_category
            FROM learning_feedback
            WHERE share_for_learning = 1
              AND TRIM(merchant_name) != ''
              AND TRIM(purchased_items) != ''
              AND confirmed_category IN (
                  'Food and Beverage',
                  'Groceries',
                  'Transportation',
                  'Shopping',
                  'Health & Personal Care'
              )
            """
        ).fetchall()

    return [
        {
            "merchant_name": row[0],
            "purchased_items": row[1],
            "confirmed_category": row[2],
        }
        for row in rows
    ]

if __name__ == "__main__":
    initialize_database()
    print("Database initialised successfully!")
    print(f"Database location: {DB_PATH}")
