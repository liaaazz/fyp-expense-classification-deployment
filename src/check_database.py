
import sqlite3
from database import DB_PATH, initialize_database

initialize_database()

with sqlite3.connect(DB_PATH) as conn:
    cursor = conn.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
    """)

    tables = cursor.fetchall()

print("Database file:", DB_PATH)
print("Tables found:")

for table in tables:
    print("-", table[0])

if any(table[0] == "expenses" for table in tables):
    print("\nSUCCESS: The expenses table exists!")
else:
    print("\nERROR: The expenses table was not found.")
