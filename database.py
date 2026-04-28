import sqlite3

conn = sqlite3.connect("traffic.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS fines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate TEXT,
    fine INTEGER,
    date TEXT,
    time TEXT
)
""")

conn.commit()
conn.close()

print("✅ Database created")