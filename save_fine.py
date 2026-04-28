import sqlite3
from datetime import datetime

def save_to_db(plate_text):

    conn = sqlite3.connect("traffic.db")
    cursor = conn.cursor()

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    fine_amount = 500

    cursor.execute("""
    INSERT INTO fines (plate, fine, date, time)
    VALUES (?, ?, ?, ?)
    """, (plate_text, fine_amount, date, time))

    conn.commit()
    conn.close()

    print(f"✅ Saved: {plate_text} fined ₹{fine_amount}")