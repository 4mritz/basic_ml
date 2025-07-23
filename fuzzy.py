import pandas as pd
import sqlite3
from rapidfuzz import process, fuzz

DB_NAME = "people.db"
EXCEL_FILE = "people.xlsx"
TABLE_NAME = "people"

def import_excel_to_sqlite():
    df = pd.read_excel(EXCEL_FILE)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        initial TEXT NOT NULL,
        mobile TEXT NOT NULL
    )
    """)
    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

def search_by_initial_and_fuzzy_name(initial_input, fuzzy_name_input):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, name, initial, mobile FROM {TABLE_NAME} WHERE initial = ?", (initial_input,))
    results = cursor.fetchall()
    if not results:
        print("No records found with initial:", initial_input)
        return
    names_list = [row[1] for row in results]
    best_match, score, index = process.extractOne(fuzzy_name_input, names_list, scorer=fuzz.WRatio)
    matched_row = results[index]
    print("Name      :", matched_row[1])
    print("Initial   :", matched_row[2])
    print("Mobile    :", matched_row[3])
    print("Similarity:", score, "%")
    conn.close()

def search_by_mobile(mobile_input):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, name, initial, mobile FROM {TABLE_NAME} WHERE mobile = ?", (mobile_input,))
    result = cursor.fetchone()
    if result:
        print("Name    :", result[1])
        print("Initial :", result[2])
        print("Mobile  :", result[3])
    else:
        print("No match found for mobile number.")
    conn.close()

if __name__ == "__main__":
    import_excel_to_sqlite()
    print("1. Search by Initial and Fuzzy Name")
    print("2. Search by Mobile Number")
    choice = input("Choose option (1 or 2): ").strip()
    if choice == "1":
        initial = input("Enter initial: ").strip().upper()
        name = input("Enter name (with possible typos): ").strip()
        search_by_initial_and_fuzzy_name(initial, name)
    elif choice == "2":
        mobile = input("Enter mobile number: ").strip()
        search_by_mobile(mobile)
    else:
        print("Invalid choice.")
