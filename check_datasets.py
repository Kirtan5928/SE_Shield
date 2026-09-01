import pandas as pd
import os

# 🔁 Update paths if needed
DATA_PATH = "data/raw"

files = [
    "enron.csv",
    "phishing.csv",
    "spam_email.csv",
    "synthetic.csv"
]

total_rows = 0

print("\n📊 DATASET ANALYSIS REPORT\n")

for file in files:
    path = os.path.join(DATA_PATH, file)

    print("="*50)
    print(f"📁 File: {file}")

    try:
        df = pd.read_csv(path)

        rows, cols = df.shape
        total_rows += rows

        print(f"➡️ Rows: {rows}")
        print(f"➡️ Columns: {cols}")
        print(f"➡️ Column Names: {list(df.columns)}")

        print("\n🔍 Null Values:")
        print(df.isnull().sum())

        print("\n🔁 Duplicate Rows:", df.duplicated().sum())

        print("\n🧪 Sample Data:")
        print(df.head(2))

    except Exception as e:
        print(f"❌ Error reading {file}: {e}")

    print("="*50 + "\n")

print(f"🔥 TOTAL ROWS ACROSS ALL DATASETS: {total_rows}")