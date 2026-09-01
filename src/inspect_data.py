import pandas as pd
import os

# Get the project root regardless of where script runs from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')

print("Looking for data in:", RAW_DIR)
print("Files found:", os.listdir(RAW_DIR) if os.path.exists(RAW_DIR) else "FOLDER NOT FOUND")

print("\n" + "=" * 60)
print("ENRON DATASET")
print("=" * 60)
try:
    enron = pd.read_csv(os.path.join(RAW_DIR, 'enron.csv'))
    print("Columns:", enron.columns.tolist())
    print("Shape:", enron.shape)
    print(enron.head(2))
    print("Label unique values:", enron.iloc[:, -1].unique())
except Exception as e:
    print("Error:", e)

print("\n" + "=" * 60)
print("PHISHING DATASET")
print("=" * 60)
try:
    phishing = pd.read_csv(os.path.join(RAW_DIR, 'phishing.csv'))
    print("Columns:", phishing.columns.tolist())
    print("Shape:", phishing.shape)
    print(phishing.head(2))
    print("Label unique values:", phishing.iloc[:, -1].unique())
except Exception as e:
    print("Error:", e)

print("\n" + "=" * 60)
print("SPAM EMAIL DATASET")
print("=" * 60)
try:
    spam = pd.read_csv(os.path.join(RAW_DIR, 'spam_email.csv'))
    print("Columns:", spam.columns.tolist())
    print("Shape:", spam.shape)
    print(spam.head(2))
    print("Label unique values:", spam.iloc[:, -1].unique())
except Exception as e:
    print("Error:", e)

print("\n" + "=" * 60)
print("SYNTHETIC DATASET")
print("=" * 60)
try:
    synthetic = pd.read_csv(os.path.join(RAW_DIR, 'synthetic.csv'))
    print("Columns:", synthetic.columns.tolist())
    print("Shape:", synthetic.shape)
    print(synthetic.head(2))
    print("Label unique values:", synthetic['label'].unique())
except Exception as e:
    print("Error:", e)
