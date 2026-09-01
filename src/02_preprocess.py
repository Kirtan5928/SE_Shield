import pandas as pd
import numpy as np
import re
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
SPLITS_DIR    = os.path.join(BASE_DIR, 'data', 'splits')
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR,    exist_ok=True)

RANDOM_STATE     = 42
TARGET_PER_CLASS = 50000

# ── Helper — Enron body extraction ────────────────────────────────────────────
def extract_enron_body(raw: str) -> str:
    if not isinstance(raw, str):
        return ''
    parts = re.split(r'\n\n', raw, maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else raw.strip()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD ALL 4 DATASETS
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1 — LOADING ALL 4 DATASETS")
print("=" * 60)

# Enron — legitimate only
print("\n[1/4] Enron...")
enron_raw = pd.read_csv(os.path.join(RAW_DIR, 'enron.csv'))
enron = pd.DataFrame({
    'text' : enron_raw['message'].apply(extract_enron_body),
    'label': 0
})
enron = enron[enron['text'].str.len() > 20].reset_index(drop=True)
print(f"   Loaded : {len(enron):,} rows")

# Phishing
print("\n[2/4] Phishing...")
phishing_raw = pd.read_csv(os.path.join(RAW_DIR, 'phishing.csv'))
phishing = pd.DataFrame({
    'text' : phishing_raw['text_combined'],
    'label': phishing_raw['label'].astype(int)
})
phishing = phishing.dropna(subset=['text']).reset_index(drop=True)
print(f"   Loaded : {len(phishing):,} rows | {phishing['label'].value_counts().to_dict()}")

# Spam Email — combine subject + body
print("\n[3/4] Spam Email...")
spam_raw = pd.read_csv(os.path.join(RAW_DIR, 'spam_email.csv'))
spam = pd.DataFrame({
    'text' : spam_raw['subject'].fillna('') + ' ' + spam_raw['body'].fillna(''),
    'label': spam_raw['label'].astype(int)
})
spam = spam.dropna(subset=['text']).reset_index(drop=True)
print(f"   Loaded : {len(spam):,} rows | {spam['label'].value_counts().to_dict()}")

# Synthetic
print("\n[4/4] Synthetic...")
synth_raw = pd.read_csv(os.path.join(RAW_DIR, 'synthetic.csv'))
synth = pd.DataFrame({
    'text' : synth_raw['text'],
    'label': synth_raw['label'].astype(int)
})
synth = synth.dropna(subset=['text']).reset_index(drop=True)
print(f"   Loaded : {len(synth):,} rows | {synth['label'].value_counts().to_dict()}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — MERGE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2 — MERGING")
print("=" * 60)

df = pd.concat([enron, phishing, spam, synth], ignore_index=True)
df = df.dropna(subset=['text', 'label'])
df['text']  = df['text'].astype(str)
df['label'] = df['label'].astype(int)

print(f"Total rows     : {len(df):,}")
print(f"Attack (1)     : {(df['label']==1).sum():,}")
print(f"Legitimate (0) : {(df['label']==0).sum():,}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — CLEANING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3 — CLEANING")
print("=" * 60)

before = len(df)
df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
print(f"Removed duplicates : {before - len(df):,}")

df['word_count'] = df['text'].apply(lambda x: len(x.split()))
removed_short    = (df['word_count'] < 3).sum()
df = df[df['word_count'] >= 3].reset_index(drop=True)
print(f"Removed short (<3w): {removed_short:,}")

# Cap long texts — vectorized using pandas str operations
df['text'] = df['text'].apply(lambda x: ' '.join(x.split()[:500]))
df = df.drop(columns=['word_count'])
print(f"After cleaning     : {len(df):,} rows")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — BALANCE DATASET
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4 — BALANCING")
print("=" * 60)

legit  = df[df['label'] == 0]
attack = df[df['label'] == 1]

print(f"Before : Legit={len(legit):,}  Attack={len(attack):,}")

legit_sample  = legit.sample(
    n=min(TARGET_PER_CLASS, len(legit)),
    random_state=RANDOM_STATE
)
attack_sample = attack.sample(
    n=min(TARGET_PER_CLASS, len(attack)),
    random_state=RANDOM_STATE
)

df = pd.concat([legit_sample, attack_sample]).sample(
    frac=1, random_state=RANDOM_STATE
).reset_index(drop=True)

print(f"After  : Legit={( df['label']==0).sum():,}  "
      f"Attack={(df['label']==1).sum():,}  "
      f"Total={len(df):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — SAVE + SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5 — SAVING + TRAIN/TEST SPLIT")
print("=" * 60)

df.to_csv(os.path.join(PROCESSED_DIR, 'merged_dataset.csv'), index=False)
print("Saved → data/processed/merged_dataset.csv")

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.2,
    random_state = RANDOM_STATE,
    stratify     = y
)

print(f"Train : {len(X_train):,} | Attack: {y_train.sum():,} | Legit: {(y_train==0).sum():,}")
print(f"Test  : {len(X_test):,}  | Attack: {y_test.sum():,}  | Legit: {(y_test==0).sum():,}")

joblib.dump(X_train, os.path.join(SPLITS_DIR, 'X_train.pkl'))
joblib.dump(X_test,  os.path.join(SPLITS_DIR, 'X_test.pkl'))
joblib.dump(y_train, os.path.join(SPLITS_DIR, 'y_train.pkl'))
joblib.dump(y_test,  os.path.join(SPLITS_DIR, 'y_test.pkl'))
print("Saved → data/splits/")
print("\nPreprocessing complete. Ready for adversarial prep.")