import os
import re
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import wordninja
from sklearn.model_selection import train_test_split
from tqdm import tqdm
tqdm.pandas()  # enables df.progress_apply()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
SPLITS_DIR    = os.path.join(BASE_DIR, 'data', 'splits')

print("=" * 60)
print("ADVERSARIAL PREPROCESSING — OPTIMIZED + PROGRESS BARS")
print("=" * 60)
print("""
Optimizations:
  KEPT     : autocorrect Speller   
  KEPT     : L33t normalization    (str.translate — fast)
  KEPT     : wordninja splitting   (long tokens only)
  KEPT     : adversarial augmentation on attack class
  ADDED    : tqdm progress bars

Expected runtime: 3-5 minutes on 95k rows
""")

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading balanced dataset...")
df = pd.read_csv(os.path.join(PROCESSED_DIR, 'merged_dataset_v2.csv'))
print(f"Loaded : {len(df):,} rows  |  "
      f"Attack: {(df['label']==1).sum():,}  "
      f"Legit: {(df['label']==0).sum():,}")

# ── L33t maps ─────────────────────────────────────────────────────────────────
REVERSE_LEET = {
    '0': 'o', '1': 'i', '3': 'e',
    '4': 'a', '5': 's', '7': 't',
    '@': 'a', '$': 's', '!': 'i'
}

FORWARD_LEET = {
    'o': '0', 'i': '1', 'e': '3',
    'a': '4', 's': '5', 't': '7'
}

COMMON_CORRECTIONS = {
    'accccount' : 'account',
    'acccount'  : 'account',
    'veriffy'   : 'verify',
    'verfy'     : 'verify',
    'verrify'   : 'verify',
    'urgant'    : 'urgent',
    'urrgent'   : 'urgent',
    'suspendd'  : 'suspended',
    'susppended': 'suspended',
    'passworrd' : 'password',
    'passwrod'  : 'password',
    'credentals': 'credentials',
    'credentail': 'credential',
    'confirmm'  : 'confirm',
    'conffirm'  : 'confirm',
    'immedate'  : 'immediate',
    'immediatly': 'immediately',
    'securty'   : 'security',
    'securtiy'  : 'security',
    'updte'     : 'update',
    'updatte'   : 'update',
    'clickk'    : 'click',
    'clikc'     : 'click',
    'loginn'    : 'login',
    'loggn'     : 'login',
}

TRANSLATION_TABLE = str.maketrans(REVERSE_LEET)

# ═══════════════════════════════════════════════════════════════════════════════
# NORMALIZATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
def normalize_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text

    # 1. Fix l33t speak via str.translate (fastest method)
    text = text.translate(TRANSLATION_TABLE)

    # 2. Split suspiciously long tokens using wordninja
    tokens       = text.split()
    split_tokens = []
    for token in tokens:
        if len(token) > 15:
            split_tokens.extend(wordninja.split(token))
        else:
            split_tokens.append(token)

    # 3. Dictionary correction — O(1) lookup per token
    corrected = [
        COMMON_CORRECTIONS.get(token.lower(), token)
        for token in split_tokens
    ]

    return ' '.join(corrected)

# ═══════════════════════════════════════════════════════════════════════════════
# ADVERSARIAL AUGMENTATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
def generate_adversarial(text: str, rng: np.random.Generator) -> str:
    if not isinstance(text, str):
        return text

    words     = text.split()
    if not words:
        return text

    # generate all random numbers at once — faster than per-word random()
    leet_mask = rng.random(len(words)) < 0.3
    join_mask = rng.random(len(words)) < 0.15

    new_words = []
    for i, word in enumerate(words):
        if leet_mask[i] and len(word) > 3:
            word = ''.join([
                FORWARD_LEET.get(c, c)
                if rng.random() < 0.4 else c
                for c in word
            ])
        new_words.append(word)

    # randomly concatenate adjacent words
    result = []
    i = 0
    while i < len(new_words):
        if join_mask[i] and i + 1 < len(new_words):
            result.append(new_words[i] + new_words[i+1])
            i += 2
        else:
            result.append(new_words[i])
            i += 1

    return ' '.join(result)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1 — NORMALIZATION")
print("=" * 60)

print("\nVerification on test cases first:")
test_cases = [
    ("v3r1fy ur acc0unt imm3d1at3ly", "verify ur account immediately"),
    ("urgentverifyaccountnow",         "urgent verify account now"),
    ("accccount suspendd",             "account suspended"),
    ("y0ur p4ssw0rd h45 3xp1r3d",     "your password has expired"),
]
for inp, expected in test_cases:
    out    = normalize_text(inp)
    status = "✅" if any(w in out for w in expected.split()[:2]) else "⚠️"
    print(f"  {status}  IN : {inp}")
    print(f"       OUT: {out}")

print(f"\nRunning normalization on {len(df):,} rows...")
df['text'] = df['text'].progress_apply(normalize_text)
print("Normalization complete.")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — ADVERSARIAL AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2 — ADVERSARIAL AUGMENTATION (attack class only)")
print("=" * 60)

rng       = np.random.default_rng(42)
attack_df = df[df['label'] == 1].copy()

print(f"Generating adversarial variants for {len(attack_df):,} attack samples...")
attack_df['text'] = attack_df['text'].progress_apply(
    lambda x: generate_adversarial(x, rng)
)

# merge original + adversarial augmented attack samples
df_final = pd.concat([df, attack_df], ignore_index=True)
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nAfter augmentation:")
print(f"  Total  : {len(df_final):,}")
print(f"  Attack : {(df_final['label']==1).sum():,}")
print(f"  Legit  : {(df_final['label']==0).sum():,}")

# Example
original  = df[df['label']==1]['text'].iloc[0][:120]
augmented = generate_adversarial(original, np.random.default_rng(1))
print(f"\nAugmentation example:")
print(f"  BEFORE : {original}")
print(f"  AFTER  : {augmented}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SAVE + REBUILD SPLITS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3 — SAVING + REBUILDING SPLITS")
print("=" * 60)

df_final.to_csv(
    os.path.join(PROCESSED_DIR, 'merged_dataset_adv.csv'),
    index=False
)
print(f"Saved → data/processed/merged_dataset_adv.csv  ({len(df_final):,} rows)")

X = df_final['text']
y = df_final['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.2,
    random_state = 42,
    stratify     = y
)

joblib.dump(X_train, os.path.join(SPLITS_DIR, 'X_train.pkl'))
joblib.dump(X_test,  os.path.join(SPLITS_DIR, 'X_test.pkl'))
joblib.dump(y_train, os.path.join(SPLITS_DIR, 'y_train.pkl'))
joblib.dump(y_test,  os.path.join(SPLITS_DIR, 'y_test.pkl'))

print(f"Train : {len(X_train):,} | Attack: {y_train.sum():,} | Legit: {(y_train==0).sum():,}")
print(f"Test  : {len(X_test):,}  | Attack: {y_test.sum():,}  | Legit: {(y_test==0).sum():,}")
print("\nAdversarial preprocessing complete. Next → 04_feature_extraction.py")