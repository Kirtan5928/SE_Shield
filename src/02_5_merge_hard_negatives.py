import os
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR  = os.path.join(BASE_DIR, 'data', 'processed')
DATA_GEN_DIR   = os.path.join(BASE_DIR, 'data_generation')

print("=" * 60)
print("STEP 02.5 — MERGING HARD NEGATIVES INTO DATASET")
print("=" * 60)

# ── Load datasets ──────────────────────────────────────────────────────────────
original_path = os.path.join(PROCESSED_DIR, 'merged_dataset_adv.csv')
hard_neg_path = os.path.join(DATA_GEN_DIR, 'phishing_dataset.csv')

print(f"Loading original dataset...")
original = pd.read_csv(original_path)

print(f"Loading hard negatives dataset...")
hard_neg = pd.read_csv(hard_neg_path)

print(f"\nOriginal dataset : {original.shape}")
print(f"Hard negatives   : {hard_neg.shape}")

# ── Basic validation ───────────────────────────────────────────────────────────
assert 'text' in original.columns and 'label' in original.columns, \
    "Original dataset must have 'text' and 'label' columns"

assert 'text' in hard_neg.columns and 'label' in hard_neg.columns, \
    "Hard negative dataset must have 'text' and 'label' columns"

# ── Merge datasets ─────────────────────────────────────────────────────────────
print("\nMerging datasets...")
df_new = pd.concat([original, hard_neg], ignore_index=True)

# ── Shuffle ────────────────────────────────────────────────────────────────────
df_new = df_new.sample(frac=1, random_state=42).reset_index(drop=True)

# ── Stats ──────────────────────────────────────────────────────────────────────
print("\nFinal dataset stats:")
print(f"Total rows : {len(df_new):,}")
print(df_new['label'].value_counts())

attack_ratio = df_new['label'].mean() * 100
print(f"Attack ratio : {attack_ratio:.2f}%")

# ── Save ───────────────────────────────────────────────────────────────────────
output_path = os.path.join(PROCESSED_DIR, 'merged_dataset_v2.csv')
df_new.to_csv(output_path, index=False)

print(f"\nSaved → {output_path}")

# ── Sample check ───────────────────────────────────────────────────────────────
print("\nSample rows:")
print(df_new.head(5))

print("\nSTEP 02.5 COMPLETE — Ready for adversarial preprocessing")
