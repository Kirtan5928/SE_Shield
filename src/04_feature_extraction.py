import os
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
SPLITS_DIR    = os.path.join(BASE_DIR, 'data', 'splits')
MODELS_DIR    = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD TRAIN/TEST SPLITS
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1 — LOADING TRAIN/TEST SPLITS")
print("=" * 65)

X_train = joblib.load(os.path.join(SPLITS_DIR, 'X_train.pkl'))
X_test  = joblib.load(os.path.join(SPLITS_DIR, 'X_test.pkl'))
y_train = joblib.load(os.path.join(SPLITS_DIR, 'y_train.pkl'))
y_test  = joblib.load(os.path.join(SPLITS_DIR, 'y_test.pkl'))

print(f"X_train : {len(X_train):,} rows")
print(f"X_test  : {len(X_test):,} rows")
print(f"y_train : Attack={y_train.sum():,}  Legit={(y_train==0).sum():,}")
print(f"y_test  : Attack={y_test.sum():,}   Legit={(y_test==0).sum():,}")
print(f"\nSource  : data/processed/merged_dataset_adv.csv")
print(f"          (adversarially preprocessed — L33t + word split + augmentation)")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — TF-IDF VECTORIZATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 2 — TF-IDF VECTORIZATION")
print("=" * 65)
print("""
Parameters:
  max_features = 10,000    top 10k unigrams + bigrams
  ngram_range  = (1, 2)    captures single words + two-word phrases
  min_df       = 2         word must appear in at least 2 documents
  max_df       = 0.95      ignore words appearing in 95%+ of docs
  sublinear_tf = True      log(tf) scaling — reduces impact of very
                           frequent words (better for SVM)

Why these settings:
  Bigrams capture manipulation phrases like "verify account",
  "urgent action", "click here", "suspended immediately" that
  unigrams alone would miss.

  sublinear_tf is specifically recommended for SVM on text
  (Joachims 1998) — standard in NLP+SVM literature.
""")

tfidf = TfidfVectorizer(
    max_features = 10000,
    ngram_range  = (1, 2),
    min_df       = 2,
    max_df       = 0.95,
    sublinear_tf = True,
    strip_accents = 'unicode',
    analyzer     = 'word',
    token_pattern = r'\b[a-zA-Z][a-zA-Z]+\b'  # letters only, min 2 chars
)

print("Fitting TF-IDF on training data...")
X_train_tfidf = tfidf.fit_transform(X_train)

print("Transforming test data...")
X_test_tfidf  = tfidf.transform(X_test)

print(f"\nTF-IDF matrix shape (train) : {X_train_tfidf.shape}")
print(f"TF-IDF matrix shape (test)  : {X_test_tfidf.shape}")
print(f"Vocabulary size             : {len(tfidf.vocabulary_):,}")
print(f"Matrix density              : "
      f"{X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]) * 100:.4f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — VOCABULARY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 3 — VOCABULARY ANALYSIS")
print("=" * 65)

feature_names = tfidf.get_feature_names_out()
idf_scores    = tfidf.idf_

# Most common — low IDF, appear in nearly every document
top_common_idx = np.argsort(idf_scores)[:30]
print("\nTop 30 most common features (low IDF — appear everywhere):")
print([feature_names[i] for i in top_common_idx])

# Most unique — high IDF, rare but informative
top_unique_idx = np.argsort(idf_scores)[-30:]
print("\nTop 30 most informative features (high IDF — rare + specific):")
print([feature_names[i] for i in top_unique_idx])

# SE-specific signal words — check if they're captured
se_signals = [
    'verify', 'urgent', 'account', 'suspended', 'password',
    'credential', 'click', 'confirm', 'immediately', 'security',
    'verify account', 'click here', 'urgent action', 'account suspended'
]
print("\nSocial engineering signal words captured in vocabulary:")
for word in se_signals:
    if word in tfidf.vocabulary_:
        idx = tfidf.vocabulary_[word]
        print(f"  ✅  '{word}'  (IDF={idf_scores[idx]:.3f})")
    else:
        print(f"  ❌  '{word}'  NOT in vocabulary")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — SAVE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 4 — SAVING")
print("=" * 65)

joblib.dump(tfidf,         os.path.join(MODELS_DIR,  'tfidf_vectorizer.pkl'))
joblib.dump(X_train_tfidf, os.path.join(SPLITS_DIR,  'X_train_tfidf.pkl'))
joblib.dump(X_test_tfidf,  os.path.join(SPLITS_DIR,  'X_test_tfidf.pkl'))
joblib.dump(y_train,       os.path.join(SPLITS_DIR,  'y_train.pkl'))
joblib.dump(y_test,        os.path.join(SPLITS_DIR,  'y_test.pkl'))

print("Saved → models/tfidf_vectorizer.pkl")
print("Saved → data/splits/X_train_tfidf.pkl")
print("Saved → data/splits/X_test_tfidf.pkl")
print("Saved → data/splits/y_train.pkl")
print("Saved → data/splits/y_test.pkl")

print(f"""
Summary:
  Train matrix : {X_train_tfidf.shape[0]:,} samples × {X_train_tfidf.shape[1]:,} features
  Test matrix  : {X_test_tfidf.shape[0]:,} samples × {X_test_tfidf.shape[1]:,} features
  Vectorizer   : saved and ready for inference pipeline

TF-IDF complete. Features are FIXED for all SVM experiments.
Next → 05_train_models.py
""")