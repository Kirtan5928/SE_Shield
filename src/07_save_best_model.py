import os
import joblib
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import f1_score, roc_auc_score

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SPLITS_DIR = os.path.join(BASE_DIR, 'data', 'splits')

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD EVALUATION SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1 — LOADING EVALUATION SUMMARY")
print("=" * 65)

summary = joblib.load(os.path.join(MODELS_DIR, 'evaluation_summary.pkl'))
y_test  = summary['y_test']

print(f"Loaded evaluation summary")
print(f"Threshold used    : {summary['threshold']}")
print(f"Routed to LR      : {summary['n_routed_to_lr']:,}")
print(f"Bypassed by SVM   : {summary['n_bypassed']:,}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — SELECT BEST SVM MODEL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 2 — SELECT BEST SVM MODEL (Stage 1A)")
print("=" * 65)

svm_candidates = {
    'svm_paper10_baseline' : 'LinearSVC default     — F1=98.85% AUC=99.91%',
    'svm_enron2'           : 'LinearSVC C=1 tuned   — F1=98.85% AUC=99.91%',
    'svm_enron1'           : 'LinearSVC C=1 tuned   — F1=98.85% AUC=99.91%',
}

print("\nAll SVM models tied on F1 and AUC.")
print("Selection criterion: fastest inference + simplest model")
print("→ Winner: svm_paper10_baseline (default LinearSVC)")
print("  Justification: identical performance, no tuning overhead,")
print("  fastest to retrain on new data, most interpretable config")

best_svm_name  = 'svm_paper10_baseline'
best_svm_model = joblib.load(
    os.path.join(MODELS_DIR, f'{best_svm_name}.pkl')
)
print(f"\nSelected SVM : {best_svm_name}.pkl")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SELECT BEST LR MODEL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 3 — SELECT BEST LR MODEL (Stage 1B)")
print("=" * 65)

lr_candidates = {
    'lr_baseline' : {'f1': 0.9841, 'auc': 0.9987, 'C': 1.0,
                     'note': 'no tuning'},
    'lr_enron2'   : {'f1': 0.9884, 'auc': 0.9992, 'C': 10,
                     'note': 'GridSearch, accuracy scoring'},
    'lr_enron3'   : {'f1': 0.9884, 'auc': 0.9992, 'C': 10,
                     'note': 'GridSearch, f1_macro scoring'},
}

print("\nLR candidates:")
for name, m in lr_candidates.items():
    print(f"  {name:<15} F1={m['f1']*100:.2f}%  AUC={m['auc']*100:.2f}%  "
          f"C={m['C']}  ({m['note']})")

print("\nPrimary criterion   : F1-score (macro)")
print("Secondary criterion : ROC-AUC")
print("→ Enron 2 and Enron 3 tie on both metrics.")
print("Selection: lr_enron2 (C=10, accuracy-tuned)")
print("  Justification: accuracy scoring matches Stage 1A objective")
print("  (filter speed). F1-macro scoring in Enron 3 is more")
print("  suitable for imbalanced scenarios — note as future swap.")

best_lr_name  = 'lr_enron2'
best_lr_model = joblib.load(
    os.path.join(MODELS_DIR, f'{best_lr_name}.pkl')
)
print(f"\nSelected LR : {best_lr_name}.pkl")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — BUILD FINAL PIPELINE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 4 — BUILDING FINAL PIPELINE CONFIGURATION")
print("=" * 65)

final_pipeline = {
    # Models
    'svm_model'    : best_svm_model,
    'lr_model'     : best_lr_model,

    # Scoring config
    'threshold'    : 0.5,
    'risk_scoring' : 'probability * 100',

    # Risk bands
    'risk_bands'   : {
        'low'      : (0,  25),
        'medium'   : (25, 50),
        'high'     : (50, 75),
        'critical' : (75, 100),
    },

    # Confidence scoring
    'confidence_method'  : 'distance',
    'confidence_formula' : '|P(attack) - threshold| * 200',

    # Architecture
    'architecture' : {
        'stage_1a' : 'SVM — binary triage gatekeeper',
        'stage_1b' : 'LR  — probabilistic risk scorer',
        'output'   : 'risk_score (0-100) + label + confidence',
    },
}

print("\nPipeline config:")
print(f"  SVM model       : {best_svm_name}")
print(f"  LR model        : {best_lr_name}")
print(f"  Threshold       : {final_pipeline['threshold']}")
print(f"  Risk scoring    : {final_pipeline['risk_scoring']}")
print(f"  Confidence      : {final_pipeline['confidence_formula']}")
print(f"  Risk bands      : {final_pipeline['risk_bands']}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — PERFORMANCE METADATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 5 — PERFORMANCE METADATA")
print("=" * 65)

metadata = {
    'dataset' : {
        'train_rows'  : 112217,
        'test_rows'   : 28055,
        'features'    : 10000,
        'sources'     : ['enron', 'phishing', 'spam_email', 'synthetic'],
        'adversarial' : True,
    },
    'svm_performance' : {
        'model'     : best_svm_name,
        'accuracy'  : 98.95,
        'f1_macro'  : 98.85,
        'roc_auc'   : 99.91,
        'infer_ms'  : 0.0007,
        'stage1_ok' : True,
    },
    'lr_performance' : {
        'model'    : best_lr_name,
        'accuracy' : 98.93,
        'f1_macro' : 98.84,
        'roc_auc'  : 99.92,
        'infer_ms' : 0.0005,
        'best_C'   : 10,
    },
    'hybrid_performance' : {
        'threshold'          : 0.5,
        'messages_bypassed'  : int(summary['n_bypassed']),
        'messages_to_lr'     : int(summary['n_routed_to_lr']),
        'bypass_pct'         : round(
            summary['n_bypassed'] / 28055 * 100, 2),
    },
    'design_decisions' : {
        'svm_role'    : 'binary triage — eliminate obvious benign',
        'lr_role'     : 'probabilistic scorer — generate risk score',
        'why_not_svm_only'   : 'SVM has no native probability output',
        'why_not_lr_only'    : 'LR runs on all messages — slower',
        'hybrid_advantage'   : 'SVM filters ~X% traffic, LR only runs on flagged',
    },
}

print("\nMetadata summary:")
print(f"  Train rows      : {metadata['dataset']['train_rows']:,}")
print(f"  Test rows       : {metadata['dataset']['test_rows']:,}")
print(f"  SVM F1          : {metadata['svm_performance']['f1_macro']}%")
print(f"  LR F1           : {metadata['lr_performance']['f1_macro']}%")
print(f"  Messages to LR  : {metadata['hybrid_performance']['messages_to_lr']:,}")
print(f"  Bypass rate     : {metadata['hybrid_performance']['bypass_pct']}%")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — SAVE EVERYTHING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 6 — SAVING")
print("=" * 65)

# Save final pipeline
joblib.dump(
    final_pipeline,
    os.path.join(MODELS_DIR, 'final_pipeline.pkl')
)
print("Saved → models/final_pipeline.pkl")

# Save metadata as JSON (human readable)
meta_path = os.path.join(MODELS_DIR, 'pipeline_metadata.json')
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print("Saved → models/pipeline_metadata.json")

# Save SVM and LR separately for Flask API loading
joblib.dump(best_svm_model,
            os.path.join(MODELS_DIR, 'stage1a_svm_final.pkl'))
joblib.dump(best_lr_model,
            os.path.join(MODELS_DIR, 'stage1b_lr_final.pkl'))
print("Saved → models/stage1a_svm_final.pkl")
print("Saved → models/stage1b_lr_final.pkl")

print(f"""
Final model structure:
  models/
  ├── final_pipeline.pkl       ← complete pipeline (SVM + LR + config)
  ├── pipeline_metadata.json   ← human-readable metrics + design decisions
  ├── stage1a_svm_final.pkl    ← SVM only (for Flask Stage 1A)
  ├── stage1b_lr_final.pkl     ← LR only  (for Flask Stage 1B)
  └── tfidf_vectorizer.pkl     ← TF-IDF   (already saved)

Pipeline is ready for Flask API integration.
Next → Flask backend + React frontend.
""")

print("=" * 65)
print("STAGE 1 COMPLETE")
print("=" * 65)
print("""
  Stage 1A (SVM)    : ✅ trained, evaluated, saved
  Stage 1B (LR)     : ✅ trained, evaluated, saved
  Hybrid pipeline   : ✅ simulated, validated, saved
  Risk scoring      : ✅ P(attack) × 100
  Confidence scores : ✅ |P - 0.5| × 200
  Risk bands        : ✅ Low / Medium / High / Critical
  Final models      : ✅ saved to models/

  Ready for:
  Stage 2 → DistilRoBERTa fine-tuning (Colab)
  Stage 3 → Sliding window conversation memory
  Frontend → Flask API + React dashboard
""")