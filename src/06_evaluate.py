import os
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import time

from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, auc)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
SPLITS_DIR  = os.path.join(BASE_DIR, 'data', 'splits')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'svm'), exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD EVERYTHING
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1 — LOADING MODEL + DATA")
print("=" * 65)

X_test = joblib.load(os.path.join(SPLITS_DIR, 'X_test_tfidf.pkl')).tocsr()
y_test = joblib.load(os.path.join(SPLITS_DIR, 'y_test.pkl'))

# Best SVM — LinearSVC baseline (all strategies tied at 98.85%)
svm_model = joblib.load(os.path.join(MODELS_DIR, 'svm_paper10_baseline.pkl'))

print(f"SVM model : svm_paper10_baseline.pkl  (LinearSVC, default)")
print(f"X_test    : {X_test.shape}")
print(f"y_test    : Attack={y_test.sum():,}  Legit={(y_test==0).sum():,}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — SVM PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 2 — SVM PREDICTIONS")
print("=" * 65)

t0       = time.time()
svm_pred = svm_model.predict(X_test)
svm_prob = svm_model.predict_proba(X_test)[:, 1]
svm_time = (time.time() - t0) * 1000 / len(y_test)

print(f"SVM inference : {svm_time:.4f} ms/sample")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — CONFIDENCE SCORE CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 3 — CONFIDENCE SCORE CALCULATION")
print("=" * 65)
print("""
  Two confidence measures implemented:

  1. DISTANCE CONFIDENCE
     Formula : confidence = |P(attack) - threshold| x 200
     Range   : 0-100
     Meaning : how far the prediction is from the decision
               boundary. Score near 50% = uncertain.
               Score near 0% or 100% = highly confident.

     Examples:
       P=0.97 -> |0.97-0.5|x200 = 94  (very confident attack)
       P=0.51 -> |0.51-0.5|x200 =  2  (barely certain - flag for review)
       P=0.03 -> |0.03-0.5|x200 = 94  (very confident benign)

  2. ENTROPY CONFIDENCE
     Formula : entropy = -P*log(P) - (1-P)*log(1-P)
               confidence = (1 - entropy/log(2)) * 100
     Range   : 0-100
     Meaning : how uncertain the model is. Maximum entropy
               at P=0.5 means 50/50 - model has no idea.
               Zero entropy means complete certainty.
""")

THRESHOLD = 0.5
eps       = 1e-10  # prevent log(0)

# Distance confidence
distance_conf = np.abs(svm_prob - THRESHOLD) * 200
distance_conf = np.clip(distance_conf, 0, 100)

# Entropy confidence
p            = np.clip(svm_prob, eps, 1 - eps)
ent          = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
entropy_conf = (1 - ent) * 100
entropy_conf = np.clip(entropy_conf, 0, 100)

print(f"  Confidence stats (distance method):")
print(f"    Mean   : {distance_conf.mean():.1f}")
print(f"    Median : {np.median(distance_conf):.1f}")
print(f"    < 20   : {(distance_conf < 20).sum():,} samples  (uncertain - review)")
print(f"    >= 80  : {(distance_conf >= 80).sum():,} samples  (high confidence)")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — METRICS COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 4 — METRICS COMPUTATION")
print("=" * 65)

acc  = accuracy_score (y_test, svm_pred)
prec = precision_score(y_test, svm_pred, average='macro', zero_division=0)
rec  = recall_score   (y_test, svm_pred, average='macro', zero_division=0)
f1   = f1_score       (y_test, svm_pred, average='macro', zero_division=0)
auc_ = roc_auc_score  (y_test, svm_prob)
cm   = confusion_matrix(y_test, svm_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n  Accuracy   : {acc*100:.2f}%")
print(f"  Precision  : {prec*100:.2f}%  (macro)")
print(f"  Recall     : {rec*100:.2f}%   (macro)")
print(f"  F1 Score   : {f1*100:.2f}%   (macro)")
print(f"  ROC-AUC    : {auc_*100:.2f}%")
print(f"  Infer/samp : {svm_time:.4f} ms")
print(f"\n  Confusion Matrix:")
print(f"    TP={tp:,}  FP={fp:,}")
print(f"    FN={fn:,}  TN={tn:,}")
print(f"\n  False Positive Rate : {fp/(fp+tn)*100:.2f}%")
print(f"  False Negative Rate : {fn/(fn+tp)*100:.2f}%")

print("\n  Classification Report:")
print(classification_report(
    y_test, svm_pred,
    target_names=['Legitimate', 'Attack'],
    digits=4
))

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — THRESHOLD ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 5 — THRESHOLD ANALYSIS")
print("=" * 65)
print("""
  Threshold controls how aggressive the SVM probability scorer is.
  Lower threshold -> more messages flagged as attack
                  -> higher recall, higher false positive rate
  Higher threshold -> fewer messages flagged
                   -> lower recall, lower false positive rate
""")

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print(f"\n{'Threshold':>10} {'Flagged':>9} {'Acc':>7} {'Recall':>8} "
      f"{'Precision':>10} {'F1':>7} {'FPR':>7}")
print("-" * 70)

thresh_results = []
for t in thresholds:
    t_pred = (svm_prob >= t).astype(int)
    t_cm   = confusion_matrix(y_test, t_pred)
    t_tn, t_fp, t_fn, t_tp = t_cm.ravel()

    t_acc  = accuracy_score (y_test, t_pred)
    t_rec  = recall_score   (y_test, t_pred, average='macro', zero_division=0)
    t_prec = precision_score(y_test, t_pred, average='macro', zero_division=0)
    t_f1   = f1_score       (y_test, t_pred, average='macro', zero_division=0)
    t_fpr  = t_fp / (t_fp + t_tn) if (t_fp + t_tn) > 0 else 0
    flagged = t_pred.sum()

    thresh_results.append({
        'threshold': t, 'f1': t_f1, 'recall': t_rec,
        'precision': t_prec, 'fpr': t_fpr
    })

    mark = " <-recommended" if t == THRESHOLD else ""
    print(f"{t:>10.1f} {flagged:>9,} {t_acc*100:>6.2f}% "
          f"{t_rec*100:>7.2f}% {t_prec*100:>9.2f}% "
          f"{t_f1*100:>6.2f}% {t_fpr*100:>6.2f}%{mark}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 6 — GENERATING PLOTS")
print("=" * 65)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('SVM — Social Engineering Detection Evaluation',
             fontsize=16, fontweight='bold', y=1.01)

# ── Plot 1: Probability distribution ─────────────────────────────────────────
ax = axes[0, 0]
ax.hist(svm_prob[y_test == 0], bins=50, alpha=0.6, color='#3b82f6',
        label='Legitimate', density=True)
ax.hist(svm_prob[y_test == 1], bins=50, alpha=0.6, color='#ef4444',
        label='Attack',     density=True)
ax.axvline(x=THRESHOLD, color='black', linestyle='--',
           linewidth=1.5, label=f'Threshold ({THRESHOLD})')
ax.set_title('Predicted Probability Distribution', fontweight='bold')
ax.set_xlabel('P(Attack)')
ax.set_ylabel('Density')
ax.legend()
ax.grid(alpha=0.3)

# ── Plot 2: Risk score bands ──────────────────────────────────────────────────
ax    = axes[0, 1]
score = svm_prob * 100
bands = {
    'Low\n(0-25)'      : (score < 25).sum(),
    'Medium\n(25-50)'  : ((score >= 25) & (score < 50)).sum(),
    'High\n(50-75)'    : ((score >= 50) & (score < 75)).sum(),
    'Critical\n(75-100)': (score >= 75).sum(),
}
colors = ['#22c55e', '#eab308', '#f97316', '#ef4444']
bars   = ax.bar(bands.keys(), bands.values(), color=colors,
                edgecolor='black', linewidth=0.8)
for bar, val in zip(bars, bands.values()):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f'{val:,}', ha='center', fontsize=9, fontweight='bold')
ax.set_title('Risk Score Band Distribution', fontweight='bold')
ax.set_ylabel('Sample Count')
ax.grid(axis='y', alpha=0.3)

# ── Plot 3: ROC curve ─────────────────────────────────────────────────────────
ax             = axes[0, 2]
fpr_c, tpr_c, _ = roc_curve(y_test, svm_prob)
auc_c          = auc(fpr_c, tpr_c)
ax.plot(fpr_c, tpr_c, color='#3b82f6', lw=2,
        label=f'SVM (AUC = {auc_c:.4f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
ax.set_title('ROC Curve', fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
ax.grid(alpha=0.3)

# ── Plot 4: Confidence distribution ──────────────────────────────────────────
ax = axes[1, 0]
ax.hist(distance_conf[y_test == 1], bins=50, alpha=0.6,
        color='#ef4444', label='Attack',     density=True)
ax.hist(distance_conf[y_test == 0], bins=50, alpha=0.6,
        color='#3b82f6', label='Legitimate', density=True)
ax.axvline(x=20, color='orange', linestyle='--',
           linewidth=1.5, label='Low confidence (<20)')
ax.set_title('Confidence Score Distribution\n|P - threshold| x 200',
             fontweight='bold')
ax.set_xlabel('Confidence Score (0-100)')
ax.set_ylabel('Density')
ax.legend()
ax.grid(alpha=0.3)

# ── Plot 5: Threshold analysis ────────────────────────────────────────────────
ax      = axes[1, 1]
t_vals  = [r['threshold']    for r in thresh_results]
f1_vals = [r['f1'] * 100     for r in thresh_results]
rc_vals = [r['recall'] * 100 for r in thresh_results]
pr_vals = [r['precision'] * 100 for r in thresh_results]
ax.plot(t_vals, f1_vals, 'o-', color='#8b5cf6', lw=2, label='F1')
ax.plot(t_vals, rc_vals, 's-', color='#ef4444', lw=2, label='Recall')
ax.plot(t_vals, pr_vals, '^-', color='#3b82f6', lw=2, label='Precision')
ax.axvline(x=THRESHOLD, color='black', linestyle='--',
           linewidth=1.5, label=f'Chosen ({THRESHOLD})')
ax.set_title('Threshold Analysis', fontweight='bold')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score (%)')
ax.legend()
ax.grid(alpha=0.3)

# ── Plot 6: Confusion matrix ──────────────────────────────────────────────────
ax = axes[1, 2]
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
            xticklabels=['Legit', 'Attack'],
            yticklabels=['Legit', 'Attack'],
            annot_kws={'size': 13})
ax.set_title('Confusion Matrix — SVM', fontweight='bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, 'svm', 'svm_evaluation.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved -> results/svm/svm_evaluation.png")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 7 — FINAL SUMMARY")
print("=" * 65)
print(f"""
  SVM — LinearSVC Baseline

    Accuracy   : {acc*100:.2f}%
    Precision  : {prec*100:.2f}%  (macro)
    Recall     : {rec*100:.2f}%   (macro)
    F1 Score   : {f1*100:.2f}%   (macro)
    ROC-AUC    : {auc_*100:.2f}%
    Speed      : {svm_time:.4f} ms/sample

    Confusion Matrix:
      TP={tp:,}  FP={fp:,}
      FN={fn:,}  TN={tn:,}
      FPR : {fp/(fp+tn)*100:.2f}%
      FNR : {fn/(fn+tp)*100:.2f}%

  Confidence scoring : |P - 0.5| x 200
  Risk bands         : Low(0-25)  Medium(25-50)  High(50-75)  Critical(75-100)
  Recommended threshold : {THRESHOLD}
""")

# ── Save evaluation summary ───────────────────────────────────────────────────
joblib.dump({
    'svm_pred'      : svm_pred,
    'svm_prob'      : svm_prob,
    'distance_conf' : distance_conf,
    'entropy_conf'  : entropy_conf,
    'y_test'        : y_test,
    'threshold'     : THRESHOLD,
    'metrics'       : {
        'accuracy'  : acc,
        'precision' : prec,
        'recall'    : rec,
        'f1_score'  : f1,
        'roc_auc'   : auc_,
        'infer_ms'  : svm_time,
    },
}, os.path.join(MODELS_DIR, 'svm_evaluation_summary.pkl'))

print("Saved -> models/svm_evaluation_summary.pkl")
print("\nEvaluation complete.")