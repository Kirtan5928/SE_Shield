import os
import joblib
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     roc_auc_score, confusion_matrix,
                                     classification_report)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(BASE_DIR, 'data', 'splits')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Load TF-IDF features ──────────────────────────────────────────────────────
print("=" * 65)
print("LOADING TF-IDF FEATURES — FIXED ACROSS ALL LR EXPERIMENTS")
print("=" * 65)

X_train = joblib.load(os.path.join(SPLITS_DIR, 'X_train_tfidf.pkl'))
X_test  = joblib.load(os.path.join(SPLITS_DIR, 'X_test_tfidf.pkl'))
y_train = joblib.load(os.path.join(SPLITS_DIR, 'y_train.pkl'))
y_test  = joblib.load(os.path.join(SPLITS_DIR, 'y_test.pkl'))

X_train = X_train.tocsr()
X_test  = X_test.tocsr()

print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"y_train : Attack={y_train.sum():,}  Legit={(y_train==0).sum():,}")
print(f"y_test  : Attack={y_test.sum():,}   Legit={(y_test==0).sum():,}")
print("\nFeatures fixed — only LR tuning strategies differ.")

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED EVALUATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(name: str, model, X_te, y_te) -> dict:
    t0     = time.time()
    y_pred = model.predict(X_te)
    infer  = (time.time() - t0) * 1000 / len(y_te)

    y_prob = model.predict_proba(X_te)[:, 1]
    auc    = roc_auc_score(y_te, y_prob)
    acc    = accuracy_score (y_te, y_pred)
    prec   = precision_score(y_te, y_pred, average='macro', zero_division=0)
    rec    = recall_score   (y_te, y_pred, average='macro', zero_division=0)
    f1     = f1_score       (y_te, y_pred, average='macro', zero_division=0)
    cm     = confusion_matrix(y_te, y_pred)

    print(f"\n{'─'*65}")
    print(f"  RESULTS — {name}")
    print(f"{'─'*65}")
    print(f"  Accuracy   : {acc*100:.2f}%")
    print(f"  Precision  : {prec*100:.2f}%  (macro)")
    print(f"  Recall     : {rec*100:.2f}%   (macro)")
    print(f"  F1 Score   : {f1*100:.2f}%   (macro)")
    print(f"  ROC-AUC    : {auc*100:.2f}%")
    print(f"  Infer/samp : {infer:.4f} ms")
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix:")
    print(f"    TP={tp:,}  FP={fp:,}")
    print(f"    FN={fn:,}  TN={tn:,}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_te, y_pred,
        target_names=['Legitimate', 'Attack'],
        digits=4
    ))

    return {
        'name'      : name,
        'accuracy'  : acc,
        'precision' : prec,
        'recall'    : rec,
        'f1_score'  : f1,
        'roc_auc'   : auc,
        'infer_ms'  : infer,
        'y_pred'    : y_pred,
        'y_prob'    : y_prob,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — BASELINE (Enron Paper 1)
# Fixed params: C=1.0, L2, max_iter=1000 — no search
# ═══════════════════════════════════════════════════════════════════════════════
def experiment_baseline(X_tr, y_tr, X_te, y_te) -> dict:
    print("\n" + "=" * 65)
    print("EXPERIMENT 1 — BASELINE (Enron Paper 1)")
    print("=" * 65)
    print("  Model   : LogisticRegression")
    print("  Params  : C=1.0, penalty=l2, max_iter=1000")
    print("  Tuning  : NONE — fixed params as specified in paper")

    model = LogisticRegression(
        C           = 1.0,
        penalty     = 'l2',
        max_iter    = 1000,
        random_state= 42,
        solver      = 'lbfgs',
        n_jobs      = -1
    )

    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = (time.time() - t0) * 1000
    print(f"\n  Train time : {train_time:.0f} ms")

    result = evaluate("Baseline LR — Enron Paper 1 (C=1.0, no tuning)",
                      model, X_te, y_te)
    result['train_time_ms'] = train_time
    result['best_params']   = "C=1.0, penalty=l2 (fixed — no search)"

    joblib.dump(model, os.path.join(MODELS_DIR, 'lr_baseline.pkl'))
    print("  Saved → models/lr_baseline.pkl")
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — ENRON PAPER 2
# GridSearchCV over C = [0.1, 1, 10, 100], default scoring
# ═══════════════════════════════════════════════════════════════════════════════
def experiment_enron2(X_tr, y_tr, X_te, y_te) -> dict:
    print("\n" + "=" * 65)
    print("EXPERIMENT 2 — ENRON PAPER 2 (GridSearch C, default scoring)")
    print("=" * 65)

    param_grid = {'C': [0.1, 1, 10, 100]}

    print(f"  Model      : LogisticRegression")
    print(f"  Tuning     : GridSearchCV")
    print(f"  Param grid : {param_grid}")
    print(f"  Scoring    : accuracy (paper default)")
    print(f"  CV folds   : 5 (stratified)")

    base_lr = LogisticRegression(
        penalty     = 'l2',
        max_iter    = 1000,
        random_state= 42,
        solver      = 'lbfgs',
        n_jobs      = -1
    )
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        base_lr, param_grid,
        scoring = 'accuracy',
        cv      = cv,
        n_jobs  = -1,
        verbose = 1,
        refit   = True
    )

    t0 = time.time()
    grid.fit(X_tr, y_tr)
    train_time = (time.time() - t0) * 1000

    print(f"\n  Best params    : {grid.best_params_}")
    print(f"  Best CV score  : {grid.best_score_*100:.2f}% (accuracy)")
    print(f"  Train time     : {train_time/1000:.1f}s")

    result = evaluate("Enron Paper 2 — GridSearch C=[0.1,1,10,100]",
                      grid.best_estimator_, X_te, y_te)
    result['train_time_ms'] = train_time
    result['best_params']   = str(grid.best_params_)
    result['cv_best_score'] = grid.best_score_

    joblib.dump(grid.best_estimator_,
                os.path.join(MODELS_DIR, 'lr_enron2.pkl'))
    print("  Saved → models/lr_enron2.pkl")
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — ENRON PAPER 3
# GridSearchCV over C = [0.1, 1, 10], scoring = f1_macro
# ═══════════════════════════════════════════════════════════════════════════════
def experiment_enron3(X_tr, y_tr, X_te, y_te) -> dict:
    print("\n" + "=" * 65)
    print("EXPERIMENT 3 — ENRON PAPER 3 (GridSearch C, f1_macro scoring)")
    print("=" * 65)

    param_grid = {'C': [0.1, 1, 10]}

    print(f"  Model      : LogisticRegression")
    print(f"  Tuning     : GridSearchCV")
    print(f"  Param grid : {param_grid}")
    print(f"  Scoring    : f1_macro (explicitly specified in paper)")
    print(f"  CV folds   : 5 (stratified)")
    print(f"\n  Note: f1_macro scoring is better than accuracy for")
    print(f"        imbalanced classes — directly optimises F1")

    base_lr = LogisticRegression(
        penalty     = 'l2',
        max_iter    = 1000,
        random_state= 42,
        solver      = 'lbfgs',
        n_jobs      = -1
    )
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        base_lr, param_grid,
        scoring = 'f1_macro',
        cv      = cv,
        n_jobs  = -1,
        verbose = 1,
        refit   = True
    )

    t0 = time.time()
    grid.fit(X_tr, y_tr)
    train_time = (time.time() - t0) * 1000

    print(f"\n  Best params    : {grid.best_params_}")
    print(f"  Best CV F1     : {grid.best_score_*100:.2f}% (f1_macro)")
    print(f"  Train time     : {train_time/1000:.1f}s")

    result = evaluate("Enron Paper 3 — GridSearch C=[0.1,1,10] f1_macro",
                      grid.best_estimator_, X_te, y_te)
    result['train_time_ms'] = train_time
    result['best_params']   = str(grid.best_params_)
    result['cv_best_score'] = grid.best_score_

    joblib.dump(grid.best_estimator_,
                os.path.join(MODELS_DIR, 'lr_enron3.pkl'))
    print("  Saved → models/lr_enron3.pkl")
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    print("\n" + "=" * 65)
    print("RUNNING ALL 3 LR TUNING EXPERIMENTS")
    print("Fixed : TF-IDF features, same dataset, same train/test split")
    print("Varies: C regularization tuning strategy only")
    print("=" * 65)

    total_start = time.time()

    results = []
    results.append(experiment_baseline(X_train, y_train, X_test, y_test))
    results.append(experiment_enron2  (X_train, y_train, X_test, y_test))
    results.append(experiment_enron3  (X_train, y_train, X_test, y_test))

    total_time = (time.time() - total_start) / 60

    # ── Final comparison table ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FINAL COMPARISON TABLE — LR TUNING STRATEGIES")
    print("=" * 65)

    baseline_f1 = results[0]['f1_score']
    best_f1     = max(r['f1_score'] for r in results)

    print(f"\n{'Model':<45} {'Acc':>7} {'Prec':>7} {'Rec':>7} "
          f"{'F1':>7} {'AUC':>7} {'vs Base':>8}")
    print("-" * 95)

    for r in results:
        delta     = r['f1_score'] - baseline_f1
        delta_str = (f"+{delta*100:.2f}%"
                     if delta >= 0 else f"{delta*100:.2f}%")
        best_mark = " ←BEST" if r['f1_score'] == best_f1 else ""
        print(
            f"{r['name']:<45} "
            f"{r['accuracy']*100:>6.2f}% "
            f"{r['precision']*100:>6.2f}% "
            f"{r['recall']*100:>6.2f}% "
            f"{r['f1_score']*100:>6.2f}% "
            f"{r['roc_auc']*100:>6.2f}% "
            f"{delta_str:>8}"
            f"{best_mark}"
        )

    # ── Best params summary ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("BEST HYPERPARAMETERS")
    print("=" * 65)
    for r in results:
        print(f"\n  {r['name']}")
        print(f"    {r['best_params']}")

    # ── Analysis ───────────────────────────────────────────────────────────────
    best    = max(results, key=lambda x: x['f1_score'])
    fastest = min(results, key=lambda x: x['train_time_ms'])

    print("\n" + "=" * 65)
    print("ANALYSIS")
    print("=" * 65)
    print(f"""
  Best tuning strategy  : {best['name']}
  Best F1 score         : {best['f1_score']*100:.2f}%
  Improvement vs base   : +{(best['f1_score']-baseline_f1)*100:.2f}%

  Fastest to train      : {fastest['name']}
  Total experiment time : {total_time:.1f} minutes

  Key insight:
  — Baseline (C=1.0)    : no search, fastest, good default
  — Enron Paper 2       : broader C range, accuracy-optimised search
  — Enron Paper 3       : tighter C range, F1-optimised search
                          better for imbalanced data scenarios

  For your project:
  — If Enron Paper 3 wins → use C from f1_macro search
    justification: directly optimises the metric that matters
    most for SE detection (catching attacks = high recall/F1)
  — If baseline ties with tuned → C=1.0 is already optimal
    for this dataset, tuning adds no meaningful gain
    """)

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump({
        'results'    : [{k: v for k, v in r.items()
                         if k not in ['y_pred', 'y_prob']}
                        for r in results],
        'y_test'     : y_test,
        'all_y_pred' : {r['name']: r['y_pred'] for r in results},
        'all_y_prob' : {r['name']: r['y_prob'] for r in results},
        'winner'     : best['name'],
    }, os.path.join(MODELS_DIR, 'lr_comparison_results.pkl'))

    print("Saved → models/lr_comparison_results.pkl")
    print("\nAll LR experiments complete.")