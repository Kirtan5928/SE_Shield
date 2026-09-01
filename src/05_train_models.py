import os
import joblib
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.utils import resample

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("  [INFO] tqdm not installed — run 'pip install tqdm' for progress bars")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(BASE_DIR, 'data', 'splits')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Mitigation config ─────────────────────────────────────────────────────────
# Fraction of X_train used during GridSearch CV.
# Best params found on this subsample, final model refitted on full X_train.
# 0.25 = 25% subsample → ~4× speedup on grid search wall time.
SUBSAMPLE_FRACTION = 0.25

# ── Load TF-IDF features ──────────────────────────────────────────────────────
print("=" * 65)
print("LOADING TF-IDF FEATURES — FIXED ACROSS ALL EXPERIMENTS")
print("=" * 65)

X_train = joblib.load(os.path.join(SPLITS_DIR, 'X_train_tfidf.pkl'))
X_test = joblib.load(os.path.join(SPLITS_DIR, 'X_test_tfidf.pkl'))
y_train = joblib.load(os.path.join(SPLITS_DIR, 'y_train.pkl'))
y_test = joblib.load(os.path.join(SPLITS_DIR, 'y_test.pkl'))

X_train = X_train.tocsr()
X_test = X_test.tocsr()

print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"y_train : Attack={y_train.sum():,}  Legit={(y_train == 0).sum():,}")
print(f"y_test  : Attack={y_test.sum():,}   Legit={(y_test == 0).sum():,}")

# ── Build stratified subsample for grid search (Mitigation 2) ─────────────────
n_subsample = int(len(y_train) * SUBSAMPLE_FRACTION)
sub_idx = resample(
    np.arange(len(y_train)),
    n_samples=n_subsample,
    stratify=y_train,
    random_state=42,
    replace=False,
)
X_search = X_train[sub_idx]
y_search = y_train.iloc[sub_idx]

print(f"\n  Subsample for grid search ({int(SUBSAMPLE_FRACTION * 100)}% of train):")
print(f"    X_search : {X_search.shape}")
print(f"    Attack={y_search.sum():,}  Legit={(y_search == 0).sum():,}")
print(f"\nFeatures fixed — only tuning strategies differ across experiments.")
print(f"Mitigations active:")
print(f"  [1] LinearSVC replaces kernel SVC in Exp 2 & 3 — O(n×f) vs O(n²×f)")
print(f"  [2] Grid search runs on {int(SUBSAMPLE_FRACTION * 100)}% subsample; final model refits on full train set")


# ── tqdm-compatible GridSearchCV wrapper ──────────────────────────────────────
class ProgressGridSearchCV(GridSearchCV):
    """
    GridSearchCV subclass that shows a tqdm progress bar.    Falls back silently if tqdm is not installed."""

    def fit(self, X, y=None, **fit_params):
        if not TQDM_AVAILABLE:
            return super().fit(X, y, **fit_params)

        total = (
            len(self.param_grid) if isinstance(self.param_grid, list)
            else np.prod([len(v) for v in self.param_grid.values()])
        )
        n_splits = self.cv.get_n_splits() if hasattr(self.cv, 'get_n_splits') else 5
        total_fits = int(total * n_splits)

        with tqdm(total=total_fits, desc="  Grid search", unit="fit",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} fits [{elapsed}<{remaining}]") as pbar:
            self.set_params(verbose=0)
            result = super().fit(X, y, **fit_params)
            pbar.update(total_fits - pbar.n)
        return result

    # ═══════════════════════════════════════════════════════════════════════════════


# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(name: str, model, X_te, y_te) -> dict:
    t0 = time.time()
    y_pred = model.predict(X_te)
    infer = (time.time() - t0) * 1000 / len(y_te)

    try:
        y_prob = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob)
    except Exception:
        y_prob = y_pred.astype(float)
        auc = roc_auc_score(y_te, y_prob)

    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_te, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_te, y_pred)

    print(f"\n{'─' * 65}")
    print(f"  RESULTS — {name}")
    print(f"{'─' * 65}")
    print(f"  Accuracy   : {acc * 100:.2f}%")
    print(f"  Precision  : {prec * 100:.2f}%  (macro)")
    print(f"  Recall     : {rec * 100:.2f}%   (macro)")
    print(f"  F1 Score   : {f1 * 100:.2f}%   (macro)")
    print(f"  ROC-AUC    : {auc * 100:.2f}%")
    print(f"  Infer/samp : {infer:.4f} ms")
    print(f"\n  Confusion Matrix:")
    tn, fp, fn, tp = cm.ravel()
    print(f"    TP={tp:,}  FP={fp:,}")
    print(f"    FN={fn:,}  TN={tn:,}")
    print(f"\n  Classification Report:")
    print(classification_report(y_te, y_pred, target_names=['Legitimate', 'Attack']))

    return {
        'name': name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': auc,
        'infer_ms': infer,
        'y_pred': y_pred,
        'y_prob': y_prob,
    }


def refit_on_full(best_C: float, X_full, y_full) -> tuple:
    """
    Refit CalibratedClassifierCV(LinearSVC(C=best_C)) on the full train set.    Returns (fitted_model, refit_time_ms).    """
    model = CalibratedClassifierCV(
        LinearSVC(C=best_C, dual='auto', max_iter=3000, random_state=42)
    )
    print(f"\n  Refitting LinearSVC(C={best_C}) on full train set "
          f"({X_full.shape[0]:,} samples)...")
    if TQDM_AVAILABLE:
        with tqdm(total=1, desc="  Refitting", unit="model",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as pbar:
            t0 = time.time()
            model.fit(X_full, y_full)
            refit_ms = (time.time() - t0) * 1000
            pbar.update(1)
    else:
        t0 = time.time()
        model.fit(X_full, y_full)
        refit_ms = (time.time() - t0) * 1000

    print(f"  Refit time : {refit_ms / 1000:.1f}s")
    return model, refit_ms


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — PAPER 10 BASELINE
# LinearSVC, no tuning, default parameters
# ═══════════════════════════════════════════════════════════════════════════════
def experiment_paper10(X_tr, y_tr, X_te, y_te) -> dict:
    print("\n" + "=" * 65)
    print("EXPERIMENT 1 — PAPER 10 (Baseline LinearSVC, No Tuning)")
    print("=" * 65)
    print("  Model        : LinearSVC (default params)")
    print("  Tuning       : NONE")
    print("  Mitigations  : N/A — already O(n×f), no grid search")
    print("  Note         : This is the baseline all others are compared against")

    model = CalibratedClassifierCV(
        LinearSVC(dual='auto', max_iter=3000, random_state=42)
    )

    if TQDM_AVAILABLE:
        with tqdm(total=1, desc="  Training", unit="model",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as pbar:
            t0 = time.time()
            model.fit(X_tr, y_tr)
            train_time = (time.time() - t0) * 1000
            pbar.update(1)
    else:
        t0 = time.time()
        model.fit(X_tr, y_tr)
        train_time = (time.time() - t0) * 1000

    print(f"\n  Train time : {train_time:.0f} ms")

    result = evaluate("Paper 10 — Baseline LinearSVC", model, X_te, y_te)
    result['train_time_ms'] = train_time
    result['best_params'] = "Default — no tuning"

    joblib.dump(model, os.path.join(MODELS_DIR, 'svm_paper10_baseline.pkl'))
    print("  Saved → models/svm_paper10_baseline.pkl")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — ENRON PAPER 2
# MITIGATION 1: SVC → LinearSVC (O(n×f), no kernel matrix)
# MITIGATION 2: Grid search on SUBSAMPLE_FRACTION of train, refit on full
# ═══════════════════════════════════════════════════════════════════════════════
def experiment_enron2(X_tr, y_tr, X_se, y_se, X_te, y_te) -> dict:
    print("\n" + "=" * 65)
    print("EXPERIMENT 2 — ENRON PAPER 2 (LinearSVC + Subsample GridSearch)")
    print("=" * 65)

    param_grid = {'estimator__C': [0.1, 1, 10]}
    n_candidates = len(param_grid['estimator__C'])
    n_splits = 5

    print(f"  Model        : LinearSVC  [Mitigation 1: no kernel matrix]")
    print(f"  Tuning       : GridSearchCV on {int(SUBSAMPLE_FRACTION * 100)}% subsample  [Mitigation 2]")
    print(f"  Param grid   : C in {param_grid['estimator__C']}")
    print(f"  Scoring      : f1_macro")
    print(f"  CV folds     : {n_splits} (stratified)")
    print(f"  Search size  : {X_se.shape[0]:,} samples  (full train: {X_tr.shape[0]:,})")
    print(f"  Total fits   : {n_candidates} candidates × {n_splits} folds = {n_candidates * n_splits}")

    calibrated_base = CalibratedClassifierCV(
        LinearSVC(dual='auto', max_iter=3000, random_state=42)
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    grid = ProgressGridSearchCV(
        calibrated_base, param_grid,
        scoring='f1_macro',
        cv=cv,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    t0 = time.time()
    grid.fit(X_se, y_se)
    search_time = (time.time() - t0) * 1000

    best_C = grid.best_params_['estimator__C']
    print(f"\n  Best C (from subsample) : {best_C}")
    print(f"  Best CV F1 (subsample)  : {grid.best_score_ * 100:.2f}%")
    print(f"  Grid search time        : {search_time / 1000:.1f}s")

    final_model, refit_ms = refit_on_full(best_C, X_tr, y_tr)
    train_time = search_time + refit_ms

    result = evaluate("Enron Paper 2 — LinearSVC SubsampleGrid", final_model, X_te, y_te)
    result['train_time_ms'] = train_time
    result['best_params'] = str({'C': best_C})
    result['cv_best_score'] = grid.best_score_

    joblib.dump(final_model, os.path.join(MODELS_DIR, 'svm_enron2.pkl'))
    print("  Saved → models/svm_enron2.pkl")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — ENRON PAPER 1
# MITIGATION 1: SVC(kernel=rbf) → LinearSVC (O(n×f), no kernel matrix)
# MITIGATION 2: Grid search on SUBSAMPLE_FRACTION of train, refit on full
# ═══════════════════════════════════════════════════════════════════════════════
def experiment_enron1(X_tr, y_tr, X_se, y_se, X_te, y_te) -> dict:
    print("\n" + "=" * 65)
    print("EXPERIMENT 3 — ENRON PAPER 1 (LinearSVC + Subsample GridSearch)")
    print("=" * 65)

    # gamma dropped — not applicable to LinearSVC
    param_grid = {'estimator__C': [0.1, 1, 10, 100]}
    n_candidates = len(param_grid['estimator__C'])
    n_splits = 5

    print(f"  Model        : LinearSVC  [Mitigation 1: replaces SVC(kernel='rbf')]")
    print(f"  Note         : gamma removed — not applicable to LinearSVC")
    print(f"  Tuning       : GridSearchCV on {int(SUBSAMPLE_FRACTION * 100)}% subsample  [Mitigation 2]")
    print(f"  Param grid   : C in {param_grid['estimator__C']}")
    print(f"  Scoring      : f1_macro")
    print(f"  CV folds     : {n_splits} (stratified)")
    print(f"  Search size  : {X_se.shape[0]:,} samples  (full train: {X_tr.shape[0]:,})")
    print(f"  Total fits   : {n_candidates} candidates × {n_splits} folds = {n_candidates * n_splits}")

    calibrated_base = CalibratedClassifierCV(
        LinearSVC(dual='auto', max_iter=3000, random_state=42)
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    grid = ProgressGridSearchCV(
        calibrated_base, param_grid,
        scoring='f1_macro',
        cv=cv,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    t0 = time.time()
    grid.fit(X_se, y_se)
    search_time = (time.time() - t0) * 1000

    best_C = grid.best_params_['estimator__C']
    print(f"\n  Best C (from subsample) : {best_C}")
    print(f"  Best CV F1 (subsample)  : {grid.best_score_ * 100:.2f}%")
    print(f"  Grid search time        : {search_time / 1000:.1f}s")

    final_model, refit_ms = refit_on_full(best_C, X_tr, y_tr)
    train_time = search_time + refit_ms

    result = evaluate("Enron Paper 1 — LinearSVC SubsampleGrid", final_model, X_te, y_te)
    result['train_time_ms'] = train_time
    result['best_params'] = str({'C': best_C})
    result['cv_best_score'] = grid.best_score_

    joblib.dump(final_model, os.path.join(MODELS_DIR, 'svm_enron1.pkl'))
    print("  Saved → models/svm_enron1.pkl")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    print("\n" + "=" * 65)
    print("RUNNING ALL 3 SVM TUNING EXPERIMENTS")
    print("Fixed   : TF-IDF features, same dataset, same train/test split")
    print("Variable: tuning strategy only")
    print(f"Active mitigations:")
    print(f"  [1] LinearSVC replaces kernel SVC in Exp 2 & 3")
    print(f"  [2] Grid search on {int(SUBSAMPLE_FRACTION * 100)}% subsample; final model on full train")
    print("=" * 65)

    total_start = time.time()
    results = []

    experiments = [
        ("Exp 1: Paper 10 Baseline", experiment_paper10,
         dict(X_tr=X_train, y_tr=y_train,
              X_te=X_test, y_te=y_test)),
        ("Exp 2: Enron Paper 2", experiment_enron2,
         dict(X_tr=X_train, y_tr=y_train,
              X_se=X_search, y_se=y_search,
              X_te=X_test, y_te=y_test)),
        ("Exp 3: Enron Paper 1", experiment_enron1,
         dict(X_tr=X_train, y_tr=y_train,
              X_se=X_search, y_se=y_search,
              X_te=X_test, y_te=y_test)),
    ]

    if TQDM_AVAILABLE:
        exp_bar = tqdm(
            experiments,
            desc="Overall progress",
            unit="exp",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} experiments [{elapsed}<{remaining}]"
        )
        for label, fn, kwargs in exp_bar:
            exp_bar.set_description(f"Running {label}")
            results.append(fn(**kwargs))
    else:
        for label, fn, kwargs in experiments:
            results.append(fn(**kwargs))

    total_time = (time.time() - total_start) / 60

    # ── Final comparison table ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FINAL COMPARISON TABLE — ALL SVM TUNING STRATEGIES")
    print("=" * 65)

    baseline_f1 = results[0]['f1_score']

    print(f"\n{'Model':<50} {'Acc':>7} {'Prec':>7} {'Rec':>7} "
          f"{'F1':>7} {'AUC':>7} {'vs Base':>8}")
    print("-" * 100)

    for r in results:
        delta = r['f1_score'] - baseline_f1
        delta_str = f"+{delta * 100:.2f}%" if delta >= 0 else f"{delta * 100:.2f}%"
        best_mark = " ←BEST" if r['f1_score'] == max(x['f1_score'] for x in results) else ""
        print(
            f"{r['name']:<50} "
            f"{r['accuracy'] * 100:>6.2f}% "
            f"{r['precision'] * 100:>6.2f}% "
            f"{r['recall'] * 100:>6.2f}% "
            f"{r['f1_score'] * 100:>6.2f}% "
            f"{r['roc_auc'] * 100:>6.2f}% "
            f"{delta_str:>8}"
            f"{best_mark}"
        )

        # ── Best params summary ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("BEST HYPERPARAMETERS PER EXPERIMENT")
    print("=" * 65)
    for r in results:
        print(f"\n  {r['name']}")
        print(f"    {r['best_params']}")

        # ── Analysis ───────────────────────────────────────────────────────────────
    best = max(results, key=lambda x: x['f1_score'])
    worst = min(results, key=lambda x: x['f1_score'])
    fastest = min(results, key=lambda x: x['train_time_ms'])

    print("\n" + "=" * 65)
    print("ANALYSIS")
    print("=" * 65)
    print(f"""  
  Best tuning strategy  : {best['name']}  
  Best F1 score         : {best['f1_score'] * 100:.2f}%  
  Improvement vs base   : +{(best['f1_score'] - baseline_f1) * 100:.2f}%  

  Weakest strategy      : {worst['name']}  
  Weakest F1 score      : {worst['f1_score'] * 100:.2f}%  

  Fastest to train      : {fastest['name']}  
  Fastest train time    : {fastest['train_time_ms'] / 1000:.1f}s  

  Total experiment time : {total_time:.1f} minutes  
  Trade-off summary:  — Paper 10 (baseline) : fastest, no tuning, lowest F1  — Enron Paper 2       : LinearSVC, C tuned on subsample  — Enron Paper 1       : LinearSVC, C tuned on subsample (wider C range)  
  Mitigation impact:  — LinearSVC swap      : O(n²×f) → O(n×f), eliminates kernel matrix  — Subsample search    : grid search data reduced to {int(SUBSAMPLE_FRACTION * 100)}% of train set  
  — Combined speedup    : estimated 20–50× vs original kernel SVC grid search    """)

    # ── Save all results ───────────────────────────────────────────────────────
    joblib.dump({
        'results': [{k: v for k, v in r.items()
                     if k not in ['y_pred', 'y_prob']}
                    for r in results],
        'y_test': y_test,
        'all_y_pred': {r['name']: r['y_pred'] for r in results},
        'all_y_prob': {r['name']: r['y_prob'] for r in results},
        'winner': best['name'],
    }, os.path.join(MODELS_DIR, 'svm_comparison_results.pkl'))

    print("Saved → models/svm_comparison_results.pkl")
    print("\nAll experiments complete.")