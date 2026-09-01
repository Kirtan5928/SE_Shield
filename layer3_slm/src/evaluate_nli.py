"""
layer3_slm/src/evaluate_nli.py
================================
Layer 3 NLI pipeline evaluation.

Evaluation strategy
-------------------
The dataset has NO sub-type labels (only binary benign/attack).
We therefore evaluate on two axes:

  1. Binary recall/precision — does NLI correctly separate attack from benign?
       TP : label=attack, pred ≠ "benign"
       FN : label=attack, pred = "benign"   ← the critical failure mode
       TN : label=benign, pred = "benign"
       FP : label=benign, pred ≠ "benign"

     Primary metric: Recall on attack class (TP / (TP+FN)).
     Layer 3 is downstream of Layer 2, so precision matters less than
     recall here — false negatives at Layer 3 are silent misses.

  2. Operational metrics:
       - Latency distribution (p50 / p95 / p99)
       - Confidence distribution (mean, std, low-confidence rate)
       - Sub-type distribution (which labels NLI assigns to attacks)

Usage
-----
  # From project root:
  python layer3_slm/src/evaluate_nli.py --split test --n 500
  python layer3_slm/src/evaluate_nli.py --split test --n 0  # all rows
  python layer3_slm/src/evaluate_nli.py --split test --n 200 --output results/layer3_eval.json

  # Force all messages through NLI regardless of Layer 2 threshold (default):
  # use --force-nli flag (sets layer2_risk_score=100 for every message)

  # Simulate real pipeline with actual Layer 2 models:
  # use --use-layer2 flag (loads pickled models and runs full L2→L3 chain)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allow running from project root or from within layer3_slm/
# ---------------------------------------------------------------------------
_HERE        = Path(__file__).resolve().parent            # layer3_slm/src/
_LAYER3_ROOT = _HERE.parent                               # layer3_slm/
_PROJECT_ROOT = _LAYER3_ROOT.parent                       # hybrid_se/

# Insert PROJECT_ROOT first (lower priority — pushed to [1])
# Insert LAYER3_ROOT second (higher priority — lands at [0])
# This ensures `from src.layer3_pipeline` resolves to layer3_slm/src/,
# not the project-level hybrid_se/src/ which has no layer3_pipeline.py.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_LAYER3_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAYER3_ROOT))

from config_layer3 import (
    ATTACK_LABELS,
    HYPOTHESIS_TEMPLATES,
    LAYER2_THRESHOLD,
    MAX_LENGTH,
    MODEL_NAME,
)
from src.layer3_pipeline import Layer3Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("evaluate_nli")

LOW_CONF_THRESHOLD = 0.35   # matches explainer.py
SPLITS_DIR = _PROJECT_ROOT / "data" / "splits"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Candidate locations searched in order when --data-path is not provided.
# Covers both the standard splits layout and the raw merged dataset.
_SPLIT_SEARCH_PATHS = [
    "{project_root}/data/splits/{split}.csv",
    "{project_root}/data/processed/{split}.csv",
    "{project_root}/layer3_slm/data/{split}.csv",
    "{project_root}/data/{split}.csv",
]
_MERGED_CANDIDATES = [
    "{project_root}/data/raw/merged_dataset_v2.csv",
    "{project_root}/data/processed/merged_dataset_v2.csv",
    "{project_root}/data/merged_dataset_v2.csv",
]


def load_split(
    split: str,
    n: int | None,
    data_path: str | None = None,
) -> pd.DataFrame:
    """
    Load a data split for evaluation.

    Search order (when data_path is not given):
      1. data/splits/{split}.csv
      2. data/processed/{split}.csv
      3. layer3_slm/data/{split}.csv
      4. data/{split}.csv
      5. Fallback: load merged_dataset_v2.csv and take a stratified sample
         labelled as the requested split.  This lets evaluation run even
         before the splitting pipeline has been executed.

    Expected CSV columns: text (or message), label
      label values: 0/1  OR  "benign"/"phishing"  OR  "benign"/"attack"
    """
    path: Path | None = None

    # ── 1. Explicit path ─────────────────────────────────────────────────────
    if data_path:
        path = Path(data_path)
        # Try as-is (works for absolute paths or correct relative paths from cwd)
        if not path.exists() and not path.is_absolute():
            # Try resolving relative to project root
            path = _PROJECT_ROOT / data_path
        if not path.exists():
            raise FileNotFoundError(
                f"\n--data-path not found.\n"
                f"  Tried (relative to cwd)         : {Path(data_path).resolve()}\n"
                f"  Tried (relative to project root): {_PROJECT_ROOT / data_path}\n\n"
                f"Hint: The auto-detected data location is:\n"
                f"  data/processed/merged_dataset_v2.csv\n\n"
                f"Try:\n"
                f"  --data-path data/processed/merged_dataset_v2.csv"
            )

    # ── 2. Search standard locations ─────────────────────────────────────────
    if path is None:
        for template in _SPLIT_SEARCH_PATHS:
            candidate = Path(
                template.format(project_root=_PROJECT_ROOT, split=split)
            )
            if candidate.exists():
                path = candidate
                break

    # ── 3. Fallback: merged dataset ──────────────────────────────────────────
    if path is None:
        merged_path: Path | None = None
        for template in _MERGED_CANDIDATES:
            candidate = Path(template.format(project_root=_PROJECT_ROOT))
            if candidate.exists():
                merged_path = candidate
                break

        if merged_path is None:
            raise FileNotFoundError(
                f"\nNo data found for split='{split}'.\n"
                f"Searched:\n"
                + "\n".join(
                    f"  {t.format(project_root=_PROJECT_ROOT, split=split)}"
                    for t in _SPLIT_SEARCH_PATHS
                )
                + "\n\nOptions:\n"
                "  1. Run your data splitting pipeline first (src/02_preprocess.py etc.)\n"
                "  2. Use --data-path /path/to/any_labelled.csv to point at your data\n"
                "  3. Place merged_dataset_v2.csv in data/raw/"
            )

        logger.info(
            "No split files found — loading merged dataset from %s and "
            "sampling for '%s' evaluation.", merged_path, split
        )
        df_all = pd.read_csv(merged_path)

        # Reproducible 70/15/15 split matching the project's split strategy
        from sklearn.model_selection import train_test_split
        split_map = {"train": 0, "val": 1, "test": 2}
        train_df, temp_df = train_test_split(
            df_all, test_size=0.30, random_state=42,
            stratify=df_all["label"] if "label" in df_all.columns else None,
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, random_state=42,
            stratify=temp_df["label"] if "label" in temp_df.columns else None,
        )
        splits = [train_df, val_df, test_df]
        df = splits[split_map[split]].reset_index(drop=True)
        logger.info("Derived %s split: %d rows from merged dataset.", split, len(df))
        path = None   # no file to log

    else:
        df = pd.read_csv(path)
        logger.info("Loaded data from %s", path)

    # ── Normalise column names ────────────────────────────────────────────────
    if "message" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"message": "text"})
    if "text" not in df.columns:
        # Last resort: use the first string column as text
        str_cols = [c for c in df.columns if df[c].dtype == object and c != "label"]
        if str_cols:
            df = df.rename(columns={str_cols[0]: "text"})
            logger.warning("Used column '%s' as text.", str_cols[0])
        else:
            raise ValueError(
                "No 'text' or 'message' column found. "
                f"Columns present: {list(df.columns)}"
            )

    # ── Normalise binary labels ───────────────────────────────────────────────
    def _to_binary(v: object) -> str:
        s = str(v).strip().lower()
        return "benign" if s in ("0", "benign") else "attack"

    df["label_binary"] = df["label"].apply(_to_binary)

    # ── Optional row limit ────────────────────────────────────────────────────
    if n and n > 0:
        df = df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)

    attack_n = (df["label_binary"] == "attack").sum()
    benign_n = (df["label_binary"] == "benign").sum()
    logger.info(
        "Evaluation set: %d rows  (attack=%d  benign=%d)",
        len(df), attack_n, benign_n,
    )
    return df


# ---------------------------------------------------------------------------
# Layer 2 loader (optional, for --use-layer2 mode)
# ---------------------------------------------------------------------------

def load_layer2_models() -> tuple:
    """Load the Layer 2 TF-IDF + SVM + LR models.

    Uses joblib.load() — the standard serialisation format for sklearn models.
    pickle.load() fails on joblib-saved files with 'invalid load key' errors
    because joblib uses a different binary format for large numpy arrays.
    joblib.load() also handles pickle-saved files, so this works either way.
    """
    try:
        import joblib
    except ImportError:
        raise ImportError("joblib is required: pip install joblib --break-system-packages")

    models_dir = _PROJECT_ROOT / "models"

    # Try the final production models first, fall back to alternatives
    tfidf_candidates = ["tfidf_vectorizer.pkl"]
    svm_candidates   = ["stage1a_svm_final.pkl", "svm.pkl"]
    lr_candidates    = ["stage1b_lr_final.pkl",  "logistic_regression.pkl"]

    def _load(candidates: list[str], label: str):
        for name in candidates:
            path = models_dir / name
            if path.exists():
                model = joblib.load(path)
                logger.info("Loaded %s from %s", label, name)
                return model
        raise FileNotFoundError(
            f"Could not find {label} model. Tried: {candidates}\n"
            f"Models directory: {models_dir}\n"
            f"Available: {[f.name for f in models_dir.glob('*.pkl')]}"
        )

    tfidf = _load(tfidf_candidates, "TF-IDF")
    svm   = _load(svm_candidates,   "SVM")
    lr    = _load(lr_candidates,    "LR")
    logger.info("Layer 2 models loaded successfully.")
    return tfidf, svm, lr


def layer2_risk(text: str, tfidf, svm, lr) -> tuple[str, int]:
    """
    Run Layer 2 pipeline on one message.
    Returns (svm_label, risk_score).
    """
    vec = tfidf.transform([text])
    svm_label = "suspicious" if svm.predict(vec)[0] else "benign"
    prob = lr.predict_proba(vec)[0][1]   # P(attack)
    risk_score = int(prob * 100)
    return svm_label, risk_score


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate(
    pipeline: Layer3Pipeline,
    df: pd.DataFrame,
    force_nli: bool = True,
    use_layer2: bool = False,
    layer2_models: tuple | None = None,
) -> dict:
    """
    Run pipeline over df rows and accumulate metrics.

    Parameters
    ----------
    force_nli    : If True, every message is sent to NLI (risk_score=100).
                   Use for pure Layer 3 evaluation.
    use_layer2   : If True, run actual Layer 2 models to get risk_score.
                   Evaluates end-to-end recall including Layer 2 gating.
    layer2_models: (tfidf, svm, lr) tuple — required if use_layer2=True.
    """
    y_true: list[str] = []     # "benign" | "attack"
    y_pred: list[str] = []     # "benign" | "attack"
    confidences: list[float] = []
    latencies:   list[float] = []
    sub_type_dist: dict[str, int] = {}
    reason_samples: list[dict] = []
    low_conf_count = 0

    for i, row in df.iterrows():
        text         = str(row["text"])
        true_binary  = row["label_binary"]      # "benign" | "attack"

        # ── Determine risk score and SVM label ───────────────────────────
        if use_layer2 and layer2_models:
            l2_label, risk_score = layer2_risk(text, *layer2_models)
        else:
            # force_nli: treat everything as suspicious so SVM gate never fires
            l2_label   = "suspicious"
            risk_score = 100 if force_nli else LAYER2_THRESHOLD + 1

        # ── Layer 3 inference ────────────────────────────────────────────
        # Pass l2_label so pipeline.run() uses SVM decision for gating.
        # This is the critical fix: Layer 3 gates on SVM label, not risk score.
        result     = pipeline.run(
            text=text,
            layer2_risk_score=risk_score,
            layer2_label=l2_label,
        )
        pred_label = result["label"]
        conf       = result["confidence"]

        # Binary collapse
        pred_binary = "benign" if pred_label == "benign" else "attack"

        y_true.append(true_binary)
        y_pred.append(pred_binary)
        confidences.append(conf)
        latencies.append(result["latency_ms"])
        sub_type_dist[pred_label] = sub_type_dist.get(pred_label, 0) + 1

        if conf < LOW_CONF_THRESHOLD:
            low_conf_count += 1

        if len(reason_samples) < 15:
            reason_samples.append({
                "true":       true_binary,
                "predicted":  pred_label,
                "confidence": conf,
                "reason":     result["reason"],
                "text":       text[:120],
            })

    # ── Confusion matrix ─────────────────────────────────────────────────
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "attack" and p == "attack")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "attack" and p == "benign")
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == "benign" and p == "benign")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == "benign" and p == "attack")

    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    accuracy  = (tp + tn) / len(y_true) if y_true else 0.0

    lat = np.array(latencies)
    conf_arr = np.array(confidences)

    return {
        "n":           len(df),
        "mode":        "force_nli" if force_nli else ("layer2_gated" if use_layer2 else "threshold_50"),
        "metrics": {
            "recall_attack":    round(recall,    4),
            "precision_attack": round(precision, 4),
            "f1_attack":        round(f1,        4),
            "accuracy":         round(accuracy,  4),
        },
        "confusion": {
            "TP": tp, "FN": fn, "TN": tn, "FP": fp,
            "note": (
                "FN = attacks called benign by Layer 3 — the primary failure mode. "
                "Recall = TP / (TP+FN)."
            ),
        },
        "sub_type_distribution": dict(
            sorted(sub_type_dist.items(), key=lambda x: -x[1])
        ),
        "latency_ms": {
            "p50":  round(float(np.percentile(lat, 50)),  1),
            "p95":  round(float(np.percentile(lat, 95)),  1),
            "p99":  round(float(np.percentile(lat, 99)),  1),
            "mean": round(float(lat.mean()),              1),
            "max":  round(float(lat.max()),               1),
        },
        "confidence_stats": {
            "mean":             round(float(conf_arr.mean()), 4),
            "std":              round(float(conf_arr.std()),  4),
            "min":              round(float(conf_arr.min()),  4),
            "max":              round(float(conf_arr.max()),  4),
            "low_conf_pct":     round(low_conf_count / len(df) * 100, 1),
            "low_conf_threshold": LOW_CONF_THRESHOLD,
        },
        "reason_samples": reason_samples,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Layer 3 NLI pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Which data split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--n", type=int, default=500,
        help="Max rows to evaluate. 0 = full split (default: 500)",
    )
    parser.add_argument(
        "--data-path", default=None,
        help=(
            "Path to any labelled CSV with columns: text (or message), label.\n"
            "Use this when splits haven't been generated yet.\n"
            "Example: --data-path data/raw/merged_dataset_v2.csv"
        ),
    )
    parser.add_argument(
        "--output", default=None,
        help="Write JSON results to this file path (optional)",
    )
    parser.add_argument(
        "--use-layer2", action="store_true",
        help="Run actual Layer 2 models (loads .pkl files). "
             "Evaluates end-to-end recall including Layer 2 gating.",
    )
    parser.add_argument(
        "--no-force-nli", action="store_true",
        help="Don't force all messages through NLI. "
             "Instead simulate L2 threshold (risk_score = threshold+1).",
    )
    args = parser.parse_args()

    n = args.n if args.n > 0 else None

    pipeline = Layer3Pipeline(
        model_name=MODEL_NAME,
        labels=ATTACK_LABELS,
        hypothesis_templates=HYPOTHESIS_TEMPLATES,
        layer2_threshold=LAYER2_THRESHOLD,
        max_length=MAX_LENGTH,
    )

    df = load_split(args.split, n=n, data_path=args.data_path)

    layer2_models = None
    if args.use_layer2:
        layer2_models = load_layer2_models()

    force_nli = not args.no_force_nli

    logger.info(
        "Starting evaluation: %d samples, mode=%s",
        len(df),
        "layer2_gated" if args.use_layer2 else ("force_nli" if force_nli else "threshold"),
    )

    t0 = time.perf_counter()
    results = evaluate(
        pipeline=pipeline,
        df=df,
        force_nli=force_nli,
        use_layer2=args.use_layer2,
        layer2_models=layer2_models,
    )
    elapsed = time.perf_counter() - t0

    results["total_wall_time_s"]    = round(elapsed, 1)
    results["throughput_msg_per_s"] = round(len(df) / elapsed, 2)

    # Pretty print
    print("\n" + "=" * 64)
    print("  LAYER 3 NLI EVALUATION RESULTS")
    print("=" * 64)

    m = results["metrics"]
    print(f"\n  Recall (attack)    : {m['recall_attack']:.4f}   ← primary metric")
    print(f"  Precision (attack) : {m['precision_attack']:.4f}")
    print(f"  F1 (attack)        : {m['f1_attack']:.4f}")
    print(f"  Accuracy           : {m['accuracy']:.4f}")

    c = results["confusion"]
    print(f"\n  Confusion  TP={c['TP']}  FN={c['FN']}  TN={c['TN']}  FP={c['FP']}")

    lat = results["latency_ms"]
    print(f"\n  Latency    p50={lat['p50']}ms  p95={lat['p95']}ms  p99={lat['p99']}ms")

    cs = results["confidence_stats"]
    print(f"  Confidence mean={cs['mean']:.3f}  std={cs['std']:.3f}  "
          f"low_conf={cs['low_conf_pct']}%")

    print(f"\n  Throughput : {results['throughput_msg_per_s']} msg/s")
    print(f"  Wall time  : {results['total_wall_time_s']}s for {results['n']} samples")

    print("\n  Sub-type distribution (Layer 3 predictions on this split):")
    for label, count in results["sub_type_distribution"].items():
        bar = "█" * int(count / results["n"] * 40)
        pct = count / results["n"] * 100
        print(f"    {label:<35} {count:>5}  ({pct:5.1f}%)  {bar}")

    print("\n  Sample reasons:")
    for s in results["reason_samples"][:5]:
        print(f"    [{s['true']} → {s['predicted']} | conf={s['confidence']:.2f}]")
        print(f"      {s['reason']}")
        print()

    print("=" * 64 + "\n")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        logger.info("Results written to %s", out)

    t0 = time.perf_counter()
    results = evaluate(
        pipeline=pipeline,
        df=df,
        force_nli=force_nli,
        use_layer2=args.use_layer2,
        layer2_models=layer2_models,
    )
    elapsed = time.perf_counter() - t0

    results["total_wall_time_s"]    = round(elapsed, 1)
    results["throughput_msg_per_s"] = round(len(df) / elapsed, 2)

    # Pretty print
    print("\n" + "=" * 64)
    print("  LAYER 3 NLI EVALUATION RESULTS")
    print("=" * 64)

    m = results["metrics"]
    print(f"\n  Recall (attack)    : {m['recall_attack']:.4f}   ← primary metric")
    print(f"  Precision (attack) : {m['precision_attack']:.4f}")
    print(f"  F1 (attack)        : {m['f1_attack']:.4f}")
    print(f"  Accuracy           : {m['accuracy']:.4f}")

    c = results["confusion"]
    print(f"\n  Confusion  TP={c['TP']}  FN={c['FN']}  TN={c['TN']}  FP={c['FP']}")

    lat = results["latency_ms"]
    print(f"\n  Latency    p50={lat['p50']}ms  p95={lat['p95']}ms  p99={lat['p99']}ms")

    cs = results["confidence_stats"]
    print(f"  Confidence mean={cs['mean']:.3f}  std={cs['std']:.3f}  "
          f"low_conf={cs['low_conf_pct']}%")

    print(f"\n  Throughput : {results['throughput_msg_per_s']} msg/s")
    print(f"  Wall time  : {results['total_wall_time_s']}s for {results['n']} samples")

    print("\n  Sub-type distribution (Layer 3 predictions on this split):")
    for label, count in results["sub_type_distribution"].items():
        bar = "█" * int(count / results["n"] * 40)
        pct = count / results["n"] * 100
        print(f"    {label:<35} {count:>5}  ({pct:5.1f}%)  {bar}")

    print("\n  Sample reasons:")
    for s in results["reason_samples"][:5]:
        print(f"    [{s['true']} → {s['predicted']} | conf={s['confidence']:.2f}]")
        print(f"      {s['reason']}")
        print()

    print("=" * 64 + "\n")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        logger.info("Results written to %s", out)


if __name__ == "__main__":
    main()