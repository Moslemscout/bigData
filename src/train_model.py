"""
02 - JALUR BATCH: Melatih model klasifikasi fraud dari data historis.

Model dilatih SEKALI dari data batch, lalu artefaknya dipakai berulang kali
oleh consumer streaming untuk scoring real-time.

Mode:
  * scikit-learn tersedia -> RandomForest (class_weight balanced) -> .joblib
  * tidak tersedia        -> Logistic Regression NumPy murni      -> .json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CLEAN_CSV, MODEL_DIR, MODEL_PATH
from src.common import to_feature_matrix, FEATURE_COLUMNS, TARGET

JSON_MODEL_PATH = MODEL_DIR / "fraud_model.json"


# --------------------------------------------------------------------------
# Metrik (ditulis manual agar tetap jalan tanpa sklearn)
# --------------------------------------------------------------------------
def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # ROC-AUC via peringkat (setara statistik Mann-Whitney U)
    order = np.argsort(y_prob)
    ranks = np.empty(len(y_prob), dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1)
    n_pos, n_neg = int(y_true.sum()), int((1 - y_true).sum())
    auc = ((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
           if n_pos and n_neg else float("nan"))

    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": precision, "recall": recall, "f1": f1, "roc_auc": auc}


def print_metrics(name: str, m: dict) -> None:
    print(f"\n--- EVALUASI: {name} ---")
    print(f"  Confusion: TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}   <- paling penting: fraud yang lolos")
    print(f"  F1-score  : {m['f1']:.4f}")
    print(f"  ROC-AUC   : {m['roc_auc']:.4f}")
    print("  (Accuracy sengaja TIDAK dipakai: data 99.8% non-fraud, "
          "tebak 'bukan fraud' saja sudah 99.8%)")


# --------------------------------------------------------------------------
# Fallback: Logistic Regression NumPy dengan class weighting
# --------------------------------------------------------------------------
def train_numpy_logreg(X, y, epochs=300, lr=0.1):
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma
    Xs = np.hstack([Xs, np.ones((len(Xs), 1))])           # bias

    # Bobot kelas: fraud jarang -> diberi bobot besar
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    w = np.where(y == 1, n_neg / max(n_pos, 1), 1.0)
    w = w / w.mean()

    theta = np.zeros(Xs.shape[1])
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-np.clip(Xs @ theta, -30, 30)))
        theta -= lr * (Xs.T @ (w * (p - y))) / len(y)

    return {"type": "numpy_logreg", "features": FEATURE_COLUMNS,
            "mean": mu.tolist(), "std": sigma.tolist(), "theta": theta.tolist()}


def main() -> None:
    if not CLEAN_CSV.exists():
        raise SystemExit("[X] Jalankan dulu: python src/prepare_batch.py")

    df = pd.read_csv(CLEAN_CSV)
    X = to_feature_matrix(df).values
    y = df[TARGET].astype(int).values
    print(f"[1/3] Dataset: {X.shape[0]:,} baris x {X.shape[1]} fitur | "
          f"fraud={y.sum():,} ({y.mean():.3%})")

    # Split kronologis (bukan acak): latih masa lalu, uji masa depan.
    # Ini meniru kondisi nyata sistem streaming.
    split = int(len(df) * 0.8)
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    print(f"[2/3] Split kronologis 80/20 -> train={len(Xtr):,}, test={len(Xte):,}")

    try:
        from sklearn.ensemble import RandomForestClassifier
        import joblib

        model = RandomForestClassifier(
            n_estimators=200, max_depth=12,
            class_weight="balanced_subsample",
            n_jobs=-1, random_state=42,
        )
        model.fit(Xtr, ytr)
        prob = model.predict_proba(Xte)[:, 1]
        print_metrics("RandomForest (scikit-learn)", evaluate(yte, prob))

        joblib.dump({"model": model, "features": FEATURE_COLUMNS}, MODEL_PATH)
        print(f"\n[3/3] Model disimpan: {MODEL_PATH}")

        imp = sorted(zip(FEATURE_COLUMNS, model.feature_importances_),
                     key=lambda t: -t[1])[:8]
        print("\n  Fitur paling berpengaruh:")
        for f, v in imp:
            print(f"    {f:<22} {v:.4f}")

    except ImportError:
        print("\n[!] scikit-learn tidak terpasang -> memakai Logistic Regression NumPy.")
        print("    (untuk hasil terbaik: pip install scikit-learn)")
        art = train_numpy_logreg(Xtr, ytr.astype(float))

        mu = np.array(art["mean"]); sd = np.array(art["std"])
        th = np.array(art["theta"])
        Xs = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))])
        prob = 1 / (1 + np.exp(-np.clip(Xs @ th, -30, 30)))
        print_metrics("Logistic Regression (NumPy)", evaluate(yte, prob))

        JSON_MODEL_PATH.write_text(json.dumps(art))
        print(f"\n[3/3] Model disimpan: {JSON_MODEL_PATH}")


if __name__ == "__main__":
    main()
