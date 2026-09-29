"""
Pemuat model untuk scoring real-time.

Dipakai oleh consumer streaming. Memilih artefak terbaik yang tersedia:
  1. models/fraud_model.joblib -> model scikit-learn
  2. models/fraud_model.json   -> Logistic Regression NumPy
  3. rule-based                -> baseline heuristik (tanpa training)

Adanya opsi 3 membuat pipeline streaming bisa didemokan walau model
belum sempat dilatih.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import MODEL_PATH, MODEL_DIR
from src.common import add_features, to_feature_matrix

JSON_MODEL_PATH = MODEL_DIR / "fraud_model.json"


class BaseScorer:
    name = "base"

    def score_df(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def score_one(self, event: dict) -> float:
        """Skor satu event transaksi (dict) -> probabilitas fraud."""
        df = add_features(pd.DataFrame([event]))
        return float(self.score_df(df)[0])


class SklearnScorer(BaseScorer):
    name = "scikit-learn"

    def __init__(self, path: Path):
        import joblib
        bundle = joblib.load(path)
        self.model = bundle["model"]

    def score_df(self, df):
        return self.model.predict_proba(to_feature_matrix(df))[:, 1]


class NumpyLogRegScorer(BaseScorer):
    name = "numpy-logreg"

    def __init__(self, path: Path):
        art = json.loads(Path(path).read_text())
        self.mu = np.array(art["mean"])
        self.sd = np.array(art["std"])
        self.theta = np.array(art["theta"])

    def score_df(self, df):
        X = to_feature_matrix(df).values
        Xs = np.hstack([(X - self.mu) / self.sd, np.ones((len(X), 1))])
        return 1 / (1 + np.exp(-np.clip(Xs @ self.theta, -30, 30)))


class RuleScorer(BaseScorer):
    """Baseline domain-knowledge, tanpa model terlatih."""
    name = "rule-based"

    def score_df(self, df):
        df = df.copy()
        s = np.zeros(len(df))
        # Fraud PaySim hanya pada TRANSFER / CASH_OUT
        s += 0.35 * df["type"].isin(["TRANSFER", "CASH_OUT"]).to_numpy()
        # Saldo pengirim dikuras habis
        s += 0.30 * (df["amountRatioOrig"] > 0.95).to_numpy()
        # Saldo tujuan tidak ter-update (inkonsistensi)
        s += 0.25 * (df["errorBalanceDest"].abs() > 1.0).to_numpy()
        # Nominal besar
        s += 0.10 * (df["amount"] > 200_000).to_numpy()
        return np.clip(s, 0, 1)


def load_scorer() -> BaseScorer:
    if MODEL_PATH.exists():
        try:
            sc = SklearnScorer(MODEL_PATH)
            print(f"[scorer] Memakai model scikit-learn: {MODEL_PATH.name}")
            return sc
        except ImportError:
            print("[scorer] joblib/sklearn tidak ada, lanjut ke artefak berikutnya.")
    if JSON_MODEL_PATH.exists():
        print(f"[scorer] Memakai model NumPy: {JSON_MODEL_PATH.name}")
        return NumpyLogRegScorer(JSON_MODEL_PATH)
    print("[scorer] Model belum dilatih -> memakai baseline rule-based.")
    return RuleScorer()
