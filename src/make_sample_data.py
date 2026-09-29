"""
00 - Membuat data contoh mirip PaySim (opsional).

Dipakai supaya pipeline bisa langsung diuji TANPA mengunduh file PaySim 470MB.
Untuk laporan/UTS, ganti dengan dataset PaySim asli dari Kaggle:
  https://www.kaggle.com/datasets/ealaxi/paysim1
"""
import sys
import numpy as np
import pandas as pd

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import RAW_CSV
from src.common import TRANSACTION_TYPES

N_ROWS = 50_000
FRAUD_RATE = 0.004
SEED = 42


def make_sample(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    step = rng.integers(1, 744, size=n_rows)                       # 30 hari
    ttype = rng.choice(TRANSACTION_TYPES, size=n_rows,
                       p=[0.22, 0.35, 0.01, 0.34, 0.08])
    amount = np.round(rng.lognormal(mean=8.0, sigma=1.6, size=n_rows), 2)

    oldbalanceOrg = np.round(rng.lognormal(mean=9.0, sigma=1.5, size=n_rows), 2)
    is_fraud = (rng.random(n_rows) < FRAUD_RATE).astype(int)

    # Fraud di PaySim hanya terjadi pada TRANSFER / CASH_OUT
    is_fraud = np.where(np.isin(ttype, ["TRANSFER", "CASH_OUT"]), is_fraud, 0)

    # Fraud menguras habis saldo pengirim
    amount = np.where(is_fraud == 1, oldbalanceOrg, amount)
    newbalanceOrig = np.maximum(oldbalanceOrg - amount, 0).round(2)

    oldbalanceDest = np.round(rng.lognormal(mean=8.5, sigma=1.8, size=n_rows), 2)
    newbalanceDest = (oldbalanceDest + amount).round(2)
    # Fraud: saldo tujuan sering tidak ter-update (inkonsistensi)
    newbalanceDest = np.where(is_fraud == 1, 0.0, newbalanceDest)

    # Akun merchant diawali 'M' dan hanya jadi tujuan PAYMENT
    dest_prefix = np.where(ttype == "PAYMENT", "M", "C")
    nameDest = [f"{p}{i}" for p, i in
                zip(dest_prefix, rng.integers(10**8, 10**9, size=n_rows))]

    df = pd.DataFrame({
        "step": step,
        "type": ttype,
        "amount": amount,
        "nameOrig": [f"C{i}" for i in rng.integers(10**8, 10**9, size=n_rows)],
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "nameDest": nameDest,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "isFraud": is_fraud,
        "isFlaggedFraud": ((is_fraud == 1) & (amount > 200_000)).astype(int),
    })

    # Sisipkan sedikit duplikat & missing value agar tahap cleaning ada gunanya
    df = pd.concat([df, df.sample(50, random_state=seed)], ignore_index=True)
    df.loc[df.sample(80, random_state=seed).index, "oldbalanceDest"] = np.nan
    return df.sort_values("step").reset_index(drop=True)


if __name__ == "__main__":
    df = make_sample()
    df.to_csv(RAW_CSV, index=False)
    print(f"[OK] Data contoh dibuat: {RAW_CSV}")
    print(f"     {len(df):,} baris x {df.shape[1]} kolom | "
          f"fraud = {int(df.isFraud.sum()):,} ({df.isFraud.mean():.3%})")
