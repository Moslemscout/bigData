"""
Feature engineering bersama untuk jalur BATCH dan STREAMING.

PENTING: modul ini dipakai oleh kedua jalur supaya fitur saat training (batch)
dan saat scoring (streaming) SAMA PERSIS. Perbedaan fitur antara training dan
inference ("training-serving skew") adalah bug klasik pada sistem ML produksi.
"""
from __future__ import annotations
import pandas as pd

# Kolom asli PaySim (11 kolom)
RAW_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Fitur numerik final yang masuk ke model
FEATURE_COLUMNS = [
    "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "errorBalanceOrig", "errorBalanceDest",
    "amountRatioOrig",
    "hour", "day",
    "isMerchantDest",
    "isLargeAmount",
] + [f"type_{t}" for t in TRANSACTION_TYPES]

TARGET = "isFraud"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan fitur turunan. Input = skema PaySim mentah, output = + fitur baru.

    Menaikkan jumlah kolom dari 11 menjadi >12 sesuai syarat tugas.
    """
    df = df.copy()

    # --- Fitur waktu: 'step' pada PaySim = 1 jam simulasi ---
    df["hour"] = df["step"] % 24
    df["day"] = df["step"] // 24

    # --- Fitur inkonsistensi saldo (sinyal fraud terkuat di PaySim) ---
    # Pada transaksi normal saldo harus balance; pada fraud sering tidak.
    df["errorBalanceOrig"] = (
        df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    )
    df["errorBalanceDest"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    )

    # --- Rasio: berapa persen saldo yang dikuras dalam 1 transaksi ---
    df["amountRatioOrig"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)

    # --- Fitur kategorikal turunan ---
    # Akun tujuan merchant diawali huruf 'M'
    df["isMerchantDest"] = (
        df["nameDest"].astype(str).str.startswith("M").astype(int)
    )
    # PaySim: sistem internal menandai transfer > 200.000
    df["isLargeAmount"] = (df["amount"] > 200_000).astype(int)

    # --- One-hot encoding kolom kategorikal 'type' ---
    # Dibuat manual (bukan pd.get_dummies) supaya kolom selalu lengkap & urut
    # sama, baik untuk 1 baris streaming maupun jutaan baris batch.
    for t in TRANSACTION_TYPES:
        df[f"type_{t}"] = (df["type"] == t).astype(int)

    return df


def to_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Ambil hanya kolom fitur, urut & lengkap, siap masuk model."""
    df = df.copy()
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    return df[FEATURE_COLUMNS].astype(float).fillna(0.0)
