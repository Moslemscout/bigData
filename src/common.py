"""
Feature engineering bersama untuk jalur BATCH dan STREAMING.

PENTING: modul ini dipakai oleh kedua jalur supaya fitur saat training (batch)
dan saat scoring (streaming) SAMA PERSIS. Perbedaan fitur antara training dan
inference ("training-serving skew") adalah bug klasik pada sistem ML produksi.

Dataset akhir dirancang RAMPING: tepat 13 kolom (syarat tugas minimal 12).
"""
from __future__ import annotations
import pandas as pd

# Kolom asli PaySim (11 kolom)
RAW_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
# Encoding kategorikal -> numerik, dihitung di memori saja (tidak disimpan
# sebagai kolom) agar dataset tetap ramping.
TYPE_CODES = {t: i for i, t in enumerate(TRANSACTION_TYPES)}

# ---------------------------------------------------------------------------
# Dataset akhir: 13 kolom
# ---------------------------------------------------------------------------
KEEP_COLUMNS = [
    # -- identitas & waktu (2) --
    "step",              # jam ke-n simulasi
    "hour",              # turunan: jam dalam sehari (0-23)
    # -- kategorikal (2) --
    "type",              # CASH_IN / CASH_OUT / DEBIT / PAYMENT / TRANSFER
    "isMerchantDest",    # turunan: tujuan adalah merchant (0/1)
    # -- numerik inti (5) --
    "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    # -- numerik turunan (3) --
    "errorBalanceOrig",  # inkonsistensi saldo pengirim
    "errorBalanceDest",  # inkonsistensi saldo penerima
    "amountRatioOrig",   # porsi saldo yang terkuras
    # -- target (1) --
    "isFraud",
]

# Fitur yang masuk ke model (11). 'type' diganti 'typeCode' versi numerik.
FEATURE_COLUMNS = [
    "hour",
    "typeCode", "isMerchantDest",
    "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "errorBalanceOrig", "errorBalanceDest", "amountRatioOrig",
]

TARGET = "isFraud"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan fitur turunan ke skema PaySim mentah."""
    df = df.copy()

    # --- Waktu: 'step' pada PaySim = 1 jam simulasi ---
    df["hour"] = df["step"] % 24

    # --- Inkonsistensi saldo (sinyal fraud terkuat di PaySim) ---
    # Pada transaksi normal saldo harus balance; pada fraud sering tidak.
    df["errorBalanceOrig"] = (
        df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    )
    df["errorBalanceDest"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    )

    # --- Rasio: berapa porsi saldo yang dikuras dalam 1 transaksi ---
    df["amountRatioOrig"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)

    # --- Kategorikal turunan: akun merchant diawali huruf 'M' ---
    df["isMerchantDest"] = (
        df["nameDest"].astype(str).str.startswith("M").astype(int)
    )

    return df


def select_final(df: pd.DataFrame) -> pd.DataFrame:
    """Ambil 13 kolom final. Kolom ID (nameOrig/nameDest) dan isFlaggedFraud
    dibuang: ID berkardinalitas terlalu tinggi untuk dijadikan fitur, dan
    isFlaggedFraud adalah output sistem lama (berisiko kebocoran target).
    """
    return df[[c for c in KEEP_COLUMNS if c in df.columns]].copy()


def to_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Ubah menjadi matriks numerik siap model (tanpa menambah kolom di file)."""
    df = df.copy()
    if "typeCode" not in df.columns:
        df["typeCode"] = df["type"].map(TYPE_CODES).fillna(-1)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    return df[FEATURE_COLUMNS].astype(float).fillna(0.0)
