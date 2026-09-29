"""
01 - JALUR BATCH: Data Cleaning + Feature Engineering + EDA ringkas.

Input : data/paysim_raw.csv  (PaySim asli dari Kaggle, atau data contoh)
Output: data/paysim_clean.csv / .parquet  (siap untuk training)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RAW_CSV, CLEAN_CSV, CLEAN_PARQUET, OUTPUT_DIR
from src.common import add_features, FEATURE_COLUMNS, TARGET


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"[X] File tidak ditemukan: {path}\n"
            f"    -> Unduh PaySim dari Kaggle lalu simpan sebagai {path.name}, atau\n"
            f"    -> jalankan: python src/make_sample_data.py"
        )
    # chunksize opsional untuk file 6 juta baris di laptop RAM kecil
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Tahap Data Cleaning sesuai alur project UTS."""
    n0 = len(df)
    report = []

    # 1. Hapus duplikat
    df = df.drop_duplicates()
    report.append(f"duplikat dihapus      : {n0 - len(df):,}")

    # 2. Tangani missing value pada kolom saldo -> 0 (artinya rekening baru/kosong)
    saldo_cols = ["oldbalanceOrg", "newbalanceOrig",
                  "oldbalanceDest", "newbalanceDest", "amount"]
    n_missing = int(df[saldo_cols].isna().sum().sum())
    df[saldo_cols] = df[saldo_cols].fillna(0.0)
    report.append(f"missing value diisi 0 : {n_missing:,}")

    # 3. Buang baris tidak valid (amount negatif / nol)
    before = len(df)
    df = df[df["amount"] > 0]
    report.append(f"amount <= 0 dibuang   : {before - len(df):,}")

    # 4. Normalisasi kolom kategorikal
    df["type"] = df["type"].astype(str).str.upper().str.strip()

    print("--- DATA CLEANING ---")
    for r in report:
        print("  " + r)
    print(f"  baris: {n0:,} -> {len(df):,}")
    return df.reset_index(drop=True)


def eda(df: pd.DataFrame) -> None:
    """EDA ringkas + simpan ringkasan ke output/ untuk dilampirkan di laporan."""
    print("\n--- EDA RINGKAS ---")
    print(f"  Dimensi akhir : {df.shape[0]:,} baris x {df.shape[1]} kolom")
    print(f"  Fraud         : {int(df[TARGET].sum()):,} "
          f"({df[TARGET].mean():.4%})  <- sangat imbalanced")

    by_type = (df.groupby("type")
                 .agg(jumlah=(TARGET, "size"),
                      fraud=(TARGET, "sum"),
                      rata_amount=("amount", "mean"))
                 .assign(rasio_fraud=lambda x: x.fraud / x.jumlah)
                 .sort_values("fraud", ascending=False))
    print("\n  Fraud per tipe transaksi:")
    print(by_type.to_string(float_format=lambda v: f"{v:,.4f}"))

    by_type.to_csv(OUTPUT_DIR / "eda_fraud_per_type.csv")
    df[FEATURE_COLUMNS].describe().T.to_csv(OUTPUT_DIR / "eda_describe.csv")
    print(f"\n  [OK] Ringkasan EDA disimpan di {OUTPUT_DIR}/")


def main() -> None:
    df = load_raw()
    print(f"[1/3] Data mentah : {df.shape[0]:,} baris x {df.shape[1]} kolom")

    df = clean(df)

    df = add_features(df)
    print(f"\n[2/3] Feature engineering -> {df.shape[1]} kolom "
          f"(memenuhi syarat >= 12 kolom)")

    eda(df)

    df.to_csv(CLEAN_CSV, index=False)
    try:
        df.to_parquet(CLEAN_PARQUET, index=False)
        extra = f" & {CLEAN_PARQUET.name}"
    except Exception:
        extra = "  (parquet dilewati: butuh pyarrow)"
    print(f"\n[3/3] Data bersih disimpan: {CLEAN_CSV.name}{extra}")


if __name__ == "__main__":
    main()
