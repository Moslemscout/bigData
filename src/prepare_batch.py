"""
01 - JALUR BATCH: Data Cleaning + Feature Engineering + EDA ringkas.

Input : data/paysim_raw.csv  (PaySim asli dari Kaggle, atau data contoh)
Output: data/paysim_clean.csv / .parquet  (siap untuk training)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RAW_CSV, CLEAN_CSV, CLEAN_PARQUET, OUTPUT_DIR
from src.common import add_features, select_final, FEATURE_COLUMNS, TARGET


CHUNKSIZE = 500_000


def _check_exists(path: Path) -> None:
    if not path.exists():
        raise SystemExit(
            f"[X] File tidak ditemukan: {path}\n"
            f"    -> Unduh PaySim dari Kaggle lalu simpan sebagai {path.name}, atau\n"
            f"    -> jalankan: python src/make_sample_data.py"
        )


def scan_size(path: Path) -> tuple[int, int]:
    """Hitung total baris & fraud tanpa memuat seluruh file ke RAM."""
    total = fraud = 0
    for chunk in pd.read_csv(path, usecols=["isFraud"], chunksize=CHUNKSIZE):
        total += len(chunk)
        fraud += int(chunk["isFraud"].sum())
    return total, fraud


def load_raw(path: Path = RAW_CSV, max_rows: int = 0) -> pd.DataFrame:
    """Muat data mentah, dengan opsi memperkecil jumlah baris (hemat RAM).

    max_rows > 0 -> sampling STRATIFIED: baris fraud dan non-fraud diambil
    dengan proporsi sama, sehingga rasio fraud dataset asli tetap terjaga
    (mis. 0,129%). Dengan begitu angka EDA tetap mewakili populasi asli.

    Ketidakseimbangan kelas TIDAK ditangani di sini, melainkan saat training
    lewat class_weight pada model (lihat src/train_model.py).
    """
    _check_exists(path)

    if max_rows <= 0:
        return pd.read_csv(path)

    total, fraud = scan_size(path)
    print(f"      File asli: {total:,} baris ({fraud:,} fraud, {fraud/total:.4%})")

    if total <= max_rows:
        print(f"      File sudah <= {max_rows:,} baris -> dipakai seluruhnya.")
        return pd.read_csv(path)

    frac = max_rows / total
    print(f"      Sampling stratified: ambil {frac:.2%} dari kedua kelas")
    print(f"      -> target ~{max_rows:,} baris, "
          f"rasio fraud tetap ~{fraud/total:.4%} "
          f"(~{round(fraud * frac):,} baris fraud)")

    parts = []
    for chunk in pd.read_csv(path, chunksize=CHUNKSIZE):
        # Sampling terpisah per kelas memakai mask boolean (bukan
        # groupby.apply) supaya kolom isFraud tidak terserap jadi index
        # pada pandas versi baru.
        fraud_rows = chunk[chunk["isFraud"] == 1]
        normal_rows = chunk[chunk["isFraud"] == 0]
        parts.append(pd.concat([
            fraud_rows.sample(frac=frac, random_state=42),
            normal_rows.sample(frac=frac, random_state=42),
        ]))
    return pd.concat(parts, ignore_index=True).sort_values("step").reset_index(drop=True)


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
    num_cols = [c for c in df.columns if c != "type"]
    df[num_cols].describe().T.to_csv(OUTPUT_DIR / "eda_describe.csv")
    print(f"\n  [OK] Ringkasan EDA disimpan di {OUTPUT_DIR}/")


MIN_ROWS_SYARAT = 200_000


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cleaning + feature engineering + EDA untuk jalur batch.")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="batasi jumlah baris (hemat RAM). 0 = semua baris. "
                         "Contoh: --max-rows 350000")
    args = ap.parse_args()

    df = load_raw(max_rows=args.max_rows)
    print(f"[1/3] Data mentah : {df.shape[0]:,} baris x {df.shape[1]} kolom")

    df = clean(df)

    df = add_features(df)
    df = select_final(df)
    print(f"\n[2/3] Feature engineering + seleksi kolom -> {df.shape[1]} kolom")
    print(f"      {list(df.columns)}")

    eda(df)

    df.to_csv(CLEAN_CSV, index=False)
    try:
        df.to_parquet(CLEAN_PARQUET, index=False)
        extra = f" & {CLEAN_PARQUET.name}"
    except Exception:
        extra = "  (parquet dilewati: butuh pyarrow)"
    print(f"\n[3/3] Data bersih disimpan: {CLEAN_CSV.name}{extra}")

    # Cek syarat tugas: minimal 200.000 baris dan 12 kolom
    ok_rows = len(df) >= MIN_ROWS_SYARAT
    ok_cols = df.shape[1] >= 12
    print("\n--- CEK SYARAT TUGAS ---")
    print(f"  Baris  : {len(df):>9,}  (min 200.000)  "
          f"{'LOLOS' if ok_rows else 'TIDAK LOLOS'}")
    print(f"  Kolom  : {df.shape[1]:>9,}  (min 12)       "
          f"{'LOLOS' if ok_cols else 'TIDAK LOLOS'}")
    if not ok_rows:
        print("\n  [!] Baris kurang dari 200.000. Solusi:")
        print("      - Pakai dataset PaySim ASLI dari Kaggle (6.362.620 baris), atau")
        print("      - Perbesar data contoh: python src/make_sample_data.py --rows 250000")


if __name__ == "__main__":
    main()
