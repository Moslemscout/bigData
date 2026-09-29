"""
Verifikasi apakah data/paysim_raw.csv benar-benar PaySim ASLI dari Kaggle.

Jalankan setelah mengunduh dataset:
    python src/check_dataset.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RAW_CSV
from src.common import RAW_COLUMNS

# Angka resmi PaySim (ealaxi/paysim1)
ASLI_BARIS = 6_362_620
ASLI_FRAUD = 8_213
ASLI_STEP_MAX = 743
MIN_SYARAT = 200_000
CHUNK = 500_000


def main() -> None:
    if not RAW_CSV.exists():
        raise SystemExit(
            f"[X] Tidak ada file: {RAW_CSV}\n\n"
            "    Langkah:\n"
            "    1. Buka https://www.kaggle.com/datasets/ealaxi/paysim1\n"
            "    2. Login (akun gratis) lalu klik Download\n"
            "    3. Ekstrak archive.zip\n"
            "    4. Ganti nama PS_20174392719_1491204439457_log.csv "
            "menjadi paysim_raw.csv\n"
            f"    5. Pindahkan ke folder: {RAW_CSV.parent}"
        )

    size_mb = RAW_CSV.stat().st_size / 1024 / 1024
    print(f"File   : {RAW_CSV.name}")
    print(f"Ukuran : {size_mb:,.1f} MB   (PaySim asli ~493,5 MB)")

    # Baca bertahap agar hemat RAM
    baris = fraud = 0
    step_max = 0
    kolom = None
    for chunk in pd.read_csv(RAW_CSV, chunksize=CHUNK):
        if kolom is None:
            kolom = list(chunk.columns)
        baris += len(chunk)
        fraud += int(chunk["isFraud"].sum())
        step_max = max(step_max, int(chunk["step"].max()))

    print(f"Baris  : {baris:,}")
    print(f"Kolom  : {len(kolom)} -> {kolom}")
    print(f"Fraud  : {fraud:,} ({fraud/baris:.4%})")
    print(f"Step   : maks {step_max}")

    print("\n--- HASIL VERIFIKASI ---")
    cek = [
        ("Jumlah baris = 6.362.620", baris == ASLI_BARIS),
        ("Jumlah fraud = 8.213", fraud == ASLI_FRAUD),
        ("Nama kolom sesuai PaySim", kolom == RAW_COLUMNS),
        ("Step maksimum = 743 (30 hari)", step_max == ASLI_STEP_MAX),
    ]
    for label, ok in cek:
        print(f"  [{'v' if ok else 'x'}] {label}")

    asli = all(ok for _, ok in cek)
    print()
    if asli:
        print("  ==> DATA ASLI PaySim dari Kaggle. Siap dipakai untuk laporan.")
    else:
        print("  ==> BUKAN PaySim asli (kemungkinan data contoh buatan).")
        print("      Untuk laporan/UTS, unduh dataset asli dari:")
        print("      https://www.kaggle.com/datasets/ealaxi/paysim1")

    print(f"\n  Syarat minimal 200.000 baris: "
          f"{'LOLOS' if baris >= MIN_SYARAT else 'TIDAK LOLOS'} ({baris:,} baris)")


if __name__ == "__main__":
    main()
