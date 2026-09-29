"""
05 - SERVING LAYER: Menggabungkan data BATCH + STREAMING jadi satu tabel.

Inilah bukti nyata penggabungan kedua jalur (arsitektur Lambda):
  - Batch layer    : data historis PaySim yang sudah dibersihkan
  - Speed layer    : transaksi streaming yang sudah diberi skor fraud
  - Serving layer  : gabungan keduanya -> output/combined_batch_streaming.csv

Output:
  1. output/combined_batch_streaming.csv  <- SATU tabel berisi kedua sumber
  2. output/combined_summary.csv          <- perbandingan batch vs streaming
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CLEAN_CSV, SCORED_STREAM, OUTPUT_DIR
from src.common import add_features, select_final, KEEP_COLUMNS, TARGET

COMBINED_CSV = OUTPUT_DIR / "combined_batch_streaming.csv"
SUMMARY_CSV  = OUTPUT_DIR / "combined_summary.csv"

# Kolom yang dipakai bersama oleh kedua sumber (13 kolom dataset final)
SHARED = list(KEEP_COLUMNS)


def load_batch() -> pd.DataFrame:
    if not CLEAN_CSV.exists():
        raise SystemExit("[X] Jalankan dulu: python src/prepare_batch.py")
    df = pd.read_csv(CLEAN_CSV)
    df["source"] = "batch"
    df["event_time"] = pd.NaT          # data historis: tidak ada timestamp asli
    df["fraud_score"] = pd.NA          # batch = data latih, tidak di-scoring
    df["alert"] = pd.NA
    return df


def load_streaming() -> pd.DataFrame:
    if not SCORED_STREAM.exists():
        raise SystemExit(
            "[X] Belum ada hasil streaming.\n"
            "    Jalankan: python src/stream_producer.py --sink file --limit 500 "
            "--eps 0 --fraud-rate 0.05\n"
            "    lalu    : python src/stream_consumer.py --source file"
        )
    rows = [json.loads(l) for l in
            SCORED_STREAM.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit("[X] output/scored.jsonl kosong.")
    raw = pd.DataFrame(rows)
    df = select_final(add_features(raw))   # fitur SAMA dengan jalur batch
    # kolom meta streaming dibawa terpisah dari 13 kolom dataset
    for col in ("event_time", "fraud_score", "alert"):
        if col in raw.columns:
            df[col] = raw[col]
    df["source"] = "streaming"
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("source")
    out = pd.DataFrame({
        "jumlah_transaksi": g.size(),
        "fraud": g[TARGET].sum(),
        "rasio_fraud": g[TARGET].mean(),
        "rata_amount": g["amount"].mean(),
        "median_amount": g["amount"].median(),
        "max_amount": g["amount"].max(),
        "step_min": g["step"].min(),
        "step_max": g["step"].max(),
    })
    return out.loc[[s for s in ("batch", "streaming") if s in out.index]]


def main() -> None:
    batch = load_batch()
    stream = load_streaming()

    print(f"[1/3] Batch     : {len(batch):>7,} baris  (data historis, untuk melatih model)")
    print(f"      Streaming : {len(stream):>7,} baris  (transaksi baru, sudah di-scoring)")

    # Gabungkan: skema diseragamkan lewat kolom SHARED
    meta = ["source", "event_time", "fraud_score", "alert"]
    cols = SHARED + meta
    combined = pd.concat(
        [batch.reindex(columns=cols), stream.reindex(columns=cols)],
        ignore_index=True,
    )
    # Urutkan kronologis: data historis dulu, lalu transaksi terbaru
    combined = combined.sort_values(
        ["source", "step"], ascending=[False, True], kind="stable"
    ).sort_values("step", kind="stable").reset_index(drop=True)

    print(f"\n[2/3] GABUNGAN  : {len(combined):>7,} baris x {combined.shape[1]} kolom")
    print(f"      -> {COMBINED_CSV}")
    combined.to_csv(COMBINED_CSV, index=False)

    summary = summarize(combined)
    summary.to_csv(SUMMARY_CSV)

    print("\n--- PERBANDINGAN BATCH vs STREAMING ---")
    print(summary.to_string(float_format=lambda v: f"{v:,.4f}"))

    print("\n--- KOMPOSISI TIPE TRANSAKSI (%) ---")
    comp = (pd.crosstab(combined["type"], combined["source"], normalize="columns") * 100)
    print(comp.to_string(float_format=lambda v: f"{v:6.2f}"))

    # Cek data drift: apakah pola transaksi baru menyimpang dari data latih?
    print("\n--- CEK DATA DRIFT ---")
    if {"batch", "streaming"} <= set(comp.columns):
        drift = (comp["streaming"] - comp["batch"]).abs().max()
        print(f"  Selisih komposisi tipe terbesar : {drift:.2f} poin persen")
        print("  Status:", "STABIL (model masih layak dipakai)" if drift < 10
              else "DRIFT TERDETEKSI -> model perlu dilatih ulang")

    # Ringkasan alert dari jalur streaming saja
    sdf = combined[combined["source"] == "streaming"]
    if len(sdf):
        n_alert = int(pd.to_numeric(sdf["alert"], errors="coerce").fillna(0).sum())
        print(f"\n[3/3] Alert dari jalur streaming : {n_alert} "
              f"dari {len(sdf):,} transaksi")
        print(f"      Ringkasan disimpan: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
