"""Konfigurasi terpusat untuk pipeline Batch + Streaming Fraud Detection."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR   = ROOT / "data"
MODEL_DIR  = ROOT / "models"
OUTPUT_DIR = ROOT / "output"

for _d in (DATA_DIR, MODEL_DIR, OUTPUT_DIR):
    _d.mkdir(exist_ok=True)

# ---------- BATCH ----------
# Letakkan file PaySim asli di data/PS_20174392719_1491204439457_log.csv
# atau jalankan src/make_sample_data.py untuk membuat data contoh.
RAW_CSV        = DATA_DIR / "paysim_raw.csv"
CLEAN_PARQUET  = DATA_DIR / "paysim_clean.parquet"
CLEAN_CSV      = DATA_DIR / "paysim_clean.csv"
MODEL_PATH     = MODEL_DIR / "fraud_model.joblib"

# ---------- STREAMING ----------
KAFKA_BOOTSTRAP = "localhost:9092"
KAFKA_TOPIC_IN  = "transactions"        # transaksi mentah masuk
KAFKA_TOPIC_OUT = "transactions-scored" # hasil scoring
# Mode fallback: kalau Kafka tidak tersedia, event ditulis/dibaca dari file JSONL
FALLBACK_STREAM = OUTPUT_DIR / "stream.jsonl"
SCORED_STREAM   = OUTPUT_DIR / "scored.jsonl"

EVENTS_PER_SECOND = 5
FRAUD_RATE        = 0.003   # ~0.3% transaksi streaming adalah fraud
ALERT_THRESHOLD   = 0.50    # skor >= ini -> ALERT
