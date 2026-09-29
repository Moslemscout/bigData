"""
04 - JALUR STREAMING: Consumer + scoring fraud real-time.

Inilah titik TEMU batch dan streaming:
  model dilatih dari data BATCH  ->  dipakai menilai event STREAMING.

Mode:
  --source kafka : baca dari Kafka topic 'transactions'
  --source file  : baca output/stream.jsonl (TANPA dependency, untuk demo)

Contoh:
  python src/stream_consumer.py --source file
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (KAFKA_BOOTSTRAP, KAFKA_TOPIC_IN, KAFKA_TOPIC_OUT,
                    FALLBACK_STREAM, SCORED_STREAM, ALERT_THRESHOLD)
from src.scorer import load_scorer


def iter_kafka():
    from kafka import KafkaConsumer
    consumer = KafkaConsumer(
        KAFKA_TOPIC_IN,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        auto_offset_reset="latest",
        group_id="fraud-scorer",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )
    print(f"[consumer] Kafka {KAFKA_BOOTSTRAP} <- topic '{KAFKA_TOPIC_IN}'")
    for msg in consumer:
        yield msg.value


def iter_file(follow: bool = False):
    """Baca JSONL. follow=True meniru 'tail -f' untuk simulasi real-time."""
    if not FALLBACK_STREAM.exists():
        raise SystemExit(
            f"[X] {FALLBACK_STREAM} belum ada.\n"
            f"    Jalankan dulu: python src/stream_producer.py --sink file --limit 500"
        )
    print(f"[consumer] Mode file <- {FALLBACK_STREAM}")
    with FALLBACK_STREAM.open(encoding="utf-8") as fh:
        while True:
            line = fh.readline()
            if line:
                line = line.strip()
                if line:
                    yield json.loads(line)
            elif follow:
                time.sleep(0.2)
            else:
                return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["kafka", "file"], default="file")
    ap.add_argument("--follow", action="store_true",
                    help="mode file: terus tunggu data baru (tail -f)")
    ap.add_argument("--threshold", type=float, default=ALERT_THRESHOLD)
    args = ap.parse_args()

    scorer = load_scorer()
    stream = iter_kafka() if args.source == "kafka" else iter_file(args.follow)

    producer_out = None
    if args.source == "kafka":
        from kafka import KafkaProducer
        producer_out = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

    out_fh = SCORED_STREAM.open("w", encoding="utf-8")
    print(f"[consumer] threshold ALERT = {args.threshold}\n")

    n = n_alert = tp = fp = fn = 0
    t0 = time.time()
    try:
        for evt in stream:
            prob = scorer.score_one(evt)
            alert = prob >= args.threshold

            scored = {**evt, "fraud_score": round(prob, 6),
                      "alert": int(alert), "scored_by": scorer.name}
            out_fh.write(json.dumps(scored) + "\n"); out_fh.flush()
            if producer_out:
                producer_out.send(KAFKA_TOPIC_OUT, value=scored)

            n += 1
            truth = int(evt.get("isFraud", 0))
            if alert:
                n_alert += 1
                tp += truth
                fp += (1 - truth)
                print(f"  ALERT #{n_alert:<4} score={prob:.3f} "
                      f"{evt['type']:<9} amount={evt['amount']:>12,.2f} "
                      f"{evt['nameOrig']} -> {evt['nameDest']}"
                      f"{'  [BENAR fraud]' if truth else '  [false positive]'}")
            else:
                fn += truth

            if n % 200 == 0:
                print(f"  ...{n:,} transaksi diproses, {n_alert} alert")
    except KeyboardInterrupt:
        print("\n[consumer] Dihentikan pengguna.")
    finally:
        out_fh.close()
        if producer_out:
            producer_out.flush(); producer_out.close()

        dur = max(time.time() - t0, 1e-9)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        print("\n--- RINGKASAN STREAMING ---")
        print(f"  Transaksi diproses : {n:,}  ({n/dur:,.0f} transaksi/detik)")
        print(f"  Alert dikeluarkan  : {n_alert:,}")
        print(f"  Benar fraud (TP)   : {tp}")
        print(f"  False positive (FP): {fp}")
        print(f"  Fraud lolos (FN)   : {fn}")
        print(f"  Precision={prec:.3f} | Recall={rec:.3f}")
        print(f"  Hasil disimpan: {SCORED_STREAM}")


if __name__ == "__main__":
    main()
