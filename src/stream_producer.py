"""
03 - JALUR STREAMING: Producer transaksi real-time.

Membangkitkan transaksi baru berskema PaySim secara terus-menerus dan
mengirimkannya ke Kafka topic 'transactions'.

Mode:
  --sink kafka  : kirim ke Apache Kafka (butuh kafka-python + broker jalan)
  --sink file   : tulis ke output/stream.jsonl  (TANPA dependency, untuk demo)

Contoh:
  python src/stream_producer.py --sink file --limit 200 --eps 50
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (KAFKA_BOOTSTRAP, KAFKA_TOPIC_IN, FALLBACK_STREAM,
                    EVENTS_PER_SECOND, FRAUD_RATE)
from src.common import TRANSACTION_TYPES

TYPE_WEIGHTS = [0.22, 0.35, 0.01, 0.34, 0.08]


def make_event(step: int, rng: random.Random, fraud_rate: float = FRAUD_RATE) -> dict:
    """Bangkitkan satu transaksi. Skema IDENTIK dengan dataset batch PaySim."""
    ttype = rng.choices(TRANSACTION_TYPES, weights=TYPE_WEIGHTS, k=1)[0]
    is_fraud = int(rng.random() < fraud_rate and ttype in ("TRANSFER", "CASH_OUT"))

    oldbalanceOrg = round(rng.lognormvariate(9.0, 1.5), 2)
    amount = (oldbalanceOrg if is_fraud            # fraud menguras saldo
              else round(rng.lognormvariate(8.0, 1.6), 2))
    newbalanceOrig = round(max(oldbalanceOrg - amount, 0), 2)

    oldbalanceDest = round(rng.lognormvariate(8.5, 1.8), 2)
    newbalanceDest = 0.0 if is_fraud else round(oldbalanceDest + amount, 2)

    prefix = "M" if ttype == "PAYMENT" else "C"
    return {
        "event_time": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "type": ttype,
        "amount": amount,
        "nameOrig": f"C{rng.randrange(10**8, 10**9)}",
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "nameDest": f"{prefix}{rng.randrange(10**8, 10**9)}",
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "isFraud": is_fraud,          # ground truth, untuk evaluasi saja
        "isFlaggedFraud": int(is_fraud and amount > 200_000),
    }


def get_kafka_producer():
    from kafka import KafkaProducer
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sink", choices=["kafka", "file"], default="file")
    ap.add_argument("--limit", type=int, default=0, help="0 = tak terbatas")
    ap.add_argument("--eps", type=float, default=EVENTS_PER_SECOND,
                    help="event per detik")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fraud-rate", type=float, default=FRAUD_RATE,
                    help="naikkan (mis. 0.05) agar fraud cepat terlihat saat demo")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    delay = 1.0 / args.eps if args.eps > 0 else 0.0

    producer = None
    fh = None
    if args.sink == "kafka":
        producer = get_kafka_producer()
        print(f"[producer] Kafka {KAFKA_BOOTSTRAP} -> topic '{KAFKA_TOPIC_IN}'")
    else:
        FALLBACK_STREAM.parent.mkdir(exist_ok=True)
        fh = FALLBACK_STREAM.open("w", encoding="utf-8")
        print(f"[producer] Mode file -> {FALLBACK_STREAM}")

    print(f"[producer] {args.eps} event/detik | "
          f"target fraud ~{args.fraud_rate:.2%} | Ctrl+C untuk berhenti\n")

    sent = n_fraud = 0
    step = 744  # lanjut dari akhir periode data batch
    try:
        while args.limit == 0 or sent < args.limit:
            evt = make_event(step + sent // 3600, rng, args.fraud_rate)
            if producer:
                producer.send(KAFKA_TOPIC_IN, key=evt["nameOrig"], value=evt)
            else:
                fh.write(json.dumps(evt) + "\n")
                fh.flush()

            sent += 1
            n_fraud += evt["isFraud"]
            if sent % 50 == 0 or evt["isFraud"]:
                tag = "  <-- FRAUD ditanam" if evt["isFraud"] else ""
                print(f"  [{sent:>6}] {evt['type']:<9} "
                      f"amount={evt['amount']:>12,.2f}{tag}")
            if delay:
                time.sleep(delay)
    except KeyboardInterrupt:
        print("\n[producer] Dihentikan pengguna.")
    finally:
        if producer:
            producer.flush(); producer.close()
        if fh:
            fh.close()
        print(f"\n[producer] Selesai. Terkirim={sent:,} | fraud={n_fraud:,}")


if __name__ == "__main__":
    main()
