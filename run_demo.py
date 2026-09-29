"""
Jalankan SELURUH pipeline (batch + streaming) dengan satu perintah:

    python run_demo.py

Tidak butuh Kafka maupun scikit-learn. Cocok untuk demo di kelas.
"""
import subprocess
import sys

STEPS = [
    ("0/4  Membuat data contoh mirip PaySim",
     [sys.executable, "src/make_sample_data.py"]),
    ("1/4  BATCH: cleaning + feature engineering + EDA",
     [sys.executable, "src/prepare_batch.py"]),
    ("2/4  BATCH: melatih model deteksi fraud",
     [sys.executable, "src/train_model.py"]),
    ("3/4  STREAMING: membangkitkan 500 transaksi baru",
     [sys.executable, "src/stream_producer.py", "--sink", "file",
      "--limit", "500", "--eps", "0", "--fraud-rate", "0.05"]),
    ("4/4  STREAMING: scoring real-time memakai model batch",
     [sys.executable, "src/stream_consumer.py", "--source", "file"]),
]


def main() -> None:
    for title, cmd in STEPS:
        print("\n" + "=" * 72)
        print(f"  {title}")
        print("=" * 72)
        if subprocess.run(cmd).returncode != 0:
            sys.exit(f"[X] Gagal pada langkah: {title}")
    print("\n" + "=" * 72)
    print("  SELESAI - lihat folder output/ untuk hasil EDA & scoring.")
    print("=" * 72)


if __name__ == "__main__":
    main()
