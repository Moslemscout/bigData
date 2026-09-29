"""
Jalankan SELURUH pipeline (batch + streaming) dengan satu perintah:

    python run_demo.py

Tidak butuh Kafka maupun scikit-learn. Cocok untuk demo di kelas.
"""
import subprocess
import sys

STEPS = [
    ("0/5  Membuat data contoh mirip PaySim",
     [sys.executable, "src/make_sample_data.py"]),
    ("1/5  BATCH: cleaning + feature engineering + EDA",
     [sys.executable, "src/prepare_batch.py"]),
    ("2/5  BATCH: melatih model deteksi fraud",
     [sys.executable, "src/train_model.py"]),
    ("3/5  STREAMING: membangkitkan 500 transaksi baru",
     [sys.executable, "src/stream_producer.py", "--sink", "file",
      "--limit", "500", "--eps", "0", "--fraud-rate", "0.05"]),
    ("4/5  STREAMING: scoring real-time memakai model batch",
     [sys.executable, "src/stream_consumer.py", "--source", "file"]),
    ("5/5  SERVING: menggabungkan data batch + streaming jadi satu tabel",
     [sys.executable, "src/combine.py"]),
]


def main() -> None:
    for title, cmd in STEPS:
        print("\n" + "=" * 72)
        print(f"  {title}")
        print("=" * 72)
        if subprocess.run(cmd).returncode != 0:
            sys.exit(f"[X] Gagal pada langkah: {title}")
    print("\n" + "=" * 72)
    print("  SELESAI")
    print("  Hasil GABUNGAN batch+streaming: output/combined_batch_streaming.csv")
    print("  Perbandingan kedua sumber      : output/combined_summary.csv")
    print("=" * 72)


if __name__ == "__main__":
    main()
