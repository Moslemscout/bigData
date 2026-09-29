# Deteksi Fraud Transaksi Keuangan — Batch + Streaming

Project UTS mata kuliah **ACK4FBB3 | Big Data**
Prodi S1 Teknik Komputer — Fakultas Teknik Elektro — Universitas Telkom

Pipeline end-to-end yang **menggabungkan data historis (batch)** dan **data real-time (streaming)**
untuk mendeteksi transaksi penipuan (fraud).

---

## 1. Sumber Data

| Jalur | Sumber | Keterangan |
|---|---|---|
| **Batch (historis)** | [PaySim — Synthetic Financial Datasets](https://www.kaggle.com/datasets/ealaxi/paysim1) | ± 6.362.620 baris, 11 kolom, transaksi mobile money 30 hari |
| **Batch (pembanding)** | [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | 284.807 transaksi, 492 fraud (0,172%), 31 kolom |
| **Streaming (update)** | `src/stream_producer.py` → Apache Kafka | Generator transaksi baru berskema sama, dinilai real-time |

> **Kenapa streaming-nya berupa generator?** API transaksi perbankan nyata tidak tersedia untuk
> publik karena alasan kerahasiaan nasabah. Pendekatan generator/replay adalah standar akademik
> dan sesuai contoh pada slide Minggu 2 ("Python generator transaksi baru yang terus masuk").

---

## 2. Arsitektur

```
                    ┌─────────────────── JALUR BATCH ───────────────────┐
  Kaggle PaySim ──> prepare_batch.py ──> train_model.py ──> fraud_model
  (6,3 juta baris)  cleaning + EDA +     RandomForest /     (artefak)
                    feature engineering  LogReg                  │
                    (11 -> 23 kolom)                             │
                                                                 │ dipakai
                    ┌───────────────── JALUR STREAMING ──────────┼──────┐
  stream_producer ──> Kafka topic ──> stream_consumer.py <───────┘
  (transaksi baru)    'transactions'   scoring real-time
                                            │
                                            v
                                   topic 'transactions-scored'
                                   + output/scored.jsonl (ALERT)
```

**Titik temu batch & streaming:** modul `src/common.py` dipakai oleh **kedua jalur**, sehingga
fitur saat training dan saat scoring identik. Ini mencegah *training–serving skew*, bug klasik
pada sistem ML produksi.

---

## 3. Cara Menjalankan

### Cara tercepat (tanpa Kafka, tanpa scikit-learn)

```bash
pip install pandas numpy
python run_demo.py
```

Satu perintah ini menjalankan seluruh pipeline: buat data contoh → cleaning → EDA →
training → streaming → scoring real-time.

### Memakai dataset PaySim asli

1. Unduh dari [Kaggle PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1)
2. Simpan sebagai `data/paysim_raw.csv`
3. Jalankan:

```bash
pip install -r requirements.txt
python src/prepare_batch.py     # cleaning + feature engineering + EDA
python src/train_model.py       # latih model dari data historis
```

### Mode streaming penuh dengan Apache Kafka

```bash
docker compose up -d                                  # jalankan broker Kafka

# Terminal 1 — alirkan transaksi baru
python src/stream_producer.py --sink kafka --eps 20

# Terminal 2 — scoring real-time
python src/stream_consumer.py --source kafka
```

---

## 4. Struktur Project

```
fraud-bigdata/
├── config.py                  # konfigurasi terpusat (path, Kafka, threshold)
├── run_demo.py                # jalankan semua tahap sekaligus
├── docker-compose.yml         # broker Kafka single-node (KRaft)
├── requirements.txt
├── src/
│   ├── common.py              # feature engineering BERSAMA batch & streaming
│   ├── make_sample_data.py    # 00 - data contoh mirip PaySim (opsional)
│   ├── prepare_batch.py       # 01 - cleaning + feature engineering + EDA
│   ├── train_model.py         # 02 - training + evaluasi model
│   ├── stream_producer.py     # 03 - generator transaksi -> Kafka
│   ├── stream_consumer.py     # 04 - konsumsi + scoring real-time
│   └── scorer.py              # pemuat model untuk inference
├── data/    output/    models/
```

---

## 5. Feature Engineering (11 → 23 kolom)

Syarat tugas adalah minimal 12 kolom, sedangkan PaySim asli hanya 11 kolom.
Kekurangan ini ditutup pada tahap *Mempersiapkan Dataset*:

| Fitur baru | Alasan |
|---|---|
| `errorBalanceOrig`, `errorBalanceDest` | Saldo yang tidak balance = sinyal fraud terkuat |
| `amountRatioOrig` | Fraud cenderung menguras 100% saldo |
| `hour`, `day` | `step` PaySim = jam simulasi; pola waktu penting |
| `isMerchantDest` | Akun merchant berawalan `M` |
| `isLargeAmount` | Ambang internal PaySim 200.000 |
| `type_*` (5 kolom) | One-hot encoding kolom kategorikal `type` |

---

## 6. Catatan Evaluasi

Data sangat **imbalanced** (fraud < 0,2%), sehingga **accuracy menyesatkan** — menebak
"bukan fraud" untuk semua transaksi sudah memberi accuracy 99,8%.
Karena itu metrik yang dipakai adalah **Precision, Recall, F1-score, dan ROC-AUC**,
dengan penekanan pada **Recall** (fraud yang lolos jauh lebih mahal daripada false alarm).

## 7. Anggota Kelompok

| Nama | NIM |
|---|---|
| xx | xx |
| xx | xx |
