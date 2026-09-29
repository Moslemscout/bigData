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
                    (11 -> 13 kolom)                             │
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

Satu perintah ini menjalankan seluruh pipeline: buat data contoh (250.000 baris) → cleaning →
EDA → training → streaming → scoring real-time → gabungkan.

> ⚠️ **PENTING:** `run_demo.py` memakai **data contoh buatan** (`make_sample_data.py`), bukan
> PaySim asli. Gunanya hanya untuk menguji pipeline. Untuk laporan/UTS **wajib** memakai dataset
> PaySim asli dari Kaggle (lihat bagian berikutnya).

### Memakai dataset PaySim asli (WAJIB untuk laporan)

**Langkah unduh:**

1. Buka <https://www.kaggle.com/datasets/ealaxi/paysim1>
2. **Login** dulu (akun Kaggle gratis) — tombol Download tidak muncul kalau belum login
3. Klik tombol **Download** di kanan atas → dapat `archive.zip` (~180 MB terkompresi)
4. Ekstrak → isinya satu file: **`PS_20174392719_1491204439457_log.csv`** (493,53 MB)
5. **Ganti namanya** menjadi `paysim_raw.csv`
6. Pindahkan ke folder **`data/`** di dalam project ini

Hasil akhirnya harus seperti ini:

```
fraud-bigdata/
└── data/
    └── paysim_raw.csv        <- 493 MB, 6.362.620 baris
```

**Alternatif via Kaggle API** (kalau lebih suka terminal):

```bash
pip install kaggle          # taruh kaggle.json di ~/.kaggle/
kaggle datasets download -d ealaxi/paysim1
unzip paysim1.zip
mv PS_20174392719_1491204439457_log.csv data/paysim_raw.csv
```

**Verifikasi dulu bahwa datanya asli:**

```bash
python src/check_dataset.py
```

```
--- HASIL VERIFIKASI ---
  [v] Jumlah baris = 6.362.620
  [v] Jumlah fraud = 8.213
  [v] Nama kolom sesuai PaySim
  [v] Step maksimum = 743 (30 hari)
  ==> DATA ASLI PaySim dari Kaggle. Siap dipakai untuk laporan.
```

**Lalu jalankan pipeline** (JANGAN pakai `run_demo.py`, karena akan menimpa file aslimu):

```bash
pip install -r requirements.txt

# Semua 6,3 juta baris (butuh RAM ~4GB)
python src/prepare_batch.py

# ATAU batasi jumlah baris (disarankan: 350.000 -> ringan & lolos syarat)
python src/prepare_batch.py --max-rows 350000

python src/train_model.py
python src/stream_producer.py --sink file --limit 500 --eps 0
python src/stream_consumer.py --source file
python src/combine.py
```

### Kenapa PaySim 6,3 juta baris, dan cara memperkecilnya

PaySim mensimulasikan **seluruh layanan mobile money selama 30 hari** (743 jam), sekitar
8.600 transaksi per jam — jadi 6,3 juta baris itu wajar. Syarat tugas hanya minimal 200.000,
sehingga kita boleh mengambil sebagiannya saja:

```bash
python src/prepare_batch.py --max-rows 350000
```

Sampling yang dipakai adalah **stratified**: baris fraud dan non-fraud diambil dengan
proporsi sama, sehingga **rasio fraud dataset asli tetap terjaga**.

```
File asli: 6,362,620 baris (8,213 fraud, 0.1291%)
Sampling stratified: ambil 5.50% dari kedua kelas
-> target ~350,000 baris, rasio fraud tetap ~0.1291% (~452 baris fraud)
```

File dibaca **per 500.000 baris (chunk)**, jadi RAM tetap aman meski file aslinya 493 MB.

**Ketidakseimbangan kelas tidak ditangani di tahap ini.** Rasio fraud sengaja dibiarkan apa
adanya supaya angka EDA jujur mewakili populasi asli. Penanganannya dilakukan saat training
lewat `class_weight="balanced_subsample"` pada model (lihat `src/train_model.py`).

> **Wajib ditulis di laporan:** sebutkan bahwa data disampling dari 6.362.620 baris menjadi
> 350.000 baris beserta alasannya (keterbatasan RAM), dan bahwa rasio fraud tetap terjaga.

### Cek syarat tugas otomatis

`prepare_batch.py` otomatis memverifikasi syarat minimal **200.000 baris dan 12 kolom**:

```
--- CEK SYARAT TUGAS ---
  Baris  :   500,000  (min 200.000)  LOLOS
  Kolom  :        23  (min 12)       LOLOS
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
│   ├── combine.py             # 05 - GABUNGKAN batch + streaming (serving layer)
│   ├── check_dataset.py       # verifikasi dataset asli vs data contoh
│   └── scorer.py              # pemuat model untuk inference
├── data/    output/    models/
```

---

## 4b. Di Mana Gabungan Batch + Streaming Terlihat?

Penggabungan terjadi di **tiga tingkat**:

| # | Tingkat | Buktinya di mana |
|---|---|---|
| 1 | **Skema** — streaming memakai kolom yang sama dengan batch | `src/common.py` dipakai kedua jalur |
| 2 | **Model** — model hasil training batch dipakai menilai event streaming | kolom `fraud_score` di `output/scored.jsonl` |
| 3 | **Data** — kedua sumber disatukan dalam satu tabel | `output/combined_batch_streaming.csv` (kolom `source`) |

Jalankan:

```bash
python src/combine.py
```

Hasilnya:

```
[1/3] Batch     :  50,000 baris  (data historis, untuk melatih model)
      Streaming :     500 baris  (transaksi baru, sudah di-scoring)
[2/3] GABUNGAN  :  50,500 baris x 21 kolom

--- PERBANDINGAN BATCH vs STREAMING ---
source      jumlah  fraud  rasio_fraud  rata_amount  hari_min  hari_max
batch        50000     96       0.0019   10,935.14          0        30
streaming      500     11       0.0220   10,425.73         31        31

--- CEK DATA DRIFT ---
  Selisih komposisi tipe terbesar : 2.04 poin persen
  Status: STABIL (model masih layak dipakai)
```

Perhatikan kolom `hari_min`/`hari_max`: data batch berhenti di **hari 30**, lalu streaming
melanjutkan di **hari 31** — jadi keduanya menyambung sebagai satu rentang waktu, bukan dua
dataset terpisah.

---

## 5. Feature Engineering (11 → 13 kolom)

Syarat tugas minimal 12 kolom, sedangkan PaySim asli hanya 11 kolom. Dataset akhir dirancang
**ramping: tepat 13 kolom**, bukan sebanyak-banyaknya — setiap kolom punya alasan.

**Kolom dibuang (3):**

| Kolom | Alasan dibuang |
|---|---|
| `nameOrig`, `nameDest` | ID akun, kardinalitas terlalu tinggi (jutaan nilai unik) untuk dijadikan fitur |
| `isFlaggedFraud` | Output sistem penandaan lama → berisiko **kebocoran target** (target leakage) |

**Kolom ditambahkan (5):**

| Fitur baru | Alasan |
|---|---|
| `errorBalanceOrig`, `errorBalanceDest` | Saldo yang tidak balance = sinyal fraud terkuat |
| `amountRatioOrig` | Fraud cenderung menguras 100% saldo pengirim |
| `hour` | `step` PaySim = jam simulasi; pola waktu transaksi penting |
| `isMerchantDest` | Akun merchant berawalan `M` (fitur kategorikal) |

**Dataset akhir (13 kolom):**

```
step, hour, type, isMerchantDest, amount,
oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest,
errorBalanceOrig, errorBalanceDest, amountRatioOrig, isFraud
```

Sudah memenuhi syarat kombinasi tipe data:
**kategorikal** (`type`, `isMerchantDest`) + **numerik** (`amount`, saldo, rasio) + **target** (`isFraud`).

> Kolom `type` disimpan sebagai teks agar mudah dibaca saat EDA. Saat training, kolom ini
> di-encode menjadi angka (`typeCode`) **di memori saja**, jadi tidak menambah kolom di file.

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
