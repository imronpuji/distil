# Dataset Generation Pipeline — Distillation Gemma 4 E4B → Qwen

Script untuk langkah 1 & 2 dari rencana proyek: generate dataset instruction-response
Bahasa Indonesia dari teacher (Gemma 4 E4B via Ollama), lalu filter/bersihkan sebelum SFT.

**Catatan penting:** semua script ini ditulis & diuji secara sintaksis/logika di
environment cloud yang TIDAK punya akses ke Ollama kamu di EC2 g5g. Sebelum
dijalankan untuk produksi, jalankan dulu di instance EC2-mu (di mana
`ollama run gemma4:e4b` sudah aktif) dengan `--limit` kecil untuk verifikasi.

## Struktur

```
scripts/
  seed_bank.py         # generate seed prompts (manual + combinatorial)
  generate_dataset.py  # panggil Ollama API, hasilkan ChatML JSONL
  filter_dataset.py    # filter/bersihkan dataset hasil generate
data/
  seed_prompts.jsonl   # seed yang sudah digenerate (contoh: 1540 baris)
```

## 1. Generate seed prompts

```bash
python scripts/seed_bank.py --n 3000 --out data/seed_prompts.jsonl
```

Saat ini kombinasi domain x topik x template x gaya menghasilkan ~1540 seed unik.
Untuk menembus ribuan lebih:
- Tambah topik baru di dict `DOMAINS` (banyak-banyakin per domain).
- Tambah variasi kalimat di `INSTRUCTION_TEMPLATES`.
- Tambah contoh berkualitas di `SEED_MANUAL` (terutama untuk roleplay, curhat,
  multi-turn style yang sulit dibuat lewat template).

## 2. Generate dataset dari teacher (Gemma 4 E4B via Ollama)

Di instance EC2 (pastikan `ollama serve` jalan & `ollama pull gemma4:e4b` sudah selesai):

```bash
# tes dulu dengan jumlah kecil
python scripts/generate_dataset.py \
    --seeds data/seed_prompts.jsonl \
    --out data/generated.jsonl \
    --model gemma4:e4b \
    --host http://localhost:11434 \
    --workers 2 \
    --temperature 0.8 \
    --limit 20

# kalau hasilnya bagus, jalankan penuh (hapus --limit)
python scripts/generate_dataset.py \
    --seeds data/seed_prompts.jsonl \
    --out data/generated.jsonl \
    --workers 2
```

Poin penting:
- **Resume otomatis**: kalau script berhenti di tengah jalan (mis. crash/timeout),
  tinggal jalankan ulang command yang sama — seed yang sudah punya hasil di
  `data/generated.jsonl` akan dilewati.
- **`--workers`**: mulai dari 1–2 dulu. Ollama di 2x T4G punya memori terbatas;
  paralelisme tinggi bisa bikin OOM atau justru memperlambat (request antre di
  belakang layar). Naikkan bertahap sambil pantau `nvidia-smi`.
- **`--num-samples N`**: kalau mau generate N jawaban per prompt dengan temperature
  sama (untuk nanti dipilih yang terbaik / self-consistency filtering).
- Error individual (timeout, response gagal parse) dicatat di
  `data/generated.jsonl.errors.jsonl` dan otomatis di-retry saat re-run karena
  tidak dianggap "done".

## 3. Filter & bersihkan dataset

```bash
python scripts/filter_dataset.py \
    --in data/generated.jsonl \
    --out data/generated_clean.jsonl \
    --rejected data/generated_rejected.jsonl \
    --min-words 5 --max-words 400
```

Kriteria yang dibuang (tercatat di `reject_reason` pada file rejected):
- `empty_response`, `too_short`, `too_long`
- `echoes_instruction` — jawaban cuma mengulang instruksi
- `repetition_loop` — kata/frasa berulang berturut-turut (gejala looping generation)
- `mixed_language` — rasio kata bahasa Inggris umum terlalu tinggi (heuristik, bukan
  detector bahasa formal — kalibrasi `--max-mixed-ratio` sesuai hasil cek manual)
- `refusal_or_meta` — pola penolakan / "sebagai model bahasa..."
- `duplicate` — jawaban identik (case-insensitive) dengan yang sudah lolos sebelumnya

**Wajib**: cek manual 30–50 baris dari `data/generated_rejected.jsonl` untuk pastikan
filter tidak terlalu agresif (buang data bagus) atau terlalu longgar (loloskan data
jelek) sebelum dipakai untuk kalibrasi ambang.

## Langkah selanjutnya (belum dikerjakan di sesi ini)

1. Jalankan pipeline ini penuh di EC2 untuk hasilkan ~10.000–20.000 pasang bersih
   (lihat rekomendasi ukuran dataset di project brief).
2. Sisihkan held-out set (200–500 prompt) sebelum SFT untuk evaluasi manual cepat.
3. Siapkan pipeline SFT Qwen 1.5B (unsloth/axolotl) — format ChatML dari
   `generated_clean.jsonl` sudah siap pakai untuk tahap ini.
4. Evaluasi dengan IndoMMLU/IndoNLU + manual testing.
# distil
