# Dataset Generation & SFT Pipeline — Grok → Qwen2.5-0.5B (Bahasa Indonesia)

Pipeline buat generate dataset instruksi+jawaban Bahasa Indonesia dari nol pakai
Grok (xAI API), bersihkan, lalu full fine-tune (SFT) model kecil (Qwen2.5-0.5B-Instruct)
biar lebih fasih & relevan menjawab dalam Bahasa Indonesia.

## Struktur

```
scripts/
  generate_dataset.py  # Grok generate PASANGAN instruksi+jawaban dari nol (no seed file)
  filter_dataset.py    # filter/bersihkan dataset hasil generate
  train_sft.py          # full fine-tune SFT (jalankan di GPU NVIDIA / cloud)
  seed_bank.py          # dipakai generate_dataset.py cuma sebagai daftar domain/topik
                         # inspirasi (bukan sumber instruksi final)
data/
  generated.jsonl        # hasil mentah dari generate_dataset.py
  generated_clean.jsonl  # hasil setelah difilter
  mlx_dataset/            # train.jsonl / valid.jsonl / test.jsonl (siap SFT)
  held_out_test.jsonl     # 300 baris disisihkan buat eval manual (JANGAN dipakai training)
```

## 1. Generate dataset dari Grok (xAI)

Butuh `XAI_API_KEY` (taruh di `.env` di root project). Grok generate instruksi
**dan** jawabannya sekaligus, per domain/gaya yang dirotasi otomatis.

```bash
# gampangnya, mode interaktif (jalankan dari terminal kamu sendiri, bukan lewat agent)
python scripts/generate_dataset.py --interactive

# atau langsung
python scripts/generate_dataset.py \
    --total 15000 \
    --batch-size 25 \
    --workers 8 \
    --rpm 0 \
    --model grok-4-fast-non-reasoning \
    --out data/generated.jsonl
```

Poin penting:
- **`--model`**: default `grok-4-fast-non-reasoning` — murah & cepat ($0.20/1M input,
  $0.50/1M output di bawah 128k token), cocok buat generate volume besar. Model
  reasoning (`grok-4.6` dkk) lebih mahal & lambat, cuma worth it buat held-out
  eval set kecil yang butuh kualitas maksimal.
- **`--batch-size`**: berapa pasang instruksi+jawaban diminta sekaligus per request
  (model dipaksa balas JSON array). Naikin buat hemat jumlah request.
- **`--rpm`**: rate limit per menit, berlaku global lintas worker. `0` = tanpa batas.
- Resume otomatis & dedup: instruksi yang hash-nya udah ada di file output otomatis
  di-skip kalau muncul lagi.
- Error dicatat terpisah di `<out>.errors.jsonl`.

## 2. Filter & bersihkan dataset

```bash
python scripts/filter_dataset.py \
    --in data/generated.jsonl \
    --out data/generated_clean.jsonl \
    --rejected data/generated_rejected.jsonl \
    --min-words 5 --max-words 400
```

Kriteria yang dibuang (tercatat di `reject_reason` pada file rejected):
`empty_response`, `too_short`, `too_long`, `echoes_instruction`, `repetition_loop`,
`mixed_language` (di-skip khusus domain `coding` karena kode wajar mengandung kata
Inggris), `refusal_or_meta`, `duplicate`.

**Wajib**: cek manual beberapa puluh baris dari `data/generated_rejected.jsonl`
buat pastikan filter nggak kebablasan.

## 3. Split train/valid/held-out

```bash
python3 - <<'PY'
import json, random
random.seed(42)
rows = [json.loads(l) for l in open("data/generated_clean.jsonl")]
random.shuffle(rows)
valid, test, train = rows[:400], rows[400:700], rows[700:]
def write(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for r in items:
            f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
write("data/mlx_dataset/train.jsonl", train)
write("data/mlx_dataset/valid.jsonl", valid)
write("data/mlx_dataset/test.jsonl", test)
PY
```

`held_out_test.jsonl` / `mlx_dataset/test.jsonl` **jangan pernah dipakai buat
training** — itu yang dipakai buat eval manual di langkah 5.

## 4. Full fine-tune (SFT)

Dua opsi tergantung hardware:

**A. GPU NVIDIA (cloud: RunPod/Vast.ai/Colab) — direkomendasikan buat training penuh**

```bash
pip install -r requirements-train.txt
python scripts/train_sft.py \
    --train data/mlx_dataset/train.jsonl \
    --valid data/mlx_dataset/valid.jsonl \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --out checkpoints/qwen05b-sft-v1 \
    --epochs 3 --batch-size 16 --grad-accum 2 --lr 2e-5
```

**B. Mac Apple Silicon (mlx-lm) — buat eksperimen cepat/kecil, jangan buat run penuh**

```bash
pip install mlx-lm
mlx_lm.lora \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --train --data data/mlx_dataset --fine-tune-type full --num-layers -1 \
    --batch-size 8 --iters 2000 --learning-rate 2e-5 \
    --adapter-path checkpoints/qwen05b-sft-v1
```

Full fine-tune (bukan LoRA) direkomendasikan buat model sekecil ini — riset Cendol
(Cahyawijaya et al., 2024) nunjukin LoRA kurang efektif buat language adaptation
dibanding full fine-tune model yang lebih kecil sekalipun.

## 5. Evaluasi

- **Otomatis**: IndoMMLU / IndoNLU buat benchmark umum.
- **Manual (wajib)**: generate jawaban model buat semua prompt di
  `held_out_test.jsonl`, baca sendiri — cek relevansi jawaban, grammar, dan ada
  nggak mixed-language random. Ini yang paling nentuin training beneran berhasil,
  karena metrik otomatis gampang menipu buat model kecil di bahasa low-resource.
