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
  try_prompt.py         # coba prompt ke model hasil training (interaktif / sekali jalan)
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

### Anti-overfitting yang sudah dibangun ke generator

Dataset v1 (15.426 baris) bikin model hasil fine-tune jadi **lebih buruk dari model
aslinya** untuk obrolan biasa. Tiga akar masalahnya sudah diperbaiki di generator:

| Masalah v1 | Angka v1 | Perbaikan |
|---|---|---|
| Instruksi pendek nyaris tidak ada | 0,01% (2 dari 15.426) | Domain chitchat: `sapaan`, `acknowledgment`, `identitas`, `basa_basi` |
| Panjang jawaban seragam | median 36 kata, 92% di rentang 21–60 | Kuota panjang eksplisit per batch (25% pendek / 30% sedang / 45% panjang) |
| System prompt cuma 2 teks identik | dihafal & dimuntahkan sebagai jawaban | 11 parafrase per gaya + 20% baris **tanpa** system prompt |

Terukur setelah perbaikan (sampel 60 baris): sebaran panjang jawaban jadi 38% / 52% / 10%
(dari 4% / 92% / 4%), dan system prompt unik naik dari 2 ke 19.

Generate data chitchat terpisah (target ±15% dari total dataset):

```bash
python scripts/generate_dataset.py --total 3000 --batch-size 25 --workers 8 --rpm 0 \
    --domains sapaan,acknowledgment,identitas,basa_basi --out data/generated.jsonl
```

> Kuota panjang di atas sengaja berat sebelah ke "panjang" (45%) sebagai **kompensasi bias
> teacher**, bukan target sebenarnya: waktu diminta 30% jawaban panjang, Grok cuma memenuhi
> 10%. Kalau ganti model teacher, ukur ulang sebarannya sebelum percaya angka ini.

## 2. Filter & bersihkan dataset

```bash
python scripts/filter_dataset.py \
    --in data/generated.jsonl \
    --out data/generated_clean.jsonl \
    --rejected data/generated_rejected.jsonl
```

Kriteria yang dibuang (tercatat di `reject_reason` pada file rejected):
`empty_response`, `too_short`, `too_long`, `echoes_instruction`, `repetition_loop`,
`mixed_language`, `refusal_or_meta`, `duplicate`, `wrong_attribution`.

Pengecualian yang penting:
- **domain `coding`** dikecualikan dari `mixed_language` — kode wajar mengandung kata Inggris.
- **domain chitchat** (`sapaan`, `acknowledgment`, `identitas`, `basa_basi`) pakai ambang
  panjang sendiri (1–40 kata), karena jawabannya memang harus pendek.
- **`wrong_attribution`** membuang jawaban yang mengaku dibuat vendor tertentu ("Saya dibuat
  oleh OpenAI") — teacher kadang menjawab begitu, dan kalau lolos, klaim keliru itu
  ikut ke-hardcode ke model hasil fine-tune. Pengecekan hanya aktif bila jawaban memakai
  kata "saya"/"aku", supaya kalimat faktual seperti "TensorFlow dikembangkan Google" aman.

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
python3 -m venv .venv && source .venv/bin/activate   # Ubuntu memblokir pip ke system Python (PEP 668)
pip install -r requirements-train.txt

# multi-GPU (pakai accelerate, JANGAN `python` langsung — DataParallel bawaan
# Trainer bikin error "tensors on different devices")
accelerate launch --multi_gpu --num_processes 4 scripts/train_sft.py \
    --train data/mlx_dataset/train.jsonl \
    --valid data/mlx_dataset/valid.jsonl \
    --epochs 2 --batch-size 64 --grad-accum 1 --lr 1e-5 --precision fp16

# satu GPU
CUDA_VISIBLE_DEVICES=0 python scripts/train_sft.py \
    --train data/mlx_dataset/train.jsonl --valid data/mlx_dataset/valid.jsonl
```

Setelan default sudah disetel anti-overfitting berdasarkan hasil run sebelumnya:
`lr 1e-5` (LR 3e-5–7e-5 terbukti meng-overwrite bobot asli), `epochs 2`,
`weight_decay 0.01`, `warmup_steps 10`, eval+save tiap 20 step,
`load_best_model_at_end` (ambil checkpoint dengan `eval_loss` terendah, bukan step
terakhir), dan `EarlyStoppingCallback(patience=3)`.

Catatan GPU:
- **`--precision fp16` wajib di T4/V100** (Turing tidak punya native bf16; memaksa bf16
  jatuh ke emulasi software — terukur 2,5× lebih lambat). Pakai `bf16` di Ampere ke atas.
- Model di-load fp32 lalu Trainer yang cast ke fp16 via AMP. Jangan load model langsung
  fp16 — itu memicu `ValueError: Attempting to unscale FP16 gradients`.
- `build_sft_config()` otomatis membuang parameter yang tidak didukung versi `trl`
  terpasang dan mencetak `[peringatan]`. **Baca peringatan itu** — kalau
  `weight_decay`/`warmup_steps` ikut terbuang, setelan itu tidak aktif.
- Checkpoint 0,5B ±2 GB per simpan. Dengan eval tiap 20 step, disk cepat penuh
  (run sebelumnya crash `basic_ios::clear: iostream error` gara-gara ini). Di AWS DLAMI,
  arahkan ke NVMe: `ln -s /opt/dlami/nvme/checkpoints ~/distil/checkpoints`.

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

Coba prompt ke model hasil training:

```bash
python scripts/try_prompt.py --model checkpoints/qwen05b-sft-v1              # interaktif
python scripts/try_prompt.py --model checkpoints/qwen05b-sft-v1 --prompt "Apa kabar?"
```

**Regression suite wajib** — ini kasus yang bikin run v1 gagal. Bandingkan dengan model
asli (`--model Qwen/Qwen2.5-0.5B-Instruct`); hasil fine-tune **tidak boleh lebih buruk**:

| Prompt | Kriteria lulus |
|---|---|
| `hi`, `halo`, `ok`, `siap`, `makasih` | dijawab pendek & nyambung, bukan paragraf acak |
| `Apa kabar?` | jawaban basa-basi wajar |
| `Kamu siapa?`, `Nama kamu apa?` | **tidak** menyalin system prompt, **tidak** mengarang biodata |
| `Kamu buatan siapa?` | **tidak** mengaku dibuat OpenAI/Google/dll |
| `Jelaskan cara kerja fotosintesis.` | benar secara isi (model asli halusinasi di sini) |

Selain itu:
- **Angka**: gap `eval_loss` vs `train_loss` harus rapat. Run v1 timpang (0,939 vs 0,617)
  dan itu tanda overfitting — token accuracy 0,85 waktu itu menyesatkan.
- **Manual**: baca sendiri jawaban model untuk prompt di `held_out_test.jsonl` (300 baris,
  belum pernah dipakai training). Metrik otomatis gampang menipu untuk model kecil di
  bahasa low-resource.
- **Otomatis**: IndoMMLU / IndoNLU untuk benchmark umum.
