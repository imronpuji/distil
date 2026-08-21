"""
train_sft.py
============
Full fine-tune (SFT) Qwen2.5-0.5B-Instruct pakai dataset instruksi+jawaban
Bahasa Indonesia. Didesain buat dijalankan di GPU NVIDIA (cloud: RunPod,
Vast.ai, Colab, dll) -- BUKAN di Mac. Untuk eksperimen cepat di Mac (Apple
Silicon), pakai mlx-lm (lihat README).

Setup di instance GPU:
    pip install -r requirements-train.txt

Cara pakai:
    python scripts/train_sft.py \\
        --train data/mlx_dataset/train.jsonl \\
        --valid data/mlx_dataset/valid.jsonl \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --out checkpoints/qwen05b-sft-v1 \\
        --epochs 3 \\
        --batch-size 16 \\
        --grad-accum 2 \\
        --lr 2e-5

Poin penting:
- Full fine-tune (semua parameter di-update), BUKAN LoRA -- lihat diskusi
  di README soal kenapa LoRA kurang efektif untuk language adaptation di
  model sekecil ini.
- Loss cuma dihitung di token jawaban (assistant), bukan di prompt
  (system+user) -- via `completion_only_loss=True` dari SFTTrainer, biar
  model belajar "cara jawab", bukan "cara nulis pertanyaan".
- Checkpoint disimpan tiap `--save-steps` step ke `--out`, otomatis resume
  kalau `--out` sudah ada checkpoint sebelumnya (lewat `--resume`).
"""

import argparse
import inspect

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def build_sft_config(**kwargs):
    """Filter kwargs ke parameter yang beneran didukung SFTConfig di versi trl
    yang keinstall -- API trl sering berubah antar versi, jadi ini biar script
    nggak gampang patah kalau ada parameter yang berubah nama/dihapus."""
    accepted = set(inspect.signature(SFTConfig.__init__).parameters)
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = {k: v for k, v in kwargs.items() if k not in accepted}
    if dropped:
        print(f"[peringatan] Parameter berikut tidak didukung SFTConfig versi ini, "
              f"di-skip: {list(dropped.keys())}")
    return SFTConfig(**filtered)


def main():
    parser = argparse.ArgumentParser(description="Full fine-tune SFT untuk model kecil Bahasa Indonesia.")
    parser.add_argument("--train", type=str, default="data/mlx_dataset/train.jsonl")
    parser.add_argument("--valid", type=str, default="data/mlx_dataset/valid.jsonl")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--out", type=str, default="checkpoints/qwen05b-sft-v1")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16,
                         help="Per-device batch size. Naikkan kalau VRAM masih longgar.")
    parser.add_argument("--grad-accum", type=int, default=2,
                         help="Gradient accumulation steps (effective batch = batch-size x grad-accum).")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--precision", type=str, choices=["bf16", "fp16", "fp32"], default="fp16",
                         help="fp16 buat GPU Turing/lama (T4, V100 -- nggak punya native bf16, "
                              "bf16 di situ jatuhnya emulasi software yang jauh lebih lambat). "
                              "bf16 buat GPU Ampere+ (A10, A100, RTX 30xx/40xx ke atas).")
    parser.add_argument("--resume", action="store_true",
                         help="Resume dari checkpoint terakhir di --out kalau ada.")
    args = parser.parse_args()

    dtype = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[args.precision]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)

    train_ds = load_dataset("json", data_files=args.train, split="train")
    valid_ds = load_dataset("json", data_files=args.valid, split="train")

    config = build_sft_config(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_seq_length,
        bf16=(args.precision == "bf16"),
        fp16=(args.precision == "fp16"),
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        completion_only_loss=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\nSelesai. Model tersimpan di: {args.out}")


if __name__ == "__main__":
    main()
