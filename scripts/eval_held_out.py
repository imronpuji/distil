"""
eval_held_out.py
=================
Generate jawaban model untuk semua prompt di held-out test set (data yang
TIDAK PERNAH dipakai training), buat eval manual. Load model sekali, generate
semua prompt, simpan hasilnya biar bisa dibaca/dibandingin.

Cara pakai:
    python scripts/eval_held_out.py --model checkpoints/qwen05b-sft-v1 \\
        --data data/held_out_test.jsonl --out eval_results.jsonl

    # batasi jumlah dulu buat cek cepat
    python scripts/eval_held_out.py --model checkpoints/qwen05b-sft-v1 --limit 20
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate jawaban model untuk held-out test set.")
    parser.add_argument("--model", type=str, default="checkpoints/qwen05b-sft-v1")
    parser.add_argument("--data", type=str, default="data/held_out_test.jsonl")
    parser.add_argument("--out", type=str, default="eval_results.jsonl")
    parser.add_argument("--limit", type=int, default=None,
                         help="Batasi jumlah prompt yang dites (buat cek cepat).")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading {args.model} ke {device}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="bfloat16").to(device)
    model.eval()

    rows = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]
    print(f"Total prompt: {len(rows)}")

    def generate(messages):
        # Pakai system+user asli dari data (kalau ada), biar konsisten sama training.
        prompt_messages = [m for m in messages if m["role"] in ("system", "user")]
        text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    n_done = 0
    with open(args.out, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows, start=1):
            messages = row["messages"]
            instruction = next((m["content"] for m in messages if m["role"] == "user"), "")
            expected = next((m["content"] for m in messages if m["role"] == "assistant"), "")
            generated = generate(messages)

            out_f.write(json.dumps({
                "instruction": instruction,
                "expected": expected,
                "generated": generated,
                "domain": row.get("domain"),
            }, ensure_ascii=False) + "\n")
            out_f.flush()
            n_done += 1
            if n_done % 20 == 0 or n_done == len(rows):
                print(f"[{n_done}/{len(rows)}] selesai", flush=True)

    print(f"\nSelesai. Hasil tersimpan di: {args.out}")
    print("Buka file itu dan baca manual -- bandingkan 'generated' vs 'expected',")
    print("cek relevansi, grammar, dan halusinasi.")


if __name__ == "__main__":
    main()
