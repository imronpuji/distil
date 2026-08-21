"""
try_prompt.py
==============
Coba prompt interaktif ke model hasil SFT (atau model apa aja), buat eval manual
cepat. Jalankan di instance GPU tempat checkpoint-nya ada.

Cara pakai:
    # interaktif, ketik prompt satu-satu
    python scripts/try_prompt.py --model checkpoints/qwen05b-sft-v1

    # sekali jalan, satu prompt
    python scripts/try_prompt.py --model checkpoints/qwen05b-sft-v1 \\
        --prompt "Jelaskan tentang cara kerja fotosintesis."

    # bandingin sama baseline (model asli belum di-tune)
    python scripts/try_prompt.py --model Qwen/Qwen2.5-0.5B-Instruct \\
        --prompt "Jelaskan tentang cara kerja fotosintesis."
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Coba prompt ke model (interaktif atau sekali jalan).")
    parser.add_argument("--model", type=str, default="checkpoints/qwen05b-sft-v1")
    parser.add_argument("--prompt", type=str, default=None,
                         help="Kalau diisi, jalan sekali terus keluar. Kalau kosong, mode interaktif.")
    parser.add_argument("--system", type=str,
                         default="Kamu adalah asisten AI berbahasa Indonesia yang membantu, jujur, dan sopan. "
                                 "Jawab selalu dalam Bahasa Indonesia yang baku dan formal, jelas, dan tidak "
                                 "bertele-tele.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading {args.model} ke {device}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="bfloat16").to(device)
    model.eval()

    def generate(prompt):
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    if args.prompt:
        print("=" * 60)
        print("PROMPT:", args.prompt)
        print("RESPONSE:", generate(args.prompt))
        return

    print("Mode interaktif. Ketik prompt, 'quit'/'exit' buat keluar.\n")
    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit"):
            break
        print(generate(prompt))
        print()


if __name__ == "__main__":
    main()
