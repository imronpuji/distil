"""
generate_dataset.py
====================
Generate dataset instruction-response Bahasa Indonesia menggunakan
Gemma 4 E4B sebagai teacher, di-serve lokal lewat Ollama
(http://localhost:11434 secara default).

Dijalankan di instance tempat Ollama berjalan (mis. EC2 g5g), BUKAN di
sini — script ini tidak bisa dites langsung karena Ollama tidak
tersedia di environment ini.

Contoh pemakaian:
    # pastikan model sudah ditarik:
    #   ollama pull gemma4:e4b

    python scripts/generate_dataset.py \\
        --seeds data/seed_prompts.jsonl \\
        --out data/generated.jsonl \\
        --model gemma4:e4b \\
        --host http://localhost:11434 \\
        --workers 2 \\
        --temperature 0.8 \\
        --num-samples 1

Fitur:
- Resume otomatis: skip seed yang sudah punya hasil di file output.
- Retry dengan backoff untuk request yang gagal/timeout.
- System prompt disesuaikan dengan gaya (formal/santai) tiap seed.
- Opsional multi-sample per prompt (--num-samples > 1) untuk nanti
  difilter/self-consistency filtering.
- Output langsung dalam format ChatML JSONL, siap untuk tahap SFT.
- Error di-log terpisah ke <out>.errors.jsonl supaya tidak menghentikan
  proses keseluruhan.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

SYSTEM_PROMPTS = {
    "formal": (
        "Kamu adalah asisten AI berbahasa Indonesia yang membantu, jujur, dan sopan. "
        "Jawab selalu dalam Bahasa Indonesia yang baku dan formal, jelas, dan tidak bertele-tele. "
        "Jangan mencampur dengan bahasa lain kecuali istilah teknis yang memang tidak ada padanannya."
    ),
    "santai": (
        "Kamu adalah asisten AI berbahasa Indonesia yang membantu dan ramah, seperti mengobrol dengan teman dekat. "
        "Jawab selalu dalam Bahasa Indonesia sehari-hari yang santai dan natural, boleh pakai kata seperti "
        "'kamu', 'nih', 'ya', tapi tetap sopan dan jelas. Jangan mencampur dengan bahasa lain."
    ),
}

DEFAULT_SYSTEM = SYSTEM_PROMPTS["formal"]


def load_jsonl(path):
    items = []
    p = Path(path)
    if not p.exists():
        return items
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_done_ids(out_path):
    done = set()
    for row in load_jsonl(out_path):
        if "id" in row:
            done.add(row["id"])
    return done


def call_ollama(host, model, system_prompt, user_prompt, temperature, timeout=120):
    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # Ollama /api/chat non-stream response: {"message": {"role": "assistant", "content": "..."}, ...}
    return data["message"]["content"].strip()


def call_with_retry(host, model, system_prompt, user_prompt, temperature, retries=3, backoff=3):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return call_ollama(host, model, system_prompt, user_prompt, temperature)
        except Exception as e:  # noqa: BLE001 - want to catch & retry any transient error
            last_err = e
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise last_err


def process_seed(seed, host, model, temperature, num_samples):
    gaya = seed.get("gaya", "formal")
    system_prompt = SYSTEM_PROMPTS.get(gaya, DEFAULT_SYSTEM)
    instruction = seed["instruction"]

    results = []
    errors = []
    for sample_idx in range(num_samples):
        try:
            start = time.time()
            response = call_with_retry(host, model, system_prompt, instruction, temperature)
            latency = round(time.time() - start, 2)
            row_id = seed["id"] if num_samples == 1 else f"{seed['id']}_s{sample_idx}"
            results.append({
                "id": row_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response},
                ],
                "domain": seed.get("domain"),
                "gaya": gaya,
                "source": seed.get("source"),
                "meta": {
                    "model": model,
                    "temperature": temperature,
                    "latency_s": latency,
                    "sample_idx": sample_idx,
                },
            })
        except Exception as e:  # noqa: BLE001
            errors.append({"id": seed["id"], "sample_idx": sample_idx, "error": str(e)})
    return results, errors


def main():
    parser = argparse.ArgumentParser(description="Generate dataset via Ollama teacher model.")
    parser.add_argument("--seeds", type=str, default="data/seed_prompts.jsonl")
    parser.add_argument("--out", type=str, default="data/generated.jsonl")
    parser.add_argument("--model", type=str, default="gemma4:e4b")
    parser.add_argument("--host", type=str, default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--num-samples", type=int, default=1,
                         help="Jumlah sample per prompt (untuk self-consistency filtering nanti).")
    parser.add_argument("--workers", type=int, default=2,
                         help="Jumlah request paralel. Mulai kecil (1-2) dulu, naikkan sesuai kapasitas GPU.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Batasi jumlah seed yang diproses (untuk uji coba cepat).")
    args = parser.parse_args()

    seeds = load_jsonl(args.seeds)
    if not seeds:
        print(f"Tidak ada seed ditemukan di {args.seeds}. Jalankan seed_bank.py dulu.", file=sys.stderr)
        sys.exit(1)

    done_ids = load_done_ids(args.out)
    pending = [s for s in seeds if s["id"] not in done_ids]
    if args.limit:
        pending = pending[: args.limit]

    print(f"Total seed: {len(seeds)} | sudah selesai: {len(done_ids)} | akan diproses: {len(pending)}")
    if not pending:
        print("Tidak ada yang perlu diproses. Selesai.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    err_path = out_path.with_suffix(out_path.suffix + ".errors.jsonl")

    n_done = 0
    n_err = 0
    t0 = time.time()

    with out_path.open("a", encoding="utf-8") as out_f, \
         err_path.open("a", encoding="utf-8") as err_f, \
         ThreadPoolExecutor(max_workers=args.workers) as executor:

        futures = {
            executor.submit(process_seed, seed, args.host, args.model, args.temperature, args.num_samples): seed
            for seed in pending
        }

        print(f"Mengirim {len(pending)} request ke {args.host} (model={args.model}, workers={args.workers})...",
              flush=True)

        for fut in as_completed(futures):
            seed = futures[fut]
            results, errors = fut.result()
            for r in results:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            for e in errors:
                err_f.write(json.dumps(e, ensure_ascii=False) + "\n")
            out_f.flush()
            err_f.flush()

            n_done += len(results)
            n_err += len(errors)
            total_processed = n_done + n_err
            elapsed = time.time() - t0
            rate = total_processed / elapsed if elapsed > 0 else 0
            latency = results[0]["meta"]["latency_s"] if results else None
            status = "OK" if results else "ERR"
            print(f"[{total_processed}/{len(pending)}] {status} id={seed['id']} "
                  f"latency={latency}s done={n_done} err={n_err} "
                  f"({rate:.2f} it/s, {elapsed:.0f}s elapsed)", flush=True)

    print(f"\nSelesai. Total sukses: {n_done}, error: {n_err}.")
    print(f"Output: {out_path}")
    if n_err:
        print(f"Error log: {err_path} (bisa di-retry dengan menjalankan ulang script ini, "
              f"seed yang error TIDAK dianggap 'done' jadi otomatis dicoba lagi).")


if __name__ == "__main__":
    main()
