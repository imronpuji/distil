"""
generate_gemma_dataset.py (v2)
================================
Generate dataset instruksi+jawaban dari model lokal (Gemma 4) yang di-serve
lewat Ollama di GPU kamu sendiri -- gratis, nggak kena biaya API kayak
Grok/DeepSeek. Reuse logic batching/domain/dedup dari generate_dataset.py,
cuma beda provider & default-nya aja.

Prasyarat:
    ollama serve                 # kalau belum jalan (systemd nggak ada di container)
    ollama pull gemma4

Cara pakai:
    python scripts/v2/generate_gemma_dataset.py --total 5000 --batch-size 20 --workers 4

    # cek cepat dulu, pastiin koneksi & format Gemma4 oke
    python scripts/v2/generate_gemma_dataset.py --total 10 --batch-size 5 --workers 2

Catatan penting -- BUKAN buat CoT:
    Gemma 4 (setahu kami) itu model instruct biasa, BUKAN reasoning model --
    dia nggak balikin reasoning_content terpisah kayak Grok/DeepSeek. Jadi
    skrip ini generate data v1-style (instruksi+jawaban langsung), BUKAN
    chain-of-thought. Kalau ternyata Gemma4 kamu punya kemampuan "berpikir"
    eksplisit (tag <think> dsb muncul natural di output-nya), kasih tau --
    perlu skrip beda (mirip generate_cot_dataset.py) buat nangkep itu dengan
    benar.
"""

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from generate_dataset import (  # noqa: E402
    RateLimiter,
    build_tasks,
    load_done_ids,
    process_batch,
    DOMAINS,
)
import json  # noqa: E402


def check_ollama_connection(host, model):
    """Cek server Ollama nyala & model-nya udah ke-pull, biar gagalnya jelas
    di awal daripada nyoba ratusan request dulu baru ketauan error."""
    try:
        resp = requests.get(f"{host.rstrip('/')}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
    except Exception as e:  # noqa: BLE001
        print(f"[error] Nggak bisa konek ke Ollama di {host}: {e}", file=sys.stderr)
        print("        Jalankan 'ollama serve' dulu (di background kalau systemd nggak ada).",
              file=sys.stderr)
        sys.exit(1)

    if not any(model in m for m in models):
        print(f"[peringatan] Model '{model}' belum kelihatan di 'ollama list' ({models}).",
              file=sys.stderr)
        print(f"             Jalankan 'ollama pull {model}' dulu kalau belum.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset instruksi+jawaban dari Gemma 4 lokal via Ollama.")
    parser.add_argument("--out", type=str, default="data/v2/gemma_dataset.jsonl")
    parser.add_argument("--model", type=str, default="gemma4")
    parser.add_argument("--host", type=str, default="http://localhost:11434")
    parser.add_argument("--total", type=int, default=5000,
                         help="Total pasangan instruksi+jawaban yang mau digenerate.")
    parser.add_argument("--batch-size", type=int, default=20,
                         help="Berapa pasang diminta sekaligus per request. Model lokal biasanya "
                              "lebih lambat dari API cloud -- mulai kecil (10-20) dulu.")
    parser.add_argument("--workers", type=int, default=2,
                         help="Paralelisme. GPU lokal biasanya cuma sanggup 1-4 request "
                              "bersamaan tanpa OOM/melambat drastis -- beda dari API cloud yang "
                              "bisa puluhan.")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--rpm", type=int, default=0,
                         help="Rate limit -- biasanya nggak perlu buat model lokal (0 = tanpa batas).")
    parser.add_argument("--domains", type=str, default=None,
                         help="Batasi ke domain tertentu, pisahkan koma (default: semua domain).")
    args = parser.parse_args()

    check_ollama_connection(args.host, args.model)

    domains = [d.strip() for d in args.domains.split(",")] if args.domains else None
    if domains:
        unknown = [d for d in domains if d not in DOMAINS]
        if unknown:
            print(f"Domain tidak dikenal: {unknown}. Pilihan valid: {list(DOMAINS.keys())}",
                  file=sys.stderr)
            sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    err_path = out_path.with_suffix(out_path.suffix + ".errors.jsonl")

    seen_ids = load_done_ids(args.out)
    seen_lock = threading.Lock()
    print(f"Sudah ada {len(seen_ids)} pasangan di {args.out} (otomatis di-skip kalau muncul lagi).")

    tasks = build_tasks(args.total, max(args.batch_size, 1), domains)

    n_done, n_err, n_dup = 0, 0, 0
    t0 = time.time()
    limiter = RateLimiter(args.rpm)

    print(f"Mengirim {len(tasks)} request ke Ollama ({args.host}, model={args.model}), "
          f"target {args.total} pasangan, workers={args.workers}...", flush=True)

    with out_path.open("a", encoding="utf-8") as out_f, err_path.open("a", encoding="utf-8") as err_f, \
         ThreadPoolExecutor(max_workers=args.workers) as executor:

        futures = {
            executor.submit(process_batch, task, "ollama", args.host, args.model,
                             args.temperature, limiter, seen_ids, seen_lock): task
            for task in tasks
        }
        for fut in as_completed(futures):
            task = futures[fut]
            results, errors, dup = fut.result()
            for r in results:
                r["source"] = "gemma-ollama"  # koreksi -- process_batch/make_row hardcode "grok-freeform"
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            for e in errors:
                err_f.write(json.dumps(e, ensure_ascii=False) + "\n")
            out_f.flush()
            err_f.flush()

            n_done += len(results)
            n_err += len(errors)
            n_dup += dup
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            print(f"[{n_done}/{args.total} pasangan] domain={task['domain']} gaya={task['gaya']} "
                  f"+{len(results)} dup={dup} err={len(errors)} ({rate:.2f} it/s, {elapsed:.0f}s elapsed)",
                  flush=True)

    print(f"\nSelesai. Sukses: {n_done}, duplikat: {n_dup}, error: {n_err}.")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
