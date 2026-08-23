"""
judge_eval.py
=============
Skor otomatis kualitas Bahasa Indonesia dari hasil eval_held_out.py, pakai
Grok sebagai "judge" (LLM-as-judge). Ini pelengkap eval manual, bukan
pengganti -- baca beberapa contoh manual tetap wajib, tapi ini kasih angka
yang bisa dibandingin antar run/checkpoint.

Rubrik (tiap dimensi diskor 1-5):
- relevansi     : jawaban beneran menjawab instruksinya, bukan ngelantur
- tata_bahasa   : grammar Bahasa Indonesia benar, bukan kalimat rusak
- kealamian     : terdengar natural, bukan robotic/kaku/template
- akurasi_fakta : kalau ada klaim faktual, itu benar (bukan halusinasi)
Plus flag boolean:
- campur_bahasa : ada kata Inggris/asing yang nggak wajar nyelip

Cara pakai:
    export XAI_API_KEY=xai-...   # atau taruh di .env
    python scripts/judge_eval.py --in eval_results.jsonl --out eval_scored.jsonl

    # cek cepat dulu
    python scripts/judge_eval.py --in eval_results.jsonl --limit 20
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from generate_dataset import call_xai, load_dotenv, parse_json_array, RateLimiter  # noqa: E402

JUDGE_SYSTEM_PROMPT = (
    "Kamu adalah evaluator kualitas Bahasa Indonesia yang ketat dan objektif. "
    "Tugasmu menilai jawaban asisten AI, BUKAN menjawab instruksinya sendiri."
)


def build_judge_prompt(instruction, generated):
    return (
        f"Instruksi pengguna: {instruction!r}\n"
        f"Jawaban asisten AI yang harus dinilai: {generated!r}\n\n"
        "Nilai jawaban itu pada 4 dimensi, skala 1-5 (1=sangat buruk, 5=sangat baik):\n"
        "- relevansi: apakah jawaban beneran menjawab instruksinya?\n"
        "- tata_bahasa: apakah grammar Bahasa Indonesia-nya benar?\n"
        "- kealamian: apakah terdengar natural, bukan kaku/robotic/template?\n"
        "- akurasi_fakta: kalau ada klaim faktual, apakah benar? (5 kalau tidak ada klaim faktual "
        "yang perlu dicek, atau semua klaim benar)\n"
        "Plus tentukan campur_bahasa (true/false): apakah ada kata Inggris/asing yang nyelip "
        "secara nggak wajar (istilah teknis yang memang lazim dipakai TIDAK dihitung).\n\n"
        "Balas HANYA dengan JSON object, tanpa markdown code fence:\n"
        '{"relevansi": <1-5>, "tata_bahasa": <1-5>, "kealamian": <1-5>, '
        '"akurasi_fakta": <1-5>, "campur_bahasa": <true/false>, "catatan": "<1 kalimat alasan singkat>"}'
    )


def judge_one(row, provider, model, limiter, retries=3):
    prompt = build_judge_prompt(row["instruction"], row["generated"])
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            limiter.acquire()
            raw = call_xai(model, JUDGE_SYSTEM_PROMPT, prompt, temperature=0.0)
            score = parse_json_array(raw)  # nama historis, tapi cuma json.loads + strip fence
            if not isinstance(score, dict):
                raise ValueError(f"Respons judge bukan JSON object: {raw!r}")
            return {**row, "score": score}
        except Exception as e:  # noqa: BLE001
            last_err = e
    return {**row, "score": None, "judge_error": str(last_err)}


def main():
    parser = argparse.ArgumentParser(description="Skor kualitas Bahasa Indonesia pakai LLM-as-judge.")
    parser.add_argument("--in", dest="inp", type=str, default="eval_results.jsonl")
    parser.add_argument("--out", type=str, default="eval_scored.jsonl")
    parser.add_argument("--model", type=str, default="grok-4-fast-non-reasoning")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rpm", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("XAI_API_KEY"):
        print("XAI_API_KEY tidak ditemukan. Set env var atau taruh di .env.", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]
    print(f"Menilai {len(rows)} jawaban pakai model judge: {args.model}")

    limiter = RateLimiter(args.rpm)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(judge_one, row, "xai", args.model, limiter): row for row in rows}
        for i, fut in enumerate(as_completed(futures), start=1):
            results.append(fut.result())
            if i % 20 == 0 or i == len(rows):
                print(f"[{i}/{len(rows)}] dinilai", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    scored = [r for r in results if r.get("score")]
    failed = len(results) - len(scored)
    dims = ["relevansi", "tata_bahasa", "kealamian", "akurasi_fakta"]
    print(f"\n=== RINGKASAN ({len(scored)}/{len(results)} berhasil dinilai, {failed} gagal) ===")
    for d in dims:
        vals = [r["score"][d] for r in scored if d in r["score"]]
        if vals:
            print(f"  {d:15s}: rata-rata {sum(vals)/len(vals):.2f} / 5")
    mixed = sum(1 for r in scored if r["score"].get("campur_bahasa"))
    print(f"  campur_bahasa  : {mixed}/{len(scored)} ({100*mixed/len(scored):.0f}%)")

    print("\nPer domain:")
    domains = sorted(set(r.get("domain") for r in scored))
    for dom in domains:
        sub = [r for r in scored if r.get("domain") == dom]
        avg_relevansi = sum(r["score"]["relevansi"] for r in sub) / len(sub)
        print(f"  {dom or '(tanpa domain)':16s} n={len(sub):3d}  relevansi={avg_relevansi:.2f}")

    print(f"\nHasil lengkap: {args.out}")


if __name__ == "__main__":
    main()
