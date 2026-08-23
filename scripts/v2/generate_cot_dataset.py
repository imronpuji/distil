"""
generate_cot_dataset.py (v2)
=============================
Generate dataset instruksi+jawaban BER-CHAIN-OF-THOUGHT, ditangkap dari model
REASONING (grok-4.6), bukan cuma jawaban akhir kayak generate_dataset.py (v1).

Kenapa beda dari v1: kalau Grok reasoning diminta generate N pasang sekaligus
dalam satu JSON array, reasoning_content-nya itu "mikirin cara nyusun N pasang",
BUKAN reasoning buat tiap soal. Biar CoT-nya beneran per-soal, harus panggil
SATU PER SATU -- lebih mahal & lambat dari v1, makanya cuma dipakai buat
instruksi yang genuinely butuh reasoning (bukan chitchat/fakta sederhana).

Dua tahap:
  1. Generate instruksi yang butuh reasoning (batch, model non-reasoning, murah)
     -- soal matematika cerita, logika, debugging kode, perencanaan multi-langkah,
     analisis sebab-akibat, trade-off decision making.
  2. Untuk tiap instruksi, panggil model reasoning SATU PER SATU, tangkap
     reasoning_content (proses mikir) + content (jawaban akhir), gabung jadi
     satu respons berformat "<think>...</think>\\n\\n<jawaban akhir>".

Cara pakai:
    export XAI_API_KEY=xai-...   # atau taruh di .env root project
    python scripts/v2/generate_cot_dataset.py --total 300 --workers 6

    # cek cepat dulu
    python scripts/v2/generate_cot_dataset.py --total 10 --workers 3
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from generate_dataset import (  # noqa: E402
    RateLimiter,
    call_with_retry,
    load_dotenv,
    load_done_ids,
    parse_json_array,
    _hash_id,
    XAI_API_URL,
)

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

REASONING_PROVIDERS = {
    "xai": {"url": XAI_API_URL, "env_key": "XAI_API_KEY"},
    "deepseek": {"url": DEEPSEEK_API_URL, "env_key": "DEEPSEEK_API_KEY"},
}

# Topik yang genuinely butuh proses berpikir bertahap, bukan sekadar hafalan/lookup.
REASONING_TOPICS = {
    "matematika_cerita": [
        "soal cerita pembagian/perkalian sehari-hari", "soal cerita persentase diskon",
        "soal cerita kecepatan dan waktu tempuh", "soal cerita perbandingan harga",
        "soal cerita hitung untung-rugi jualan",
    ],
    "logika": [
        "teka-teki logika sederhana", "soal deduksi siapa-melakukan-apa",
        "soal urutan berdasarkan petunjuk", "paradoks atau teka-teki berpikir",
    ],
    "debugging_kode": [
        "kode Python yang hasilnya salah tapi tidak error", "off-by-one error di loop",
        "logika kondisional yang salah urutan", "bug di rekursi sederhana",
    ],
    "perencanaan": [
        "menyusun jadwal dengan beberapa kendala waktu", "merencanakan anggaran dengan prioritas",
        "menyusun urutan tugas dengan dependency", "memilih opsi terbaik dari beberapa trade-off",
    ],
    "sebab_akibat": [
        "kenapa suatu keputusan bisnis gagal", "menganalisis penyebab suatu masalah teknis",
        "menghubungkan beberapa faktor jadi satu kesimpulan",
    ],
}

REASONING_SYSTEM_PROMPT = (
    "Kamu adalah asisten AI berbahasa Indonesia yang menyelesaikan masalah dengan penalaran "
    "bertahap yang jelas dan runtut sebelum memberi jawaban akhir. "
    "PENTING: seluruh proses berpikirmu HARUS dalam Bahasa Indonesia -- jangan mikir dalam "
    "Bahasa Inggris lalu terjemahkan, dan jangan menerjemahkan ulang soalnya ke Bahasa Inggris "
    "di awal. Langsung bernalar dalam Bahasa Indonesia dari kalimat pertama."
)


def build_instruction_batch_prompt(topic_key, topics, n):
    topik_str = ", ".join(topics)
    return (
        f"Buatkan {n} instruksi/soal dalam Bahasa Indonesia bertema '{topic_key}', terinspirasi "
        f"dari: {topik_str}. Tiap instruksi HARUS butuh penalaran bertahap buat dijawab dengan "
        "benar (bukan sekadar definisi atau fakta hafalan) -- pengguna beneran perlu mikir "
        "langkah demi langkah. Instruksi harus natural, seolah ditulis pengguna asli, dan "
        "BERBEDA satu sama lain.\n\n"
        "Balas HANYA dengan JSON array berisi string instruksi, tanpa markdown code fence. "
        f"Tepat {n} elemen."
    )


INSTRUCTION_BATCH_SIZE = 20  # jangan minta terlalu banyak sekaligus -- 120/request
                             # kebukti bikin completion kegedean, timeout & JSON pecah


def call_provider_text(provider, model, system_prompt, user_prompt, temperature, limiter,
                        retries=3, backoff=3):
    """Panggil xai atau deepseek, balikin cuma teks jawaban (content) -- dipakai buat
    tahap 1 (generate instruksi) yang nggak butuh reasoning_content. Untuk xai, reuse
    call_with_retry yang sudah ada; untuk deepseek, request manual (API-nya OpenAI-compatible)."""
    if provider == "xai":
        return call_with_retry("xai", None, model, system_prompt, user_prompt, temperature, limiter)

    conf = REASONING_PROVIDERS[provider]
    api_key = os.environ.get(conf["env_key"])
    if not api_key:
        raise RuntimeError(f"{conf['env_key']} tidak ditemukan di environment.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "temperature": temperature,
    }
    if provider == "deepseek":
        # deepseek-v4-flash reasoning selalu nyala secara default (beda dari xAI yang punya
        # varian -non-reasoning eksplisit) -- matiin di sini karena tahap 1 cuma generate
        # daftar soal, nggak butuh proses mikir, dan reasoning bikin jauh lebih lambat/mahal.
        payload["thinking"] = {"type": "disabled"}
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            limiter.acquire()
            resp = requests.post(conf["url"], json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise last_err


def generate_instructions(provider, model, temperature, limiter, n_per_topic):
    """Tahap 1: generate instruksi yang butuh reasoning, pakai model biasa (murah).
    Di-chunk kecil-kecil (INSTRUCTION_BATCH_SIZE per request) + retry, biar nggak
    kena timeout/JSON-parse-error kayak waktu minta ratusan sekaligus dalam 1 call."""
    all_instructions = []
    for topic_key, topics in REASONING_TOPICS.items():
        remaining = n_per_topic
        while remaining > 0:
            n = min(INSTRUCTION_BATCH_SIZE, remaining)
            prompt = build_instruction_batch_prompt(topic_key, topics, n)
            try:
                raw = call_provider_text(provider, model,
                                          "Kamu asisten pembuat soal latihan berbahasa Indonesia.",
                                          prompt, temperature, limiter)
                items = parse_json_array(raw)
                for instr in items:
                    all_instructions.append({"instruction": str(instr).strip(), "topic": topic_key})
            except Exception as e:  # noqa: BLE001
                print(f"[peringatan] gagal generate {n} instruksi topik {topic_key}: {e}", file=sys.stderr)
            remaining -= n
    return all_instructions


def solve_with_reasoning(item, provider, model, temperature, limiter, retries=3, backoff=3):
    """Tahap 2: panggil model reasoning SATU PER SATU, tangkap reasoning_content + content.
    Provider xai (Grok) atau deepseek -- keduanya OpenAI-compatible & sama-sama balikin
    reasoning_content terpisah dari content, jadi satu fungsi ini cukup buat keduanya."""
    conf = REASONING_PROVIDERS[provider]
    api_key = os.environ.get(conf["env_key"])
    if not api_key:
        return None, f"{conf['env_key']} tidak ditemukan di environment."
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": REASONING_SYSTEM_PROMPT},
            {"role": "user", "content": item["instruction"]},
        ],
        "temperature": temperature,
    }

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            limiter.acquire()
            resp = requests.post(conf["url"], json=payload, headers=headers, timeout=180)
            resp.raise_for_status()
            message = resp.json()["choices"][0]["message"]
            reasoning = (message.get("reasoning_content") or "").strip()
            answer = (message.get("content") or "").strip()
            if not answer:
                raise ValueError("Jawaban akhir kosong.")

            if reasoning:
                combined = f"<think>\n{reasoning}\n</think>\n\n{answer}"
            else:
                combined = answer  # fallback kalau API kebetulan nggak balikin reasoning_content

            return {
                "id": _hash_id(item["instruction"]),
                "messages": [
                    {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                    {"role": "user", "content": item["instruction"]},
                    {"role": "assistant", "content": combined},
                ],
                "domain": f"reasoning_{item['topic']}",
                "gaya": "formal",
                "source": f"{provider}-cot",
                "meta": {"provider": provider, "model": model, "temperature": temperature,
                         "has_reasoning": bool(reasoning)},
            }, None
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries:
                time.sleep(backoff * attempt)
    return None, str(last_err)


def main():
    parser = argparse.ArgumentParser(description="Generate dataset CoT dari model reasoning Grok.")
    parser.add_argument("--out", type=str, default="data/v2/cot_dataset.jsonl")
    parser.add_argument("--reasoning-provider", type=str, choices=list(REASONING_PROVIDERS), default="xai",
                         help="Provider buat tahap 2 (nangkep chain-of-thought). "
                              "deepseek butuh DEEPSEEK_API_KEY di .env.")
    parser.add_argument("--instruction-provider", type=str, choices=list(REASONING_PROVIDERS), default=None,
                         help="Provider buat tahap 1 (generate instruksi). Default: ngikutin "
                              "--reasoning-provider (biar --reasoning-provider deepseek otomatis "
                              "pakai deepseek buat semua tahap).")
    parser.add_argument("--instruction-model", type=str, default=None,
                         help="Model buat tahap 1. Default: grok-4-fast-non-reasoning (xai) atau "
                              "deepseek-v4-flash (deepseek).")
    parser.add_argument("--reasoning-model", type=str, default=None,
                         help="Model reasoning tahap 2. Default: grok-4.6 (xai) atau "
                              "deepseek-v4-flash (deepseek).")
    parser.add_argument("--total", type=int, default=300,
                         help="Total instruksi yang mau diselesaikan pakai reasoning model.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=6,
                         help="Paralelisme tahap 2. Model reasoning lebih lambat/mahal, "
                              "jangan set terlalu tinggi.")
    parser.add_argument("--rpm", type=int, default=0)
    args = parser.parse_args()

    load_dotenv()

    if args.instruction_provider is None:
        args.instruction_provider = args.reasoning_provider
    if args.instruction_model is None:
        args.instruction_model = ("grok-4-fast-non-reasoning" if args.instruction_provider == "xai"
                                   else "deepseek-v4-flash")
    if args.reasoning_model is None:
        args.reasoning_model = "grok-4.6" if args.reasoning_provider == "xai" else "deepseek-v4-flash"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    err_path = out_path.with_suffix(out_path.suffix + ".errors.jsonl")
    done_ids = load_done_ids(args.out)

    limiter = RateLimiter(args.rpm)
    n_topics = len(REASONING_TOPICS)
    n_per_topic = max(1, -(-args.total // n_topics))  # ceil division, generate sedikit lebih banyak

    print(f"Tahap 1: generate instruksi reasoning ({n_per_topic}/topik x {n_topics} topik) "
          f"pakai {args.instruction_provider}/{args.instruction_model}...")
    instructions = generate_instructions(args.instruction_provider, args.instruction_model,
                                          args.temperature, limiter, n_per_topic)
    instructions = [it for it in instructions if _hash_id(it["instruction"]) not in done_ids]
    instructions = instructions[: args.total]
    print(f"Total instruksi siap diselesaikan: {len(instructions)}")

    if not instructions:
        print("Tidak ada instruksi baru. Selesai.")
        return

    print(f"Tahap 2: selesaikan pakai {args.reasoning_provider}/{args.reasoning_model} (reasoning), "
          f"workers={args.workers}, rpm={args.rpm or 'tanpa batas'}...")
    n_done, n_err = 0, 0
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as out_f, \
         err_path.open("a", encoding="utf-8") as err_f, \
         ThreadPoolExecutor(max_workers=args.workers) as executor:

        futures = {
            executor.submit(solve_with_reasoning, item, args.reasoning_provider, args.reasoning_model,
                             args.temperature, limiter): item
            for item in instructions
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            item = futures[fut]
            row, err = fut.result()
            if row:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done += 1
            else:
                err_f.write(json.dumps({"instruction": item["instruction"], "error": err},
                                        ensure_ascii=False) + "\n")
                err_f.flush()
                n_err += 1
            elapsed = time.time() - t0
            print(f"[{i}/{len(instructions)}] done={n_done} err={n_err} ({elapsed:.0f}s elapsed)", flush=True)

    print(f"\nSelesai. Sukses: {n_done}, error: {n_err}.")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
