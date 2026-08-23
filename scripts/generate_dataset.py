"""
generate_dataset.py
====================
Generate dataset instruction-response Bahasa Indonesia. Grok (via xAI API)
yang membuat PASANGAN instruksi + jawabannya sendiri, dari nol -- tidak ada
file seed yang perlu disiapkan manual.

Mendukung dua provider:
- xai (default): Grok via xAI API (https://api.x.ai/v1/chat/completions).
  Butuh env var XAI_API_KEY (lihat .env di root project).
- ollama: model lokal yang di-serve lewat Ollama
  (http://localhost:11434 secara default).

Cara paling gampang -- mode interaktif, tinggal jawab pertanyaan setup-nya:
    python scripts/generate_dataset.py --interactive

Atau langsung lewat argumen:
    export XAI_API_KEY=xai-...   # atau taruh di .env
    python scripts/generate_dataset.py \\
        --total 500 \\
        --batch-size 5 \\
        --rpm 15 \\
        --provider xai \\
        --model grok-4-fast-non-reasoning

Cara kerja:
- Tiap request, model diminta generate N pasang instruksi+jawaban sekaligus
  (N = --batch-size) untuk satu domain & gaya bahasa tertentu, dibalas
  dalam format JSON array. Domain & gaya dirotasi tiap batch biar variatif,
  dengan daftar domain/topik yang sama dengan seed_bank.py (dipakai cuma
  sebagai inspirasi topik, bukan instruksi final -- instruksi & jawaban
  tetap orisinal dari model).
- Rate limit (--rpm) dan batching menekan jumlah request per menit.
- Resume otomatis: pasangan yang instruksinya identik dengan yang sudah ada
  di file output (dicek via hash) otomatis di-skip.
- Retry dengan backoff untuk request yang gagal/timeout; kalau parsing JSON
  gagal, batch di-retry sebagai batch yang lebih kecil (per-pasangan).
- Output langsung dalam format ChatML JSONL, siap untuk tahap SFT.
- Error di-log terpisah ke <out>.errors.jsonl supaya tidak menghentikan
  proses keseluruhan.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

try:
    from seed_bank import DOMAINS  # dipakai sebagai inspirasi topik saja
except ImportError:
    DOMAINS = {"umum": ["topik sehari-hari"]}

XAI_API_URL = "https://api.x.ai/v1/chat/completions"


class RateLimiter:
    """Batasi jumlah request per menit, aman dipakai dari banyak thread sekaligus."""

    def __init__(self, max_per_minute):
        self.max_per_minute = max_per_minute
        self.lock = threading.Lock()
        self.timestamps = deque()

    def acquire(self):
        if not self.max_per_minute:
            return
        while True:
            with self.lock:
                now = time.time()
                while self.timestamps and now - self.timestamps[0] >= 60:
                    self.timestamps.popleft()
                if len(self.timestamps) < self.max_per_minute:
                    self.timestamps.append(now)
                    return
                wait = 60 - (now - self.timestamps[0])
            time.sleep(max(wait, 0.05))


def load_dotenv(path=".env"):
    """Loader .env minimal, tanpa dependency tambahan. Tidak menimpa env var yang sudah di-set."""
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


# Banyak parafrase per gaya, BUKAN satu teks tetap. Dataset v1 cuma punya 2 system
# prompt unik untuk 15.426 baris -- teksnya jadi konstanta yang dihafal model, sampai
# dimuntahkan ulang sebagai jawaban waktu ditanya "Kamu siapa?". Variasi bikin model
# belajar "ikuti arahan system prompt", bukan "hafalkan teks ini".
SYSTEM_PROMPT_VARIANTS = {
    "formal": [
        "Kamu adalah asisten AI berbahasa Indonesia yang membantu, jujur, dan sopan. "
        "Jawab dalam Bahasa Indonesia baku yang jelas dan tidak bertele-tele.",
        "Kamu asisten AI yang menjawab dalam Bahasa Indonesia formal. Berikan jawaban yang "
        "akurat, ringkas, dan mudah dipahami.",
        "Peranmu adalah asisten berbahasa Indonesia. Gunakan bahasa baku, jawab dengan tepat "
        "sesuai pertanyaan yang diajukan.",
        "Kamu adalah asisten virtual berbahasa Indonesia. Jawablah secara formal, faktual, "
        "dan langsung ke inti persoalan.",
        "Jawab setiap pertanyaan dalam Bahasa Indonesia yang baik dan benar. Utamakan "
        "kejelasan dan ketepatan informasi.",
        "Kamu asisten AI profesional. Semua jawaban dalam Bahasa Indonesia formal, singkat "
        "namun lengkap.",
        "Bertindaklah sebagai asisten berbahasa Indonesia yang informatif dan sopan. Hindari "
        "jawaban yang berbelit-belit.",
        "Kamu adalah asisten yang membantu pengguna Indonesia. Gunakan bahasa formal dan "
        "berikan informasi yang benar.",
        "Sebagai asisten AI, jawab dalam Bahasa Indonesia baku. Jujur bila tidak tahu, "
        "jangan mengarang jawaban.",
        "Kamu asisten berbahasa Indonesia. Prioritaskan akurasi, gunakan bahasa resmi yang "
        "tetap mudah dicerna.",
        "Layani pengguna dalam Bahasa Indonesia formal. Pastikan jawaban relevan dengan "
        "pertanyaan yang diajukan.",
    ],
    "santai": [
        "Kamu asisten AI berbahasa Indonesia yang ramah, kayak ngobrol sama teman. Jawab "
        "santai tapi tetap sopan dan jelas.",
        "Kamu temen ngobrol yang asik dan membantu. Pakai Bahasa Indonesia sehari-hari yang "
        "natural.",
        "Jawab pakai bahasa santai sehari-hari ya, kayak lagi chat sama temen. Tetap sopan "
        "dan informatif.",
        "Kamu asisten yang friendly. Ngobrol pakai Bahasa Indonesia kasual, boleh pakai "
        "'kamu', 'nih', 'ya'.",
        "Bantu pengguna dengan gaya santai dan akrab. Bahasa Indonesia sehari-hari, jangan kaku.",
        "Kamu asisten AI yang santai dan gampang diajak ngobrol. Jawab dengan bahasa yang "
        "natural dan nggak berjarak.",
        "Ngobrol sama pengguna pakai bahasa yang santai dan hangat. Tetap jujur dan membantu.",
        "Kamu asisten berbahasa Indonesia dengan gaya kasual. Jawab seperti teman yang paham "
        "banyak hal.",
        "Pakai Bahasa Indonesia sehari-hari yang luwes. Jangan formal-formal amat, yang "
        "penting jelas.",
        "Kamu asisten yang ramah dan nggak kaku. Jawab pakai bahasa obrolan sehari-hari.",
        "Jadi teman ngobrol yang membantu. Bahasa santai, sopan, dan gampang dimengerti.",
    ],
}

# Sebagian baris sengaja TANPA system prompt sama sekali, supaya model nggak bergantung
# pada kehadirannya dan tetap waras waktu dipanggil tanpa system prompt.
NO_SYSTEM_RATIO = 0.2

GAYA_LIST = list(SYSTEM_PROMPT_VARIANTS.keys())


def pick_system_prompt(gaya):
    """Ambil satu parafrase acak, atau None (~NO_SYSTEM_RATIO dari waktu)."""
    if random.random() < NO_SYSTEM_RATIO:
        return None
    return random.choice(SYSTEM_PROMPT_VARIANTS[gaya])


def steering_prompt(gaya):
    """System prompt untuk MENYETIR generasi Grok (bukan yang disimpan ke dataset).
    Sengaja dipatok ke satu varian biar gaya output teacher konsisten."""
    return SYSTEM_PROMPT_VARIANTS[gaya][0]


def _hash_id(text):
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()[:12]


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


def call_ollama(host, model, system_prompt, user_prompt, temperature, timeout=180):
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


def call_xai(model, system_prompt, user_prompt, temperature, timeout=180):
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY tidak ditemukan di environment (cek .env di root project).")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(XAI_API_URL, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def call_with_retry(provider, host, model, system_prompt, user_prompt, temperature, limiter,
                     retries=3, backoff=3):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            limiter.acquire()
            if provider == "xai":
                return call_xai(model, system_prompt, user_prompt, temperature)
            return call_ollama(host, model, system_prompt, user_prompt, temperature)
        except Exception as e:  # noqa: BLE001 - want to catch & retry any transient error
            last_err = e
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise last_err


# Domain percakapan pendek: sapaan, basa-basi, acknowledgment, tanya identitas.
# Ini yang HILANG di dataset v1 dan bikin model bingung waktu dikasih input pendek
# kayak "hi", "ok", "siap" -- dia malah ngarang jawaban panjang yang nggak nyambung.
CHITCHAT_DOMAINS = {
    "sapaan": (
        "sapaan pembuka super pendek (1-3 kata) seperti 'hi', 'halo', 'pagi', 'malam', "
        "'permisi', 'assalamualaikum', 'woy', 'hei'"
    ),
    "acknowledgment": (
        "respons singkat pengguna yang cuma mengiyakan/menutup obrolan, seperti 'ok', 'oke', "
        "'siap', 'sip', 'makasih', 'thanks', 'baik', 'noted', 'gitu ya', 'oh gitu', 'hmm'"
    ),
    "identitas": (
        "pertanyaan tentang identitas & kemampuan asisten, seperti 'kamu siapa?', 'nama kamu apa?', "
        "'kamu bisa apa aja?', 'kamu robot ya?', 'kamu bisa ngoding?', 'kamu buatan siapa?'"
    ),
    "basa_basi": (
        "basa-basi sosial pendek seperti 'apa kabar?', 'lagi apa?', 'sibuk nggak?', 'udah makan?', "
        "'gimana harimu?', 'bosen nih', 'capek banget'"
    ),
}


def build_chitchat_prompt(domain, gaya, n):
    """Prompt khusus buat data percakapan pendek. Beda dari domain biasa: instruksinya
    HARUS pendek, dan jawabannya juga harus pendek + wajar (bukan ceramah panjang)."""
    desc = CHITCHAT_DOMAINS[domain]
    gaya_desc = "formal dan sopan" if gaya == "formal" else "santai sehari-hari"
    return (
        f"Buatkan {n} pasang percakapan pendek dalam Bahasa Indonesia, kategori: {desc}.\n\n"
        "ATURAN PENTING:\n"
        f"1. Field \"instruction\" HARUS pendek (1-6 kata), persis seperti yang diketik pengguna asli "
        "waktu chat -- termasuk variasi kapitalisasi & typo ringan yang wajar (mis. 'Hi', 'halo', "
        "'oke', 'Kamu siapa?').\n"
        f"2. Field \"response\" HARUS pendek juga (1-2 kalimat, maksimal 25 kata) dan gaya {gaya_desc}. "
        "JANGAN menjawab dengan penjelasan panjang, daftar bernomor, atau ceramah -- ini cuma obrolan "
        "ringan, bukan pertanyaan yang butuh artikel.\n"
        "3. Jawaban harus NYAMBUNG dengan sapaan/pertanyaannya. Kalau pengguna cuma bilang 'ok', "
        "balas singkat & wajar (mis. 'Siap, kalau ada yang mau ditanyakan lagi tinggal bilang ya.') "
        "-- JANGAN tiba-tiba membahas topik acak.\n"
        "4. Untuk pertanyaan identitas, jawab sebagai asisten AI berbahasa Indonesia yang membantu. "
        "JANGAN mengarang nama orang, kota asal, atau biodata palsu, dan JANGAN menyalin ulang "
        "kalimat instruksi sistem.\n"
        "5. JANGAN PERNAH menyebut nama perusahaan atau produk AI tertentu (OpenAI, ChatGPT, GPT, "
        "Google, Gemini, Anthropic, Claude, Meta, xAI, Grok, dsb) sebagai pembuat atau identitas "
        "asisten -- itu klaim yang salah. Untuk pertanyaan 'kamu buatan siapa?' atau 'kamu dari mana?', "
        "jawab netral tanpa menyebut vendor, misalnya 'Saya asisten AI, dikembangkan untuk membantu "
        "pengguna berbahasa Indonesia.'\n"
        "6. Setiap pasangan harus berbeda satu sama lain.\n\n"
        "Balas HANYA dengan JSON array, tanpa markdown code fence atau teks lain di luar JSON. "
        f"Tepat {n} elemen, tiap elemen berupa object dengan dua field: "
        '"instruction" (string) dan "response" (string).'
    )


# Domain berisi fakta yang bisa dicek benar-salahnya (beda dari domain tips/opini
# kayak kesehatan/karir yang lebih longgar). Ditambah setelah eval nemuin halusinasi
# konkret: "Jogja adalah kota di Jawa Barat" (salah, itu DIY/Jawa Tengah).
FACT_HEAVY_DOMAINS = {"pemerintahan", "geografi_indonesia", "wisata_indonesia",
                      "perusahaan_publik", "kerusakan_lingkungan", "tambang"}


def build_freeform_prompt(domain, topics, gaya, n):
    if domain in CHITCHAT_DOMAINS:
        return build_chitchat_prompt(domain, gaya, n)
    topik_str = ", ".join(topics)
    accuracy_note = (
        "\n\nPERINGATAN KHUSUS domain ini berisi FAKTA yang bisa dicek benar-salahnya "
        "(nama provinsi, letak kota, nama pejabat/lembaga, nama perusahaan, dsb). "
        "SEBUAH KESALAHAN FAKTA (mis. salah sebut provinsi suatu kota) lebih buruk daripada "
        "jawaban pendek. Kalau ragu suatu detail spesifik (angka pasti, nama pejabat saat ini, "
        "data terbaru), JANGAN mengarang -- jawab secara umum/aman atau nyatakan itu bisa berubah "
        "seiring waktu, daripada menyebut detail yang belum tentu benar."
        if domain in FACT_HEAVY_DOMAINS else ""
    )
    gaya_desc = "formal dan baku" if gaya == "formal" else "santai sehari-hari"
    # Jumlah eksplisit, bukan "sekitar sepertiga" -- pecahan samar bikin teacher menjawab
    # pendek semua (terukur: 63% jawaban <=20 kata, jauh dari target 33%).
    # Porsi "long" sengaja dilebihkan (0.45) karena teacher konsisten kurang memenuhinya:
    # diminta 30% hanya terpenuhi 10%. Ini kompensasi bias, bukan target sebenarnya.
    n_short = max(1, round(n * 0.25))
    n_long = max(1, round(n * 0.45))
    n_mid = max(1, n - n_short - n_long)
    return (
        f"Buatkan {n} pasang instruksi-jawaban dalam Bahasa Indonesia untuk domain '{domain}'. "
        f"Boleh terinspirasi dari topik-topik ini: {topik_str} -- tapi jangan cuma mengulang "
        "kalimat topiknya mentah-mentah, buat instruksi yang natural seolah ditulis pengguna asli "
        "(bisa berupa pertanyaan, permintaan bantuan, curhat singkat, permintaan dibuatkan sesuatu, dst). "
        f"Gaya bahasa jawabannya {gaya_desc}. Setiap pasangan harus punya instruksi yang BERBEDA satu "
        "sama lain (jangan duplikat/mirip persis), dan jawabannya harus benar-benar menjawab instruksinya "
        "dengan lengkap dan natural.\n\n"
        f"WAJIB VARIASIKAN BOBOT PERTANYAAN dan panjang jawabannya. Dari {n} pasang, buat "
        f"dengan komposisi TEPAT seperti ini:\n"
        f"- {n_short} pasang: pertanyaan ringan/faktual, dijawab 1 kalimat pendek (di bawah 20 kata). "
        "Contoh: tanya ya/tidak, definisi singkat, minta rekomendasi cepat.\n"
        f"- {n_mid} pasang: pertanyaan sedang, dijawab 2-3 kalimat (sekitar 30-50 kata).\n"
        f"- {n_long} pasang: pertanyaan KOMPLEKS yang memang butuh penjelasan mendalam "
        "(minta langkah-langkah rinci, minta perbandingan, minta penjelasan sebab-akibat), "
        "dijawab satu paragraf penuh 70-120 kata.\n"
        f"Jangan curang dengan membuat semua pertanyaan jadi sederhana -- {n_long} pertanyaan "
        "kompleks itu wajib ada. Panjang jawaban harus MASUK AKAL untuk pertanyaannya: "
        "pertanyaan sederhana jangan dijawab berpanjang-panjang, pertanyaan kompleks jangan "
        "dijawab satu kalimat.\n\n"
        "Balas HANYA dengan JSON array, tanpa markdown code fence atau teks lain di luar JSON. "
        f"Tepat {n} elemen, tiap elemen berupa object dengan dua field: "
        '"instruction" (string) dan "response" (string).'
        f"{accuracy_note}"
    )


def parse_json_array(raw):
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def call_freeform_batch(provider, host, model, domain, gaya, n, temperature, limiter):
    topics_pool = DOMAINS.get(domain, ["topik umum"])
    topics = random.sample(topics_pool, k=min(3, len(topics_pool)))
    system_prompt = steering_prompt(gaya)
    user_prompt = build_freeform_prompt(domain, topics, gaya, n)

    start = time.time()
    raw = call_with_retry(provider, host, model, system_prompt, user_prompt, temperature, limiter)
    latency = round(time.time() - start, 2)

    parsed = parse_json_array(raw)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("Respons bukan JSON array berisi pasangan instruksi-jawaban.")

    pairs = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        instruction = str(item.get("instruction", "")).strip()
        response = str(item.get("response", "")).strip()
        if instruction and response:
            pairs.append((instruction, response))
    if not pairs:
        raise ValueError("Tidak ada pasangan instruksi-jawaban yang valid di respons.")
    return pairs, system_prompt, latency


def make_row(instruction, response, domain, gaya, model, temperature, latency):
    messages = []
    system_prompt = pick_system_prompt(gaya)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": instruction})
    messages.append({"role": "assistant", "content": response})

    return {
        "id": _hash_id(instruction),
        "messages": messages,
        "domain": domain,
        "gaya": gaya,
        "source": "grok-freeform",
        "meta": {
            "model": model,
            "temperature": temperature,
            "latency_s": latency,
        },
    }


def process_batch(task, provider, host, model, temperature, limiter, seen_ids, seen_lock):
    domain, gaya, n = task["domain"], task["gaya"], task["n"]
    try:
        pairs, _, latency = call_freeform_batch(provider, host, model, domain, gaya, n,
                                                  temperature, limiter)
    except Exception as e:  # noqa: BLE001
        return [], [{"domain": domain, "gaya": gaya, "n": n, "error": str(e)}], 0

    results = []
    n_dup = 0
    for instruction, response in pairs:
        row = make_row(instruction, response, domain, gaya, model, temperature, latency)
        with seen_lock:
            if row["id"] in seen_ids:
                n_dup += 1
                continue
            seen_ids.add(row["id"])
        results.append(row)
    return results, [], n_dup


def build_tasks(total, batch_size, domains=None):
    """Buat daftar task (domain, gaya, n) sampai total pasangan yang diminta terpenuhi,
    dirotasi merata lintas domain & gaya biar variatif."""
    domain_list = domains if domains else list(DOMAINS.keys()) + list(CHITCHAT_DOMAINS.keys())
    tasks = []
    remaining = total
    i = 0
    while remaining > 0:
        domain = domain_list[i % len(domain_list)]
        gaya = GAYA_LIST[i % len(GAYA_LIST)]
        n = min(batch_size, remaining)
        tasks.append({"domain": domain, "gaya": gaya, "n": n})
        remaining -= n
        i += 1
    return tasks


def ask(prompt, default):
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else str(default)


def ask_int(prompt, default):
    while True:
        raw = ask(prompt, default)
        try:
            return int(raw)
        except ValueError:
            print("  -> masukkan angka bulat yang valid.")


def ask_float(prompt, default):
    while True:
        raw = ask(prompt, default)
        try:
            return float(raw)
        except ValueError:
            print("  -> masukkan angka yang valid.")


def run_setup_wizard(args):
    print("Oke, sebelum mulai, aku mau konfirmasi dulu beberapa hal ya.")
    print("Grok yang bakal bikin instruksi DAN jawabannya sekaligus, dari nol -- "
          f"nggak perlu file seed. Domain yang dirotasi: {', '.join(DOMAINS.keys())}.\n")

    args.total = ask_int("Mau digenerate berapa banyak pasangan instruksi+jawaban", args.total)
    args.batch_size = ask_int("Berapa pasang sekaligus per request (biar hemat kuota request)",
                               args.batch_size)
    args.rpm = ask_int("Batas request per menit biar ga kena rate limit (0 = ga usah dibatasi)",
                        args.rpm)
    args.temperature = ask_float("Temperature-nya berapa (makin tinggi makin variatif/kreatif)",
                                  args.temperature)
    args.model = ask("Model apa yang dipakai", args.model)

    print("\nSip, siap jalan.\n")
    return args


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset instruksi+jawaban dari nol pakai Grok (xAI) atau Ollama.")
    parser.add_argument("--out", type=str, default="data/generated.jsonl")
    parser.add_argument("--provider", type=str, choices=["xai", "ollama"], default="xai")
    parser.add_argument("--model", type=str, default="grok-4-fast-non-reasoning",
                         help="Model xAI. Default varian non-reasoning yang murah & cepat "
                              "($0.20/1M input, $0.50/1M output di bawah 128k token) -- cocok buat "
                              "generate volume besar. Kalau butuh kualitas lebih tinggi (mis. buat "
                              "held-out eval set), pakai model reasoning kayak grok-4.6.")
    parser.add_argument("--host", type=str, default="http://localhost:11434",
                         help="Base URL Ollama (hanya dipakai jika --provider ollama).")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--total", type=int, default=200,
                         help="Total pasangan instruksi+jawaban yang mau digenerate.")
    parser.add_argument("--batch-size", type=int, default=5,
                         help="Berapa pasang instruksi+jawaban diminta sekaligus per request API.")
    parser.add_argument("--workers", type=int, default=2,
                         help="Jumlah request paralel. Mulai kecil (1-2) dulu, naikkan sesuai kebutuhan.")
    parser.add_argument("--rpm", type=int, default=20,
                         help="Batas maksimum request per menit (rate limit), berlaku global lintas worker. "
                              "0 = tanpa batas.")
    parser.add_argument("--domains", type=str, default=None,
                         help="Batasi ke domain tertentu, pisahkan koma (default: semua domain di seed_bank).")
    parser.add_argument("--interactive", action="store_true",
                         help="Mode manual: tanya-jawab setup dulu, lalu minta konfirmasi Enter "
                              "sebelum mengirim tiap batch. Berjalan satu-satu (--workers diabaikan).")
    args = parser.parse_args()

    load_dotenv()

    if args.interactive:
        args = run_setup_wizard(args)

    domains = [d.strip() for d in args.domains.split(",")] if args.domains else None
    if domains:
        unknown = [d for d in domains if d not in DOMAINS and d not in CHITCHAT_DOMAINS]
        if unknown:
            valid = list(DOMAINS.keys()) + list(CHITCHAT_DOMAINS.keys())
            print(f"Domain tidak dikenal: {unknown}. Pilihan valid: {valid}", file=sys.stderr)
            sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    err_path = out_path.with_suffix(out_path.suffix + ".errors.jsonl")

    seen_ids = load_done_ids(args.out)
    seen_lock = threading.Lock()
    print(f"Sudah ada {len(seen_ids)} pasangan di {args.out} (akan otomatis di-skip kalau muncul lagi).")

    tasks = build_tasks(args.total, max(args.batch_size, 1), domains)

    n_done = 0
    n_err = 0
    n_dup = 0
    t0 = time.time()
    limiter = RateLimiter(args.rpm)

    target = "xAI API" if args.provider == "xai" else args.host
    rpm_desc = f"{args.rpm}/menit" if args.rpm else "tanpa batas"

    with out_path.open("a", encoding="utf-8") as out_f, err_path.open("a", encoding="utf-8") as err_f:

        def write_task_result(task, results, errors, dup):
            nonlocal n_done, n_err, n_dup
            for r in results:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            for e in errors:
                err_f.write(json.dumps(e, ensure_ascii=False) + "\n")
            out_f.flush()
            err_f.flush()

            n_done += len(results)
            n_err += len(errors)
            n_dup += dup
            total_processed = n_done + n_err
            elapsed = time.time() - t0
            rate = total_processed / elapsed if elapsed > 0 else 0
            status = "OK" if not errors else "ERR"
            dup_note = f" dup={dup}" if dup else ""
            print(f"[{total_processed}/{len(tasks)} task, {n_done}/{args.total} pasangan] {status} "
                  f"domain={task['domain']} gaya={task['gaya']} +{len(results)}{dup_note} "
                  f"({rate:.2f} it/s, {elapsed:.0f}s elapsed)", flush=True)

        if args.interactive:
            print(f"Oke, totalnya ada {len(tasks)} request yang bakal dikirim ke {target} "
                  f"pakai model {args.model}, rate limit {rpm_desc}, target {args.total} pasangan.")
            print("Tinggal tekan Enter tiap mau lanjut kirim, ketik 's' kalau mau skip, "
                  "atau 'q' kalau mau berhenti aja.\n")

            for idx, task in enumerate(tasks, start=1):
                print(f"--- Request {idx}/{len(tasks)} | domain={task['domain']} | "
                      f"gaya={task['gaya']} | minta {task['n']} pasang ---")
                answer = input("Lanjut kirim request ini? (Enter=lanjut, s=skip, q=berhenti): ").strip().lower()
                if answer == "q":
                    print("Oke, dihentikan di sini.")
                    break
                if answer == "s":
                    print("Skip request ini.\n")
                    continue

                results, errors, dup = process_batch(task, args.provider, args.host, args.model,
                                                       args.temperature, limiter, seen_ids, seen_lock)
                write_task_result(task, results, errors, dup)
                print()
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(process_batch, task, args.provider, args.host, args.model,
                                     args.temperature, limiter, seen_ids, seen_lock): task
                    for task in tasks
                }

                print(f"Mengirim {len(tasks)} request (target {args.total} pasangan, "
                      f"{args.batch_size} pasang/request) ke {target} "
                      f"(provider={args.provider}, model={args.model}, workers={args.workers}, "
                      f"rate limit={rpm_desc})...", flush=True)

                for fut in as_completed(futures):
                    task = futures[fut]
                    results, errors, dup = fut.result()
                    write_task_result(task, results, errors, dup)

    print(f"\nSelesai. Total sukses: {n_done}, duplikat di-skip: {n_dup}, error: {n_err}.")
    print(f"Output: {out_path}")
    if n_err:
        print(f"Error log: {err_path} (jalankan ulang script ini untuk nambah lagi kalau hasilnya "
              f"masih kurang dari target).")


if __name__ == "__main__":
    main()
