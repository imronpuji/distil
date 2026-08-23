"""
filter_dataset.py
==================
Filter & bersihkan dataset hasil generate_dataset.py (langkah 2 di
rencana proyek): buang output aneh/pendek/campur bahasa/duplikat,
sebelum dipakai untuk SFT Qwen 1.5B.

Tidak pakai library eksternal (langdetect dll) supaya ringan dijalankan
di instance manapun — deteksi campur-bahasa pakai heuristik daftar kata
umum Bahasa Inggris vs Indonesia. Untuk dataset besar/produksi,
pertimbangkan tambah `langdetect` atau `fasttext lid.176` untuk deteksi
bahasa yang lebih akurat.

Pemakaian:
    python scripts/filter_dataset.py \\
        --in data/generated.jsonl \\
        --out data/generated_clean.jsonl \\
        --rejected data/generated_rejected.jsonl \\
        --min-words 5 --max-words 400
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# Kata fungsi bahasa Inggris yang umum & jarang muncul wajar di kalimat
# Indonesia (dipakai sebagai sinyal campur bahasa, bukan named entity/istilah teknis).
EN_STOPWORDS = {
    "the", "is", "are", "and", "of", "to", "in", "for", "with", "on", "that",
    "this", "it", "as", "was", "were", "be", "have", "has", "you", "your",
    "an", "or", "if", "not", "but", "can", "will", "would", "should", "i",
    "we", "they", "he", "she", "at", "by", "from", "so", "which", "what",
    "how", "when", "where", "why", "sorry", "however", "therefore",
}

ID_COMMON = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan",
    "adalah", "akan", "pada", "juga", "atau", "tidak", "bisa", "saya",
    "kamu", "kita", "mereka", "dia", "karena", "jadi", "sebagai", "kalau",
}

REFUSAL_PATTERNS = [
    r"\bsebagai model bahasa\b",
    r"\bas an ai\b",
    r"\bi cannot\b",
    r"\bi'm sorry, but\b",
    r"\bmaaf, saya tidak (dapat|bisa) membantu\b",
]

# Teacher (Grok) kadang menjawab "Saya dibuat oleh OpenAI" untuk pertanyaan identitas.
# Itu atribusi SALAH -- model yang kita latih adalah Qwen. Kalau lolos ke training data,
# klaim keliru itu ikut ke-hardcode ke model hasil fine-tune, jadi harus dibuang.
# Pola sengaja disempitkan ke KLAIM IDENTITAS, bukan sekadar penyebutan nama vendor,
# supaya jawaban teknis yang wajar (mis. "pakai OpenAI API") tidak ikut kebuang.
_VENDORS = r"(openai|chatgpt|gpt-?[0-9]|google|gemini|bard|anthropic|claude|meta|llama|xai|grok|microsoft|copilot)"
_MADE = r"(dibuat|diciptakan|dikembangkan|dilatih|dibangun|diproduksi)"
_QUALIFIER = r"(oleh\s+|sama\s+)?(perusahaan\s+|sistem\s+|tim\s+|teknologi\s+)?"
WRONG_ATTRIBUTION_PATTERNS = [
    rf"\b{_MADE}\s+{_QUALIFIER}{_VENDORS}",
    rf"\b{_VENDORS}\s+(lah\s+)?yang\s+(membuat|mengembangkan|menciptakan|melatih|membangun)\s+(saya|aku)",
    rf"\bsaya\s+(adalah\s+|ini\s+)?{_VENDORS}\b",
    rf"\bnama\s+saya\s+(adalah\s+)?{_VENDORS}\b",
    rf"\b(saya|aku)\s+berasal\s+dari\s+{_QUALIFIER}{_VENDORS}",
    rf"\bpembuat\s+(saya|aku)\s+(adalah\s+)?{_VENDORS}",
]

ECHO_SIMILARITY_THRESHOLD = 0.9


def word_count(text):
    return len(text.split())


def mixed_language_ratio(text):
    """Rasio kata bahasa Inggris umum dibanding total kata. Tinggi = kemungkinan campur bahasa."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return 0.0
    en_hits = sum(1 for w in words if w in EN_STOPWORDS)
    return en_hits / len(words)


def has_repetition_loop(text, max_repeat=6):
    """Deteksi kata/frasa yang berulang berturut-turut (gejala generation loop)."""
    words = text.split()
    if len(words) < max_repeat:
        return False
    for i in range(len(words) - max_repeat):
        window = words[i:i + max_repeat]
        if len(set(window)) == 1:
            return True
    return False


def looks_like_echo(instruction, response):
    instr_norm = instruction.strip().lower()
    resp_norm = response.strip().lower()
    if not resp_norm:
        return True
    if resp_norm.startswith(instr_norm) and len(instr_norm) / max(len(resp_norm), 1) > ECHO_SIMILARITY_THRESHOLD:
        return True
    return False


def has_refusal(text):
    low = text.lower()
    return any(re.search(pat, low) for pat in REFUSAL_PATTERNS)


def has_wrong_attribution(text):
    low = text.lower()
    # Syarat: jawabannya harus berbicara tentang DIRINYA SENDIRI. Tanpa penjaga ini,
    # kalimat faktual yang wajar seperti "TensorFlow dikembangkan oleh Google" di domain
    # teknologi/coding ikut kebuang, padahal itu jawaban benar.
    if not re.search(r"\b(saya|aku)\b", low):
        return False
    return any(re.search(pat, low) for pat in WRONG_ATTRIBUTION_PATTERNS)


def get_response_text(row):
    for m in row.get("messages", []):
        if m.get("role") == "assistant":
            return m.get("content", "")
    return row.get("response", "")


def get_instruction_text(row):
    for m in row.get("messages", []):
        if m.get("role") == "user":
            return m.get("content", "")
    return row.get("instruction", "")


# Domain percakapan pendek: jawaban SEHARUSNYA pendek ("Siap!", "Halo, ada yang bisa
# dibantu?"), jadi ambang min-words biasa nggak berlaku -- kalau dipaksa, semua data
# chitchat kebuang dan model balik nggak bisa merespons sapaan.
CHITCHAT_DOMAINS = {"sapaan", "acknowledgment", "identitas", "basa_basi"}
CHITCHAT_MIN_WORDS = 1
CHITCHAT_MAX_WORDS = 40

# Domain CoT (v2, scripts/v2/generate_cot_dataset.py): jawaban = <think>...</think> +
# jawaban akhir, jauh lebih panjang dari jawaban biasa (median ~784 kata di sampel 604
# baris). Ambang 400 kata biasa bakal buang 70% data ini. Tapi tetap kasih plafon --
# ketemu outlier 18.798 kata (reasoning loop nyasar), itu genuinely harus dibuang.
REASONING_DOMAIN_PREFIX = "reasoning_"
REASONING_MIN_WORDS = 5
# Sengaja nggak dibatasi atas -- biarin reasoning sepanjang apapun lolos filter, training
# (--max-seq-length) yang natural motong kalau kepanjangan buat context window-nya.
REASONING_MAX_WORDS = float("inf")


def classify(row, min_words, max_words, max_mixed_ratio):
    instruction = get_instruction_text(row)
    response = get_response_text(row)

    if not response or not response.strip():
        return "empty_response"

    domain = row.get("domain") or ""
    is_chitchat = domain in CHITCHAT_DOMAINS
    is_reasoning = domain.startswith(REASONING_DOMAIN_PREFIX)
    if is_chitchat:
        lo, hi = CHITCHAT_MIN_WORDS, CHITCHAT_MAX_WORDS
    elif is_reasoning:
        lo, hi = REASONING_MIN_WORDS, REASONING_MAX_WORDS
    else:
        lo, hi = min_words, max_words

    wc = word_count(response)
    if wc < lo:
        return "too_short"
    if wc > hi:
        return "too_long"

    if looks_like_echo(instruction, response):
        return "echoes_instruction"

    if has_repetition_loop(response):
        return "repetition_loop"

    # coding & reasoning dikecualikan dari cek mixed_language: kode wajar mengandung kata
    # Inggris, dan reasoning trace kadang selip Inggris/notasi matematika (\frac, dll) yang
    # nggak seharusnya jadi alasan buang seluruh data.
    if domain != "coding" and not is_reasoning and mixed_language_ratio(response) > max_mixed_ratio:
        return "mixed_language"

    if has_refusal(response):
        return "refusal_or_meta"

    if has_wrong_attribution(response):
        return "wrong_attribution"

    return None  # lolos semua filter


def main():
    parser = argparse.ArgumentParser(description="Filter dataset hasil generate_dataset.py")
    parser.add_argument("--in", dest="inp", type=str, default="data/generated.jsonl")
    parser.add_argument("--out", type=str, default="data/generated_clean.jsonl")
    parser.add_argument("--rejected", type=str, default="data/generated_rejected.jsonl")
    parser.add_argument("--min-words", type=int, default=3,
                         help="Diturunkan dari 5 ke 3 karena generator sekarang sengaja "
                              "menghasilkan jawaban pendek untuk pertanyaan ringan -- ambang 5 "
                              "ikut membuang jawaban singkat yang justru kita butuhkan.")
    parser.add_argument("--max-words", type=int, default=400)
    parser.add_argument("--max-mixed-ratio", type=float, default=0.15,
                         help="Ambang rasio kata bahasa Inggris umum sebelum dianggap campur bahasa.")
    args = parser.parse_args()

    in_path = Path(args.inp)
    if not in_path.exists():
        raise SystemExit(f"File tidak ditemukan: {in_path}")

    seen_hashes = set()
    stats = Counter()
    total = 0

    out_path = Path(args.out)
    rej_path = Path(args.rejected)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout, \
         rej_path.open("w", encoding="utf-8") as frej:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)

            response = get_response_text(row)
            dup_key = response.strip().lower()
            if dup_key in seen_hashes:
                stats["duplicate"] += 1
                frej.write(json.dumps({**row, "reject_reason": "duplicate"}, ensure_ascii=False) + "\n")
                continue

            reason = classify(row, args.min_words, args.max_words, args.max_mixed_ratio)
            if reason:
                stats[reason] += 1
                frej.write(json.dumps({**row, "reject_reason": reason}, ensure_ascii=False) + "\n")
                continue

            seen_hashes.add(dup_key)
            stats["kept"] += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Total baris diproses : {total}")
    print(f"Lolos filter (kept)  : {stats['kept']} ({stats['kept'] / max(total, 1) * 100:.1f}%)")
    print("Rincian ditolak:")
    for reason, count in stats.items():
        if reason == "kept":
            continue
        print(f"  - {reason:20s}: {count}")
    print(f"\nOutput bersih : {out_path}")
    print(f"Output ditolak: {rej_path} (cek manual sebagian untuk kalibrasi ambang filter)")


if __name__ == "__main__":
    main()
