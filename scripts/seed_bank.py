"""
seed_bank.py
============
Bank seed prompt Bahasa Indonesia untuk generate dataset distillation
(teacher: Gemma 4 E4B via Ollama -> student: Qwen 1.5B).

Isinya dua sumber:
1. SEED_MANUAL  - contoh yang ditulis manual, kualitas tinggi, untuk
   variasi yang sulit dihasilkan otomatis (roleplay, curhat, dsb).
2. generate_combinatorial() - menggabungkan TOPICS x DOMAINS x GAYA x
   INSTRUCTION_TEMPLATES untuk menghasilkan ribuan instruksi unik
   secara otomatis.

Setiap seed adalah dict:
    {
        "id": str,          # unique id
        "instruction": str, # instruksi/prompt untuk teacher
        "domain": str,
        "gaya": str,        # "formal" | "santai"
        "source": str,      # "manual" | "combinatorial"
    }

Jalankan langsung untuk menulis data/seed_prompts.jsonl:
    python scripts/seed_bank.py --n 3000 --out data/seed_prompts.jsonl
"""

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# 1. Domain & topik. Silakan tambah sesuai kebutuhan chatbot umum.
# ---------------------------------------------------------------------------
DOMAINS = {
    "kesehatan": [
        "pola tidur yang baik", "manfaat olahraga rutin", "cara mengatasi flu ringan",
        "pentingnya sarapan", "tips menjaga kesehatan mata di depan layar",
        "cara mengelola stres sehari-hari", "kebiasaan minum air putih",
    ],
    "teknologi": [
        "cara kerja AI generatif", "perbedaan HDD dan SSD", "tips memilih laptop untuk kuliah",
        "apa itu cloud computing", "cara mengamankan akun media sosial",
        "perkembangan mobil listrik", "apa itu machine learning",
    ],
    "pendidikan": [
        "cara belajar efektif sebelum ujian", "tips menulis esai yang baik",
        "manfaat membaca buku setiap hari", "cara mengatur waktu belajar dan bermain",
        "perbedaan kuliah S1 dan D3", "tips memilih jurusan kuliah",
    ],
    "keuangan": [
        "cara membuat anggaran bulanan", "tips menabung untuk dana darurat",
        "perbedaan investasi saham dan reksadana", "cara menghindari utang konsumtif",
        "dasar-dasar literasi keuangan", "tips mengelola gaji bulanan",
    ],
    "kuliner": [
        "resep nasi goreng sederhana", "cara membuat kopi tubruk yang enak",
        "makanan khas Yogyakarta", "tips memasak ayam agar empuk",
        "cara membuat sambal terasi", "ide menu sarapan sehat",
    ],
    "traveling": [
        "destinasi wisata alam di Jawa Barat", "tips packing untuk perjalanan jauh",
        "cara menyusun itinerary liburan hemat", "wisata kuliner di Bandung",
        "tips naik pesawat pertama kali", "rekomendasi tempat camping dekat Jakarta",
    ],
    "hukum": [
        "hak konsumen saat belanja online", "cara membuat surat perjanjian sederhana",
        "prosedur membuat KTP baru", "apa itu hak cipta",
        "dasar hukum perjanjian kerja",
    ],
    "sains": [
        "kenapa langit berwarna biru", "cara kerja fotosintesis",
        "apa itu pemanasan global", "perbedaan cuaca dan iklim",
        "bagaimana gempa bumi terjadi",
    ],
    "rumah_tangga": [
        "cara membersihkan noda membandel di baju", "tips merawat tanaman hias di rumah",
        "cara menghemat listrik di rumah", "tips menata kamar kos yang sempit",
        "cara mengusir semut secara alami",
    ],
    "karir": [
        "tips membuat CV yang menarik", "cara menjawab pertanyaan wawancara kerja",
        "tips mengelola waktu kerja dari rumah", "cara membangun personal branding di LinkedIn",
        "tips negosiasi gaji",
    ],
    "olahraga": [
        "manfaat lari pagi", "tips pemanasan sebelum olahraga",
        "perbedaan yoga dan pilates", "cara memulai kebiasaan olahraga untuk pemula",
    ],
    "hiburan": [
        "rekomendasi film Indonesia yang bagus", "tips membuat playlist musik untuk fokus",
        "cara memulai hobi menulis", "rekomendasi buku fiksi Indonesia",
    ],
    "coding": [
        "cara membuat fungsi sederhana di Python", "apa itu variabel dalam pemrograman",
        "perbedaan array dan list", "cara membaca error di kode program",
        "apa itu API secara sederhana",
    ],
    "percakapan_umum": [
        "cerita singkat tentang persahabatan", "pantun lucu tentang kerja",
        "cara menyapa teman lama yang jarang ketemu", "ide ucapan selamat ulang tahun",
        "cara meminta maaf yang tulus",
    ],
}

# ---------------------------------------------------------------------------
# 2. Template instruksi. {topic} akan diganti dengan salah satu topik domain.
# ---------------------------------------------------------------------------
INSTRUCTION_TEMPLATES = [
    "Jelaskan tentang {topic}.",
    "Apa yang dimaksud dengan {topic}? Jelaskan dengan bahasa sederhana.",
    "Berikan tips praktis soal {topic}.",
    "Aku bingung soal {topic}, bisa bantu jelaskan?",
    "Buatkan ringkasan singkat mengenai {topic}.",
    "Kenapa {topic} itu penting? Jelaskan alasannya.",
    "Berikan 3 poin utama tentang {topic}.",
    "Ceritakan sedikit tentang {topic} seolah aku baru pertama kali dengar.",
    "Apa saja hal yang perlu diperhatikan terkait {topic}?",
    "Bagaimana cara memulai jika aku ingin belajar soal {topic}?",
]

# Gaya bahasa -> instruksi tambahan untuk system prompt teacher
GAYA_STYLE_HINT = {
    "formal": "Gunakan Bahasa Indonesia formal dan baku, seperti menulis artikel edukatif.",
    "santai": "Gunakan Bahasa Indonesia santai sehari-hari, seperti mengobrol dengan teman, boleh pakai kata seperti 'kamu', 'nih', 'ya'.",
}

# ---------------------------------------------------------------------------
# 3. Seed manual — variasi yang lebih sulit digenerate template (roleplay,
#    permintaan kreatif, multi-turn style, edge case bahasa campur, dst).
# ---------------------------------------------------------------------------
SEED_MANUAL = [
    {"instruction": "Halo, kamu siapa? Bisa bantu apa saja?", "domain": "percakapan_umum", "gaya": "santai"},
    {"instruction": "Selamat pagi. Mohon bantuannya menjelaskan prosedur pengajuan cuti kerja secara formal.", "domain": "karir", "gaya": "formal"},
    {"instruction": "Buatkan cerita pendek 5 kalimat tentang anak kucing yang tersesat, dengan akhir bahagia.", "domain": "percakapan_umum", "gaya": "santai"},
    {"instruction": "Tolong buatkan draf email formal untuk meminta izin tidak masuk kerja karena sakit.", "domain": "karir", "gaya": "formal"},
    {"instruction": "Eh, menurutmu enakan kerja kantoran atau freelance? Jelasin plus minusnya dong.", "domain": "karir", "gaya": "santai"},
    {"instruction": "Apa perbedaan antara 'di' sebagai kata depan dan 'di' sebagai awalan dalam Bahasa Indonesia?", "domain": "pendidikan", "gaya": "formal"},
    {"instruction": "Gimana caranya bikin anak kecil suka makan sayur? Ada tips nggak?", "domain": "kesehatan", "gaya": "santai"},
    {"instruction": "Jelaskan secara ringkas apa itu inflasi dan dampaknya bagi masyarakat.", "domain": "keuangan", "gaya": "formal"},
    {"instruction": "Woy, rekomendasiin dong tempat makan enak yang murah meriah buat anak kos.", "domain": "kuliner", "gaya": "santai"},
    {"instruction": "Tolong terjemahkan kalimat berikut ke Bahasa Inggris: 'Saya sangat senang bisa belajar hal baru hari ini.'", "domain": "pendidikan", "gaya": "formal"},
    {"instruction": "Kalau aku capek banget abis kerja seharian, enaknya ngapain buat refreshing ringan?", "domain": "kesehatan", "gaya": "santai"},
    {"instruction": "Susun agenda rapat formal untuk pembahasan evaluasi kinerja tim triwulan.", "domain": "karir", "gaya": "formal"},
    {"instruction": "Menurutmu kenapa banyak orang susah bangun pagi walau sudah pasang alarm?", "domain": "kesehatan", "gaya": "santai"},
    {"instruction": "Apa itu bunga majemuk dalam investasi? Jelaskan dengan contoh sederhana.", "domain": "keuangan", "gaya": "formal"},
    {"instruction": "Buatin caption Instagram yang santai buat foto liburan ke pantai.", "domain": "hiburan", "gaya": "santai"},
    {"instruction": "Tolong jelaskan langkah-langkah debugging kode Python yang error tanpa pesan yang jelas.", "domain": "coding", "gaya": "formal"},
    {"instruction": "Kenapa ya kadang aku susah fokus pas kerja dari rumah? Ada solusinya nggak?", "domain": "karir", "gaya": "santai"},
    {"instruction": "Jelaskan konsep dasar 'supply and demand' dalam ekonomi secara formal.", "domain": "keuangan", "gaya": "formal"},
    {"instruction": "Aku pengen mulai olahraga tapi males banget, gimana caranya biar konsisten?", "domain": "olahraga", "gaya": "santai"},
    {"instruction": "Tuliskan pantun empat baris bertema semangat kerja.", "domain": "percakapan_umum", "gaya": "santai"},
]


def _hash_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def generate_combinatorial(n: int):
    """Hasilkan n seed unik dari kombinasi domain x topic x template x gaya."""
    combos = []
    for domain, topics in DOMAINS.items():
        for topic in topics:
            for template in INSTRUCTION_TEMPLATES:
                for gaya in GAYA_STYLE_HINT:
                    combos.append((domain, topic, template, gaya))
    random.shuffle(combos)

    seeds = []
    seen = set()
    for domain, topic, template, gaya in combos:
        instruction = template.format(topic=topic)
        key = (instruction, gaya)
        if key in seen:
            continue
        seen.add(key)
        seeds.append({
            "id": _hash_id(instruction + gaya),
            "instruction": instruction,
            "domain": domain,
            "gaya": gaya,
            "source": "combinatorial",
        })
        if len(seeds) >= n:
            break
    return seeds


def build_seed_file(n_combinatorial: int, out_path: str):
    manual = []
    for item in SEED_MANUAL:
        manual.append({
            "id": _hash_id(item["instruction"] + item["gaya"]),
            "instruction": item["instruction"],
            "domain": item["domain"],
            "gaya": item["gaya"],
            "source": "manual",
        })

    combinatorial = generate_combinatorial(n_combinatorial)

    all_seeds = manual + combinatorial
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in all_seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Total seeds ditulis: {len(all_seeds)} "
          f"({len(manual)} manual + {len(combinatorial)} combinatorial) -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate seed prompts Bahasa Indonesia.")
    parser.add_argument("--n", type=int, default=3000,
                         help="Jumlah seed combinatorial yang ingin dihasilkan (di luar manual).")
    parser.add_argument("--out", type=str, default="data/seed_prompts.jsonl",
                         help="Path output JSONL.")
    args = parser.parse_args()
    build_seed_file(args.n, args.out)
