import csv
import os
import unicodedata
from collections import Counter

from BPE_AIO import EXTRA_TOKENS, load_words, train_bpe
from encoding import BPETokenizer, load_test_text

VN_VOWELS = set("aăâeêioôơuưyáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
VALID_ENDINGS = ("c", "ch", "m", "n", "ng", "nh", "p", "t")
BAD_CLUSTERS = ("oo", "ee", "ea", "ou", "bl", "br", "cl", "cr", "dr", "pl", "pr", "st", "sp", "sk")


def tone_ok(w):
    if w.endswith(("c", "ch", "p", "t")):
        d = unicodedata.normalize("NFD", w)
        return "\u0301" in d or "\u0323" in d
    return True


def is_vn_syllable(word):
    w = unicodedata.normalize("NFC", word).lower()
    if not (1 <= len(w) <= 7) or any(c in w for c in "fjwz"):
        return False
    if not any(c in VN_VOWELS for c in w):
        return False
    if any(p in w for p in BAD_CLUSTERS) and not w.endswith("oong"):
        return False
    return ((w[-1] in VN_VOWELS) or w.endswith(VALID_ENDINGS)) and tone_ok(w)


def eval_bpe(tok, test_text):
    S = len(tok._rx.findall(test_text))
    ids = tok.encode_ids(test_text, use_cache=True)
    N = len(ids)
    unk_pct = round(ids.count(tok.unk_id) / N * 100, 4) if N else 0.0
    return S, N, round(N / S, 4) if S else 0.0, unk_pct


def eval_syllable_f1(train_counts, words):
    vocab = set(train_counts.keys()) | set(EXTRA_TOKENS)
    S = len(words)
    unk = sum(1 for w in words if w not in vocab)
    return len(vocab) + 1, S, S, 1.0, round(unk / S * 100, 4) if S else 0.0


def unk_theo_tu(words, vocab, tach_ky_tu):
    """Trả về (% từ <unk> trên mọi từ S, % từ <unk> chỉ tính từ thuần chữ cái)."""
    total = len(words)
    unk = alpha_total = alpha_unk = 0
    for w in words:
        is_unk = any(c not in vocab for c in w) if tach_ky_tu else (w not in vocab)
        if is_unk:
            unk += 1
        if w.isalpha():
            alpha_total += 1
            if is_unk:
                alpha_unk += 1
    u_all = round(unk / total * 100, 4) if total else 0.0
    u_alpha = round(alpha_unk / alpha_total * 100, 4) if alpha_total else 0.0
    return u_all, u_alpha


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CORPUS_PATH = os.path.join(BASE_DIR, "data", "vi_wiki_80mb.txt")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    raw_counts, _ = load_words(CORPUS_PATH, min_word_freq=5, vn_only=True, max_mb=75)
    test_text = unicodedata.normalize("NFC", load_test_text(CORPUS_PATH, skip_mb=75)[0])
    test_words = BPETokenizer.DEFAULT_RX.findall(test_text)
    clean_counts = Counter({w: c for w, c in raw_counts.items() if is_vn_syllable(w)})

    res_raw = train_bpe(raw_counts, 8000, level="3", min_pair_freq=2, log_interval=0, extra_tokens=EXTRA_TOKENS)
    res_clean = train_bpe(clean_counts, 8000, level="3", min_pair_freq=2, log_interval=0, extra_tokens=EXTRA_TOKENS)

    tok_raw = BPETokenizer(res_raw["vocab"], res_raw["merges"])
    tok_clean = BPETokenizer(res_clean["vocab"], res_clean["merges"])
    vocab_raw = set(raw_counts) | set(EXTRA_TOKENS)
    vocab_clean = set(clean_counts) | set(EXTRA_TOKENS)

    comp_rows = [
        ("BPE K=8000 (con tu tieng Anh)", len(res_raw["vocab"]), *eval_bpe(tok_raw, test_text), *unk_theo_tu(test_words, tok_raw.vocab, True)),
        ("BPE K=8000 (da loc tieng Anh)", len(res_clean["vocab"]), *eval_bpe(tok_clean, test_text), *unk_theo_tu(test_words, tok_clean.vocab, True)),
        ("Tach nguyen am tiet (F=1, goc)", *eval_syllable_f1(raw_counts, test_words), *unk_theo_tu(test_words, vocab_raw, False)),
        ("Tach nguyen am tiet (F=1, loc)", *eval_syllable_f1(clean_counts, test_words), *unk_theo_tu(test_words, vocab_clean, False)),
    ]

    with open(os.path.join(RESULTS_DIR, "comparison.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Phuong_phap", "Vocab_size", "Test_S", "Test_N", "F", "UNK_rate_pct", "UNK_tu_pct", "UNK_tu_chu_cai_pct"])
        w.writerows(comp_rows)

    with open(os.path.join(LOGS_DIR, "summary.log"), "w", encoding="utf-8") as f:
        f.write("=== MỤC 1: KHẢO SÁT SỐ BƯỚC GỘP K ===\n")
        with open(os.path.join(RESULTS_DIR, "k_survey.csv"), encoding="utf-8-sig") as kf:
            for r in csv.DictReader(kf):
                f.write(f"  K={r['K']} | |V|={r['Vocab_size']} | Train={r['Train_time_s']}s | N={r['Test_N']} | F={r['F']} | <unk>={r['UNK_rate_pct']}%\n")

        f.write("\n=== MỤC 2: SO SÁNH LỌC TIẾNG ANH, ĐỐI CHIẾU F = 1 & TỶ LỆ <unk> ===\n")
        f.write(f"  U' trước lọc = {len(raw_counts):,} | U' sau lọc tiếng Anh = {len(clean_counts):,}\n")
        for r in comp_rows:
            f.write(
                f"  {r[0]}: |V|={r[1]:,} | S={r[2]:,} | N={r[3]:,} | F={r[4]:.4f} | "
                f"<unk>/token={r[5]:.4f}% | <unk>/từ={r[6]:.4f}% | <unk>/từ chữ cái={r[7]:.4f}%\n"
            )
        f.write("  (<unk>/token: BPE chia cho N, tách âm tiết chia cho S; <unk>/từ và <unk>/từ chữ cái cùng chia cho số từ nên so sánh trực tiếp được.)\n")

        f.write("\n=== MỤC 3: TRUNG VỊ 3 LẦN ĐO THỜI GIAN HUẤN LUYỆN (K = 8,000) ===\n")
        with open(os.path.join(RESULTS_DIR, "benchmark_median.csv"), encoding="utf-8-sig") as mf:
            for r in csv.DictReader(mf):
                f.write(f"  Mức {r['Muc']} ({r['Thuat_toan']}): [{r['Lan_1_s']}s, {r['Lan_2_s']}s, {r['Lan_3_s']}s] -> Trung vị = {r['Trung_vi_s']}s\n")

        enc_log = os.path.join(LOGS_DIR, "encoding_eval.log")
        if os.path.exists(enc_log):
            with open(enc_log, encoding="utf-8") as ef:
                cache_lines = [ln.strip() for ln in ef if ln.startswith("-> CACHE MEDIAN:")]
            if cache_lines:
                f.write(f"  {cache_lines[-1]}\n")

    print("Hoàn tất! Xem chi tiết trong 'logs/summary.log' và các tệp CSV trong thư mục 'results'.")