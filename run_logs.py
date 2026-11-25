import os
import statistics
import time
import unicodedata

from BPE_AIO import EXTRA_TOKENS, load_words, save_model, train_bpe, write_log
from encoding import BPETokenizer, load_test_text
from Syllable import eval_bpe

K_VALUES = (500, 1000, 2000, 4000, 8000)
ALL_LEVELS = ("0", "1", "2", "3")
REPEATS = 3


def bench_encode(tok, text):
    """Đo trung vị 3 lần encode: Không Cache vs. LRU Cache (Cold Cache)."""
    runs = {False: [], True: []}
    for use_cache in (False, True):
        for _ in range(REPEATS):
            if use_cache:
                tok.clear_cache()
            t0 = time.perf_counter()
            tok.encode(text, use_cache=use_cache)
            runs[use_cache].append(round(time.perf_counter() - t0, 4))

    raw_med = round(statistics.median(runs[False]), 4)
    cache_med = round(statistics.median(runs[True]), 4)
    speedup = round(raw_med / cache_med, 2) if cache_med > 0 else 0.0
    info = tok._encode_cached.cache_info()
    total = info.hits + info.misses
    hit = round(info.hits / total * 100, 1) if total else 0.0
    return f"-> CACHE MEDIAN: NoCache={runs[False]} ({raw_med:.4f}s) | LRUCache={runs[True]} ({cache_med:.4f}s) | Speedup={speedup:.2f}x | Hit={hit}%"


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CORPUS_PATH = os.path.join(BASE_DIR, "data", "vi_wiki_80mb.txt")
    MODEL_DIR = os.path.join(BASE_DIR, "model")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    os.makedirs(LOGS_DIR, exist_ok=True)

    ans = input("Làm mới (ghi đè) log cũ trước khi chạy? (y/n, mặc định y): ").strip().lower()
    if ans != "n":
        for name in [f"bpe_level_{lv}.log" for lv in ALL_LEVELS] + ["encoding_eval.log"]:
            open(os.path.join(LOGS_DIR, name), "w", encoding="utf-8").close()

    word_counts, prep_t = load_words(CORPUS_PATH, min_word_freq=5, vn_only=True, max_mb=75)
    test_text, test_mb = load_test_text(CORPUS_PATH, skip_mb=75)
    norm_test = unicodedata.normalize("NFC", test_text)
    print(f"Đã nạp Train (75 MB, U'={len(word_counts):,}, {prep_t}s) & Test ({test_mb} MB)\n")

    for k in K_VALUES:
        print(f"==================== BẮT ĐẦU K = {k} ====================")
        for lv in ALL_LEVELS:
            log_path = os.path.join(LOGS_DIR, f"bpe_level_{lv}.log")
            model_path = os.path.join(MODEL_DIR, f"bpe_level_{lv}.json")
            size_kb = 0.0

            for r in range(1, REPEATS + 1):
                res = train_bpe(
                    word_counts, k, level=lv,
                    min_pair_freq=2, log_interval=0, extra_tokens=EXTRA_TOKENS
                )
                if r == 1:
                    save_model(model_path, res, preprocess_time=prep_t)
                    size_kb = round(os.path.getsize(model_path) / 1024, 1)

                total_t = round(prep_t + res["time"], 3)
                line = (
                    f"-> XONG Mức {lv}: K={k}, Vocab={len(res['vocab'])}, "
                    f"Tiền xử lý={prep_t}s, Train={res['time']}s, Tổng={total_t}s, File={size_kb} KB"
                )
                write_log(line, log_path)
                print(f"  [K={k:<4} | Mức {lv} | Lần {r}/{REPEATS}] Train={res['time']:.3f}s")

        tok = BPETokenizer.from_json(os.path.join(MODEL_DIR, "bpe_level_3.json"))
        S, N, F, unk_pct = eval_bpe(tok, norm_test)
        enc_line = f"-> ENCODE EVAL: |V|={len(tok.vocab)} | S={S} | N={N} | F={F:.4f} | UNK={unk_pct:.4f}%"
        if k == K_VALUES[-1]:
            enc_line += "\n" + bench_encode(tok, norm_test)
        write_log(enc_line, os.path.join(LOGS_DIR, "encoding_eval.log"))
        print(f"  {enc_line}\n")