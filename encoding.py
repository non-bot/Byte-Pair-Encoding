import json
import os
import re
import time
import unicodedata
from functools import lru_cache


class BPETokenizer:
    """Mã hóa/giải mã BPE hỗ trợ bật/tắt LRU Cache."""

    DEFAULT_PATTERN = r"\w+|[^\w\s]"
    DEFAULT_RX = re.compile(DEFAULT_PATTERN, re.UNICODE)

    def __init__(self, vocab, merges, eow="</w>", pattern=DEFAULT_PATTERN, unk_token="<unk>", cache_size=65536):
        self.vocab = vocab if isinstance(vocab, dict) else {t: i for i, t in enumerate(vocab)}
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self.merge_ranks = {tuple(p): r for r, p in enumerate(merges)}
        self.eow = eow
        self.unk_token = unk_token
        self.unk_id = self.vocab.get(unk_token, 0)
        self._rx = self.DEFAULT_RX if pattern == self.DEFAULT_PATTERN else re.compile(pattern, re.UNICODE)
        self._rx_ws = re.compile(rf"{pattern}|\s+", re.UNICODE)
        self._encode_cached = lru_cache(maxsize=cache_size)(self._encode_word)

    @classmethod
    def from_json(cls, model_path, cache_size=65536):
        with open(model_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = data.get("config", {})
        return cls(
            vocab=data["vocab"],
            merges=data["merges"],
            eow=cfg.get("end_of_word", "</w>"),
            pattern=cfg.get("pre_tokenize_pattern", cls.DEFAULT_PATTERN),
            unk_token=cfg.get("unk_token", "<unk>"),
            cache_size=cache_size,
        )

    def _encode_word(self, word, use_unk=False):
        tokens = ([ch if ch in self.vocab else self.unk_token for ch in word] if use_unk else list(word)) + [self.eow]
        ranks, inf = self.merge_ranks, float("inf")

        while len(tokens) > 1:
            best_pair = min(
                ((tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)),
                key=lambda p: ranks.get(p, inf),
            )
            if best_pair not in ranks:
                break

            p0, p1 = best_pair
            merged = p0 + p1
            out, i, n = [], 0, len(tokens)
            while i < n:
                if i < n - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
                    out.append(merged)
                    i += 2
                else:
                    out.append(tokens[i])
                    i += 1
            tokens = out

        return tuple(tokens)

    def encode(self, text, use_cache=True, keep_whitespace=False, use_unk=False):
        norm = unicodedata.normalize("NFC", text)
        fn = self._encode_cached if use_cache else self._encode_word
        out = []
        if keep_whitespace:
            for part in self._rx_ws.findall(norm):
                out.append(part) if part.isspace() else out.extend(fn(part, use_unk))
        else:
            for word in self._rx.findall(norm):
                out.extend(fn(word, use_unk))
        return out

    def decode(self, tokens, has_whitespace=False):
        eow, n = self.eow, len(self.eow)
        sep = "" if has_whitespace else " "
        text = "".join((t[:-n] + sep) if t.endswith(eow) else t for t in tokens)
        return text if has_whitespace else text.strip()

    def encode_ids(self, text, use_cache=True):
        return [self.vocab.get(t, self.unk_id) for t in self.encode(text, use_cache=use_cache, use_unk=True)]

    def decode_ids(self, ids):
        return self.decode([self.id_to_token.get(i, self.unk_token) for i in ids])

    def clear_cache(self):
        self._encode_cached.cache_clear()


def compare_encode_speed(tokenizer, test_text, repeats=3):
    """Đo trung vị 3 lần tốc độ Encode: Không Cache vs. Có LRU Cache (Cold Cache)."""
    norm = unicodedata.normalize("NFC", test_text)
    S = len(tokenizer._rx.findall(norm))
    if S == 0:
        return None

    raw_runs, cached_runs = [], []
    for i in range(repeats):
        t0 = time.perf_counter()
        toks_raw = tokenizer.encode(norm, use_cache=False)
        raw_runs.append(round(time.perf_counter() - t0, 4))

        tokenizer.clear_cache()
        t0 = time.perf_counter()
        toks_cached = tokenizer.encode(norm, use_cache=True)
        cached_runs.append(round(time.perf_counter() - t0, 4))

        if i == 0:
            assert toks_raw == toks_cached, "Lệch kết quả giữa bản có và không có cache!"

    t_raw = round(statistics.median(raw_runs), 4)
    t_cached = round(statistics.median(cached_runs), 4)
    N, F = len(toks_cached), round(len(toks_cached) / S, 4)
    unk_count = sum(1 for t in toks_cached if t not in tokenizer.vocab)
    unk_pct = round(unk_count / N * 100, 4) if N else 0.0
    info = tokenizer._encode_cached.cache_info()
    hit_rate = round(info.hits / (info.hits + info.misses) * 100, 1) if (info.hits + info.misses) else 0.0
    speedup = round(t_raw / t_cached, 2) if t_cached > 0 else 0.0
    v_raw = round(S / t_raw) if t_raw > 0 else 0
    v_cached = round(S / t_cached) if t_cached > 0 else 0

    print(f"=== SO SÁNH ENCODE - TRUNG VỊ {repeats} LẦN (S={S:,} | N={N:,} | F={F:.4f} | |V|={len(tokenizer.vocab):,}) ===")
    print(f"  Không Cache : {raw_runs} -> Trung vị = {t_raw:.4f}s | {v_raw:,} âm tiết/s")
    print(f"  LRU Cache   : {cached_runs} -> Trung vị = {t_cached:.4f}s | {v_cached:,} âm tiết/s | Tăng tốc: {speedup:.2f}x (hit={hit_rate}%)")
    print(f"  Tỷ lệ <unk> : {unk_count:,}/{N:,} ({unk_pct:.4f}%)")
    return (
        f"-> ENCODE EVAL: |V|={len(tokenizer.vocab)} | S={S} | N={N} | F={F:.4f} | UNK={unk_pct:.4f}%\n"
        f"-> CACHE MEDIAN: NoCache={t_raw:.4f}s | LRUCache={t_cached:.4f}s | Speedup={speedup:.2f}x | Hit={hit_rate}%"
    )


def load_test_text(file_path, skip_mb=75):
    """Bỏ qua phần train (skip_mb) theo ranh giới dòng bằng seek, đọc phần còn lại."""
    skip_bytes = int(skip_mb * 1024 * 1024)
    with open(file_path, "rb") as f:
        if skip_bytes > 0:
            f.seek(skip_bytes - 1)
            f.readline()
        remaining = f.read()
    return remaining.decode("utf-8"), round(len(remaining) / (1024 * 1024), 2)


if __name__ == "__main__":
    skip_mb = 75
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "model", "bpe_level_3.json")
    CORPUS_PATH = os.path.join(BASE_DIR, "data", "vi_wiki_80mb.txt")

    if not os.path.exists(MODEL_PATH):
        print(f"[LỖI] Không tìm thấy mô hình '{MODEL_PATH}'. Hãy chạy BPE_AIO.py (mức 3) trước.")
    elif not os.path.exists(CORPUS_PATH):
        print(f"[LỖI] Không tìm thấy '{CORPUS_PATH}'.")
    else:
        tok = BPETokenizer.from_json(MODEL_PATH)
        test_text, test_mb = load_test_text(CORPUS_PATH, skip_mb)
        if test_text.strip():
            print(f"Đã nạp phần kiểm thử còn lại: {test_mb} MB từ '{CORPUS_PATH}'")
            log_block = compare_encode_speed(tok, test_text)
            if log_block:
                log_path = os.path.join(BASE_DIR, "logs", "encoding_eval.log")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(log_block + "\n")
        else:
            print(f"[CẢNH BÁO] Tệp '{CORPUS_PATH}' chưa vượt quá {skip_mb} MB, không có phần kiểm thử.")