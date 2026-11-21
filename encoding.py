import json
import re
import time
import unicodedata
import os
from functools import lru_cache


class BPETokenizer:
    """Mã hóa/giải mã BPE hỗ trợ bật/tắt LRU Cache."""

    def __init__(self, vocab, merges, eow="</w>", pattern=r"\w+|[^\w\s]", unk_token="<unk>", cache_size=65536):
        self.vocab = vocab if isinstance(vocab, dict) else {t: i for i, t in enumerate(vocab)}
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self.merge_ranks = {tuple(p): r for r, p in enumerate(merges)}
        self.eow = eow
        self.unk_token = unk_token
        self.unk_id = self.vocab.get(unk_token, 0)
        self._rx = re.compile(pattern, re.UNICODE)
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
            pattern=cfg.get("pre_tokenize_pattern", r"\w+|[^\w\s]"),
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


def compare_encode_speed(tokenizer, test_text):
    """So sánh trực tiếp tốc độ Encode: Không Cache vs. Có LRU Cache."""
    S = len(tokenizer._rx.findall(unicodedata.normalize("NFC", test_text)))
    if S == 0:
        return None

    t0 = time.perf_counter()
    toks_raw = tokenizer.encode(test_text, use_cache=False)
    t_raw = time.perf_counter() - t0

    tokenizer.clear_cache()
    t0 = time.perf_counter()
    toks_cached = tokenizer.encode(test_text, use_cache=True)
    t_cached = time.perf_counter() - t0

    assert toks_raw == toks_cached, "Lệch kết quả giữa bản có và không có cache!"

    N, F = len(toks_cached), len(toks_cached) / S
    info = tokenizer._encode_cached.cache_info()
    total_calls = info.hits + info.misses
    hit_rate = (info.hits / total_calls * 100) if total_calls else 0.0
    v_raw = S / t_raw if t_raw > 0 else float("inf")
    v_cached = S / t_cached if t_cached > 0 else float("inf")
    speedup = t_raw / t_cached if t_cached > 0 else float("inf")

    print(f"=== SO SÁNH ENCODE (S={S:,} | N={N:,} | F={F:.4f} | |V|={len(tokenizer.vocab):,}) ===")
    print(f"  Không Cache : {t_raw:.4f}s | {v_raw:,.0f} âm tiết/s")
    print(f"  LRU Cache   : {t_cached:.4f}s | {v_cached:,.0f} âm tiết/s | Tăng tốc: {speedup:.2f}x (hit={hit_rate:.1f}%)")
    return {"S": S, "N": N, "F": round(F, 4), "t_raw": round(t_raw, 4), "t_cached": round(t_cached, 4), "speedup": round(speedup, 2)}


def load_test_text(file_path, skip_mb=75):
    """Bỏ qua phần train (skip_mb) theo ranh giới dòng bằng seek O(1), đọc phần còn lại."""
    skip_bytes = int(skip_mb * 1024 * 1024)
    with open(file_path, "rb") as f:
        if skip_bytes > 0:
            f.seek(skip_bytes - 1)
            f.readline()  # Đọc nốt dòng đang dở tới hết '\n'
        remaining = f.read()
    return remaining.decode("utf-8"), round(len(remaining) / (1024 * 1024), 2)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "model", "bpe_level_3.json")
    CORPUS_PATH = os.path.join(BASE_DIR, "data", "vi_wiki_80mb.txt")

    tok = BPETokenizer.from_json(MODEL_PATH)
    sample = "Sinh viên Nha Trang nghiên cứu thuật toán BPE cho tiếng Việt."
    tokens = tok.encode(sample, use_cache=True, keep_whitespace=True)
    print("Tokens:", tokens)
    print("Decode:", tok.decode(tokens, has_whitespace=True))

    if os.path.exists(CORPUS_PATH):
        test_text, test_mb = load_test_text(CORPUS_PATH, skip_mb=75)
        if test_text.strip():
            print(f"\nĐã nạp phần kiểm thử còn lại: {test_mb} MB từ '{CORPUS_PATH}'")
            compare_encode_speed(tok, test_text)
        else:
            print(f"\n[CẢNH BÁO] Tệp '{CORPUS_PATH}' chưa vượt quá 75 MB, chạy thử trên dữ liệu mẫu lặp.")
            compare_encode_speed(tok, (sample + " ") * 5000)
    else:
        print(f"\n[CẢNH BÁO] Không tìm thấy '{CORPUS_PATH}', chạy thử trên dữ liệu mẫu lặp.")
        compare_encode_speed(tok, (sample + " ") * 5000)