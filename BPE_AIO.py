import heapq
import json
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict

LEVELS = {
    "0": "Cơ bản (không tăng tốc)",
    "1": "Inverted Index",
    "2": "Inverted Index + Max-Heap",
    "3": "Inverted Index + Max-Heap + Doubly Linked List",
}


def _build_vn_chars():
    vowels = ("a", "ă", "â", "e", "ê", "i", "o", "ô", "ơ", "u", "ư", "y")
    tones = ("", "\u0300", "\u0301", "\u0309", "\u0303", "\u0323")
    s = set("abcdefghijklmnopqrstuvwxyzđ")
    for v in vowels:
        for t in tones:
            s.add(unicodedata.normalize("NFC", v + t))
    return s | {c.upper() for c in s} | set("_.,!?;:-–—()[]\"'/%+=*")

VN_CHARS = _build_vn_chars()

class _Node:
    __slots__ = ("val", "freq", "prev", "next", "alive")

    def __init__(self, val, freq):
        self.val, self.freq = val, freq
        self.prev = self.next = None
        self.alive = True


def write_log(msg, log_file=None):
    print(msg)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

def load_words(file_path, min_word_freq, vn_only, max_mb):
    t0 = time.perf_counter()
    findall = re.compile(r"\w+|[^\w\s]", re.UNICODE).findall
    normalize = unicodedata.normalize
    is_normalized = unicodedata.is_normalized
    max_bytes = int(max_mb * 1024 * 1024) if max_mb is not None else None
    counts = Counter()
    bytes_read = 0
    update = counts.update
    with open(file_path, "rb") as f:
        while max_bytes is None or bytes_read < max_bytes:
            raw = f.readline()
            if not raw:
                break
            bytes_read += len(raw)
            line = raw.decode("utf-8")
            if not is_normalized("NFC", line):
                line = normalize("NFC", line)
            update(findall(line))

    if min_word_freq > 1 or vn_only:
        counts = Counter({
            w: c for w, c in counts.items()
            if c >= min_word_freq and (not vn_only or all(ch in VN_CHARS for ch in w))
        })
    return counts, round(time.perf_counter() - t0, 3)

def save_model(output_path, result, preprocess_time=0.0):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    data = {
        "level": result["level"],
        "algorithm": LEVELS[result["level"]],
        "preprocess_time_seconds": preprocess_time,
        "train_time_seconds": result["time"],
        "total_time_seconds": round(preprocess_time + result["time"], 3),
        "vocab": result["vocab"],
        "merges": [[a, b] for a, b in result["merges"]],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return round(os.path.getsize(output_path) / 1024, 1)

def _best_linear(pair_counts, min_pair_freq, eow):
    best_pair, best_cnt, best_tie = None, min_pair_freq - 1, None
    for pair, cnt in pair_counts.items():
        if cnt < best_cnt:
            continue
        tie = (1 if pair[1].endswith(eow) else 0, pair)
        if cnt > best_cnt or (cnt >= min_pair_freq and (best_tie is None or tie < best_tie)):
            best_cnt, best_pair, best_tie = cnt, pair, tie
    return best_pair

def _merge_list(tokens, p0, p1, merged):
    out, i, n = [], 0, len(tokens)
    while i < n:
        if i < n - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
            out.append(merged)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out

def _train_dll(word_counts, num_merges, min_pair_freq, eow, vocab, log_interval, log_file):
    """Mức 3: Inverted Index (pair -> nodes) + Max-Heap + Doubly Linked List."""
    t0 = time.perf_counter()
    pair_counts, pair_nodes = defaultdict(int), defaultdict(list)

    for word, freq in word_counts.items():
        if not word or freq <= 0:
            continue
        prev = None
        for ch in list(word) + [eow]:
            node = _Node(ch, freq)
            if prev is not None:
                prev.next, node.prev = node, prev
                p = (prev.val, node.val)
                pair_counts[p] += freq
                pair_nodes[p].append(prev)
            prev = node

    heap = [(-c, 1 if p[1].endswith(eow) else 0, p) for p, c in pair_counts.items() if c >= min_pair_freq]
    heapq.heapify(heap)
    merges = []

    for step in range(1, num_merges + 1):
        best_pair = None
        while heap:
            neg_c, _, p = heapq.heappop(heap)
            cur = pair_counts.get(p, 0)
            if cur >= min_pair_freq and -neg_c == cur:
                best_pair = p
                break
        if best_pair is None:
            break

        p0, p1 = best_pair
        merged = p0 + p1
        merges.append(best_pair)
        vocab.setdefault(merged, len(vocab))

        nodes = pair_nodes.pop(best_pair, ())
        pair_counts.pop(best_pair, None)
        delta, same = defaultdict(int), (p0 == p1)

        for node in nodes:
            if not node.alive or node.val != p0 or node.next is None or node.next.val != p1:
                continue
            cur = node
            if same:
                while cur.prev is not None and cur.prev.val == p0:
                    cur = cur.prev

            while cur and cur.alive and cur.val == p0 and cur.next and cur.next.val == p1:
                nxt, f = cur.next, cur.freq
                left, right = cur.prev, nxt.next
                if left:
                    delta[(left.val, p0)] -= f
                if right:
                    delta[(p1, right.val)] -= f

                cur.val, cur.next, nxt.alive = merged, right, False
                if right:
                    right.prev = cur

                if left:
                    lp = (left.val, merged)
                    delta[lp] += f
                    pair_nodes[lp].append(left)
                if right:
                    rp = (merged, right.val)
                    delta[rp] += f
                    pair_nodes[rp].append(cur)
                cur = right if same else None

        for p, d in delta.items():
            if d == 0:
                continue
            nc = pair_counts.get(p, 0) + d
            if nc <= 0:
                pair_counts.pop(p, None)
                pair_nodes.pop(p, None)
            else:
                pair_counts[p] = nc
                if nc >= min_pair_freq:
                    heapq.heappush(heap, (-nc, 1 if p[1].endswith(eow) else 0, p))

        if log_interval > 0 and step % log_interval == 0:
            write_log(
                f"  [Mức 3 {step:>5}/{num_merges}] {best_pair} -> '{merged}' | Vocab: {len(vocab)} | {time.perf_counter()-t0:.2f}s",
                log_file,
            )

    return merges, round(time.perf_counter() - t0, 3)

def train_bpe(word_counts, num_merges, level="2", min_pair_freq=2, eow="</w>", log_interval=500, log_file=None):
    """Hàm BPE duy nhất điều phối 4 mức tăng tốc: '0', '1', '2', '3'."""
    if level not in LEVELS:
        raise ValueError(f"Mức không hợp lệ: '{level}'. Chỉ chọn: {list(LEVELS)}")

    base = {ch for w, f in word_counts.items() if w and f > 0 for ch in list(w) + [eow]}
    vocab = {"<unk>": 0}
    for tok in sorted(base):
        vocab.setdefault(tok, len(vocab))

    if level == "3":
        merges, elapsed = _train_dll(word_counts, num_merges, min_pair_freq, eow, vocab, log_interval, log_file)
        return {"level": level, "vocab": vocab, "merges": merges, "time": elapsed}

    t0 = time.perf_counter()
    words = [list(w) + [eow] for w, f in word_counts.items() if w and f > 0]
    freqs = [f for w, f in word_counts.items() if w and f > 0]
    use_inv, use_heap = level in ("1", "2"), level == "2"
    merges = []

    if use_inv:
        pair_counts, inv = defaultdict(int), defaultdict(set)
        for idx, toks in enumerate(words):
            f = freqs[idx]
            for i in range(len(toks) - 1):
                p = (toks[i], toks[i + 1])
                pair_counts[p] += f
                inv[p].add(idx)
        if use_heap:
            heap = [(-c, 1 if p[1].endswith(eow) else 0, p) for p, c in pair_counts.items() if c >= min_pair_freq]
            heapq.heapify(heap)

    for step in range(1, num_merges + 1):
        if not use_inv:
            pair_counts = defaultdict(int)
            for toks, f in zip(words, freqs):
                for i in range(len(toks) - 1):
                    pair_counts[(toks[i], toks[i + 1])] += f
            best_pair = _best_linear(pair_counts, min_pair_freq, eow)
        elif not use_heap:
            best_pair = _best_linear(pair_counts, min_pair_freq, eow)
        else:
            best_pair = None
            while heap:
                neg_c, _, p = heapq.heappop(heap)
                cur = pair_counts.get(p, 0)
                if cur >= min_pair_freq and -neg_c == cur:
                    best_pair = p
                    break
        if best_pair is None:
            break

        p0, p1 = best_pair
        merged = p0 + p1
        merges.append(best_pair)
        vocab.setdefault(merged, len(vocab))

        if not use_inv:
            for idx, toks in enumerate(words):
                if p0 in toks:
                    words[idx] = _merge_list(toks, p0, p1, merged)
        else:
            affected = inv.pop(best_pair, ())
            pair_counts.pop(best_pair, None)
            delta = defaultdict(int)
            for idx in affected:
                old_t, f = words[idx], freqs[idx]
                new_t = _merge_list(old_t, p0, p1, merged)
                words[idx] = new_t

                old_s = set()
                for j in range(len(old_t) - 1):
                    p = (old_t[j], old_t[j + 1])
                    if p != best_pair:
                        delta[p] -= f
                        old_s.add(p)
                new_s = set()
                for j in range(len(new_t) - 1):
                    p = (new_t[j], new_t[j + 1])
                    delta[p] += f
                    new_s.add(p)
                for p in old_s - new_s:
                    post = inv.get(p)
                    if post:
                        post.discard(idx)
                        if not post:
                            del inv[p]
                for p in new_s - old_s:
                    inv[p].add(idx)

            for p, d in delta.items():
                if d == 0:
                    continue
                nc = pair_counts.get(p, 0) + d
                if nc <= 0:
                    pair_counts.pop(p, None)
                    inv.pop(p, None)
                else:
                    pair_counts[p] = nc
                    if use_heap and nc >= min_pair_freq:
                        heapq.heappush(heap, (-nc, 1 if p[1].endswith(eow) else 0, p))

        if log_interval > 0 and step % log_interval == 0:
            write_log(
                f"  [Mức {level} {step:>5}/{num_merges}] {best_pair} -> '{merged}' | Vocab: {len(vocab)} | {time.perf_counter()-t0:.2f}s",
                log_file,
            )

    return {"level": level, "vocab": vocab, "merges": merges, "time": round(time.perf_counter() - t0, 3)}

def parse_levels(raw):
    """Nhận '0 1 2 3', '1, 3', '123' hoặc '2' -> trả về danh sách mức hợp lệ, không trùng."""
    tokens = [t for t in re.split(r"[\s,;]+", raw.strip()) if t]
    if not tokens:
        return None
    expanded = []
    for tok in tokens:
        if tok in LEVELS:
            expanded.append(tok)
        elif len(tok) > 1 and all(ch in LEVELS for ch in tok):
            expanded.extend(tok)
        else:
            return None
    return list(dict.fromkeys(expanded))

def choose_levels():
    print("=== CHỌN MỨC TĂNG TỐC BPE ===")
    for k, desc in LEVELS.items():
        print(f"  {k} : {desc}")
    print("Nhập 1 hoặc nhiều mức (VD: 2 | 0 1 2 3 | 1, 3 | 123):")
    while True:
        try:
            raw = input(">> Lựa chọn: ")
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Đã hủy nhập liệu từ bàn phím.")
            return []
        picks = parse_levels(raw)
        if picks:
            return picks
        print("[LỖI] Chỉ được nhập các mức: 0, 1, 2, 3.")

def main(levels=None):
    CORPUS_PATH = "data/vi_wiki_80mb.txt"
    TRAIN_MB = 75          # Chỉ đọc 75 MB đầu để huấn luyện (5 MB cuối dành cho encoding_2.py)
    NUM_MERGES = 8000
    MIN_WORD_FREQ = 5
    MIN_PAIR_FREQ = 2
    LOG_INTERVAL = 1000

    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(f"Không tìm thấy tệp '{CORPUS_PATH}'.")

    selected = levels if levels is not None else choose_levels()
    if not selected:
        return
    if "0" in selected and NUM_MERGES >= 2000:
        print(f"[CẢNH BÁO] Mức 0 (Cơ bản) với K={NUM_MERGES} trên 80 MB sẽ chạy rất lâu!")

    word_counts, read_t = load_words(
        CORPUS_PATH, min_word_freq=MIN_WORD_FREQ, vn_only=True, max_mb=TRAIN_MB
    )
    print(f"Đã đọc {TRAIN_MB} MB đầu của '{CORPUS_PATH}' trong {read_t}s | U sau lọc = {len(word_counts):,}\n")

    results = []
    for lv in selected:
        log_file = f"logs/bpe_level_{lv}.log"
        write_log(
            f"=== MỨC {lv}: {LEVELS[lv]} (K={NUM_MERGES}, U={len(word_counts):,}, Tiền xử lý={read_t}s) ===",
            log_file,
        )
        res = train_bpe(
            word_counts,
            NUM_MERGES,
            level=lv,
            min_pair_freq=MIN_PAIR_FREQ,
            log_interval=LOG_INTERVAL,
            log_file=log_file,
        )
        size_kb = save_model(f"model/bpe_level_{lv}.json", res, preprocess_time=read_t)
        total_t = round(read_t + res["time"], 3)
        write_log(
            f"-> XONG Mức {lv}: K={len(res['merges'])}, Vocab={len(res['vocab'])}, "
            f"Tiền xử lý={read_t}s, Train={res['time']}s, Tổng={total_t}s, File={size_kb} KB\n",
            log_file,
        )
        results.append(res)

    if len(results) >= 2:
        for r in results[1:]:
            assert r["merges"] == results[0]["merges"] and r["vocab"] == results[0]["vocab"]
        write_log("[VERIFY PASS] Các m ức đã chọn cho ra vocab & merges khớp nhau 100%.", "logs/summary.log")

if __name__ == "__main__":
    main()