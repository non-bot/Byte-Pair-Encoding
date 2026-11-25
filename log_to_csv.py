import csv
import os
import re
import statistics
from collections import defaultdict

LEVEL_NAMES = {
    "0": "Co ban (khong tang toc)",
    "1": "Inverted Index",
    "2": "Inverted Index + Max-Heap",
    "3": "Inverted Index + Max-Heap + Doubly Linked List",
}
BPE_RX = re.compile(r"-> XONG Mức \d: K=(\d+), Vocab=(\d+), Tiền xử lý=[\d.]+s, Train=([\d.]+)s")
ENC_RX = re.compile(r"-> ENCODE EVAL: \|V\|=(\d+) \| S=(\d+) \| N=(\d+) \| F=([\d.]+) \| UNK=([\d.]+)%")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Ghép logs/bpe_level_3.log + logs/encoding_eval.log -> results/k_survey.csv (lấy trung vị các lần chạy)
runs_by_kv = defaultdict(list)
enc_by_vocab = {}

lv3_log = os.path.join(LOGS_DIR, "bpe_level_3.log")
if os.path.exists(lv3_log):
    with open(lv3_log, encoding="utf-8") as f:
        for k, v, train_s in BPE_RX.findall(f.read()):
            runs_by_kv[(int(k), int(v))].append(float(train_s))

enc_log = os.path.join(LOGS_DIR, "encoding_eval.log")
if os.path.exists(enc_log):
    with open(enc_log, encoding="utf-8") as f:
        for v, s, n, f_val, unk_pct in ENC_RX.findall(f.read()):
            enc_by_vocab[int(v)] = (int(s), int(n), float(f_val), float(unk_pct))

with open(os.path.join(RESULTS_DIR, "k_survey.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["K", "Vocab_size", "Train_time_s", "Test_S", "Test_N", "F", "UNK_rate_pct"])
    for k, v in sorted(runs_by_kv):
        med_train = round(statistics.median(runs_by_kv[(k, v)][-3:]), 3)
        w.writerow([k, v, med_train, *enc_by_vocab.get(v, ("", "", "", ""))])

# 2. Rút trung vị 3 lần chạy K=8000 của cả 4 mức -> results/benchmark_median.csv
med_rows = []
for lv in ("0", "1", "2", "3"):
    path = os.path.join(LOGS_DIR, f"bpe_level_{lv}.log")
    if not os.path.exists(path):
        continue
    with open(path, encoding="utf-8") as f:
        runs = [float(m[2]) for m in BPE_RX.findall(f.read()) if m[0] == "8000"][-3:]
    if len(runs) == 3:
        med = round(statistics.median(runs), 3)
        med_rows.append([lv, LEVEL_NAMES[lv], *runs, med])

with open(os.path.join(RESULTS_DIR, "benchmark_median.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["Muc", "Thuat_toan", "Lan_1_s", "Lan_2_s", "Lan_3_s", "Trung_vi_s"])
    w.writerows(med_rows)

print("Đã xuất xong results/k_survey.csv và results/benchmark_median.csv!")