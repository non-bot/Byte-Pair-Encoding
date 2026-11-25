import csv
import matplotlib.pyplot as plt

with open("results/k_survey.csv", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

ks = [int(r["K"]) for r in rows]
fs = [float(r["F"]) for r in rows]

plt.plot(ks, fs, marker="o")
plt.axhline(1.0, linestyle="--", color="orange", label="F = 1")
plt.title("Khảo sát số token trung bình trên một âm tiết F theo số bước gộp K")
plt.xlabel("K")
plt.ylabel("F")
plt.xticks(ks)
plt.legend()
plt.savefig("results/f_vs_k.png", dpi=150)