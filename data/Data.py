import unicodedata
from datasets import load_dataset

dataset = load_dataset(
    "wikimedia/wikipedia", "20231101.vi", split="train", streaming=True
)
target_size_bytes = 80 * 1024 * 1024
current_size = 0
with open("data/vi_wiki_80mb.txt", "w", encoding="utf-8") as f:
    for item in dataset:
        # Chuẩn hóa NFC ngay khi trích xuất từng bài viết
        text = unicodedata.normalize("NFC", item["text"]).strip()
        if text:
            f.write(text + "\n\n")
            current_size += len(text.encode("utf-8"))
            if current_size >= target_size_bytes:
                break
print(f"Hoàn thành! Đã lưu {current_size / (1024*1024):.2f} MB (Chuẩn NFC).")
