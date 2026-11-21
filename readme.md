# Đồ án: Cài đặt thuật toán tách token BPE cho tiếng Việt

Kho mã nguồn thực hiện cài đặt thuật toán **Byte Pair Encoding (BPE)** bằng Python thuần (không dùng thư viện tách từ làm sẵn) áp dụng cho văn bản tiếng Việt, kết hợp kỹ thuật tăng tốc để tăng tốc độ huấn luyện và mã hóa.

## Thông tin đồ án

* **Đề tài:** Cài đặt thuật toán tách token BPE cho tiếng Việt
* **Sinh viên thực hiện:** Ngô Quang Tiến — **MSSV:** 66133843
* **Giảng viên hướng dẫn:** ThS. Ngô Nguyễn Tường Nghi

---

## 1. Mục tiêu và nội dung thực hiện

1. **Chuẩn bị và tiền xử lý dữ liệu tiếng Việt (`Data.py`):**
   * Trích xuất 80 MB dữ liệu văn bản từ Wikipedia tiếng Việt (`wikimedia/wikipedia`, bản `20231101.vi`) lưu thành tệp `vi_wiki_80mb.txt`.
   * Chuẩn hóa toàn bộ văn bản về bảng mã **Unicode NFC** để các chữ cái tiếng Việt có dấu được xử lý thống nhất thành 1 ký tự.
   * Chia dữ liệu theo ranh giới dòng: **75 MB đầu** dùng để huấn luyện và **5 MB cuối** dùng để kiểm thử.
   * Lọc ký tự tiếng Việt/ASCII chuẩn và loại bỏ các từ xuất hiện quá ít (`min_word_freq = 5`), thu được $U' = 32{,}335$ từ phân biệt và tập ký tự gốc $\vert V_0 \vert = 185$.

2. **Cài đặt thuật toán huấn luyện BPE theo 4 mức (`BPE_AIO_3.py`):**
   * **Mức 0 — Cơ bản:** Cài đặt trực tiếp theo định nghĩa thuật toán, duyệt toàn bộ danh sách từ và cặp ký tự ở mỗi vòng lặp gộp ($O(N + K \cdot U' \cdot L)$).
   * **Mức 1 — Dùng Chỉ mục ngược (Inverted Index):** Lưu ánh xạ từ mỗi cặp ký tự tới danh sách chỉ số các từ chứa cặp đó để chỉ cập nhật cục bộ những từ bị thay đổi.
   * **Mức 2 — Kết hợp Hàng đợi ưu tiên (Max-Heap):** Dùng `heapq` để lấy nhanh cặp có tần suất lớn nhất trong $O(\log P')$ thay vì duyệt tuyến tính toàn bộ bảng cặp.
   * **Mức 3 — Kết hợp Danh sách liên kết đôi (Doubly Linked List):** Biểu diễn mỗi từ bằng danh sách liên kết đôi để gộp cặp liền kề trực tiếp tại con trỏ trong $O(1)$.
   * Áp dụng quy tắc phá hòa ưu tiên thân âm tiết trước ký hiệu kết thúc từ `</w>` (`body_first`), đảm bảo cả 4 mức đều cho ra bộ `vocab` và `merges` **trùng khớp 100%**.

3. **Cài đặt mã hóa/giải mã và kiểm thử (`encoding_3.py`, `UnitTest.py`):**
   * Xây dựng lớp `BPETokenizer` hỗ trợ `encode`, `decode`, chuyển đổi qua lại `token IDs` và tích hợp bộ nhớ đệm **LRU Cache** để tăng tốc khi mã hóa văn bản dài.
   * Viết bộ kiểm thử đơn vị (`UnitTest.py`) tái hiện lại ví dụ chạy tay trong báo cáo và kiểm tra tính chất bảo toàn dữ liệu gốc $\text{decode}(\text{encode}(x)) = x$.

---

## 2. Cấu trúc thư mục

```text
├── data/
│   ├── data.py  
│   └── vi_wiki_80mb.txt  # Tải và chuẩn hóa NFC 80 MB dữ liệu Wikipedia tiếng Việt
├── model/
│   └── bpe_level_*.json  # Các tệp từ vựng và quy tắc gộp sinh ra sau khi huấn luyện
├── logs/
│   ├── bpe_level_*.log   # Log tiến độ và thời gian chạy từng mức
│   └── summary.log       # Log xác nhận đối chiếu chéo các mức
├── BPE_AIO_3.py          # Chương trình chính huấn luyện BPE (hỗ trợ chọn mức 0, 1, 2, 3)
├── encoding_3.py         # Lớp BPETokenizer thực hiện encode/decode và đo tốc độ LRU Cache
├── UnitTest.py           # Bộ kiểm thử đơn vị (unittest) kiểm tra thuật toán và round-trip
├── logs/                 # Thư mục lưu log kết quả chạy của từng mức
├── bpe_level_*.json      # Tệp lưu bộ từ vựng (vocab) và quy tắc gộp (merges) sau huấn luyện
└── README.md             # Tài liệu hướng dẫn của đồ án
```

---

## 3. Kết quả thực nghiệm

### Kết quả huấn luyện (75 MB đầu `vi_wiki_80mb.txt`, số bước gộp $K = 8{,}000$, kích thước từ vựng $\vert V \vert = 8{,}185$)

| Mức | Kỹ thuật cài đặt | Tiền xử lý (s) | Huấn luyện (s) | Tổng thời gian (s) | So với mức cơ bản |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | Cơ bản (không tăng tốc) | 6.219 | 199.200 | 205.419 | `1.00x` |
| **1** | Inverted Index | 6.219 | 7.120 | 13.339 | Nhanh hơn **`27.98x`** |
| **2** | Inverted Index + Max-Heap | 6.219 | 0.727 | 6.946 | Nhanh hơn **`274.00x`** |
| **3** | Inverted Index + Max-Heap + Doubly Linked List | 6.219 | **0.401** | **6.620** | Nhanh hơn **`496.76x`** |

### Kết quả mã hóa (`encoding_3.py` trên $S = 65{,}000$ âm tiết, thu được $N = 75{,}000$ token, chỉ số $F = N/S = 1.1538$)

| Chế độ chạy | Thời gian mã hóa (s) | Tốc độ xử lý (âm tiết/giây) | Tỷ lệ Cache Hit | Tăng tốc |
| :--- | :---: | :---: | :---: | :---: |
| Không dùng Cache (`use_cache=False`) | 0.3744 | 173,624 | — | `1.00x` |
| Có dùng LRU Cache (`use_cache=True`) | **0.0230** | **2,827,722** | 100.0% | **`16.29x`** |

---

## 4. Hướng dẫn cài đặt và chạy chương trình

### Yêu cầu môi trường
* **Python 3.8+** (các tệp thuật toán, mã hóa và kiểm thử đều dùng thư viện chuẩn của Python).
* Thư viện `datasets` (chỉ cần cài nếu muốn chạy lại `Data.py` để tải dữ liệu từ Hugging Face):
  ```bash
  pip install datasets
  ```

### Các bước chạy

1. **Tạo tệp dữ liệu `vi_wiki_80mb.txt`** *(bỏ qua bước này nếu đã có sẵn tệp dữ liệu)*:
   ```bash
   python Data.py
   ```

2. **Chạy huấn luyện BPE:**
   ```bash
   python BPE_AIO_3.py
   ```
   Nhập mức muốn chạy khi chương trình hiển thị lựa chọn (ví dụ nhập `3` để chạy mức tối ưu nhất, hoặc `0 1 2 3` để chạy và đối chiếu cả 4 mức):
   ```text
   === CHỌN MỨC TĂNG TỐC BPE ===
     0 : Cơ bản (không tăng tốc)
     1 : Inverted Index
     2 : Inverted Index + Max-Heap
     3 : Inverted Index + Max-Heap + Doubly Linked List
   Nhập 1 hoặc nhiều mức (VD: 2 | 0 1 2 3 | 1, 3 | 123):
   >> Lựa chọn: 3
   ```

3. **Chạy thử mã hóa, giải mã và đo tốc độ LRU Cache:**
   ```bash
   python encoding_3.py
   ```

4. **Chạy bộ kiểm thử đơn vị (Unit Test):**
   ```bash
   python -m unittest -v UnitTest.py
   ```

---

## 5. Tài liệu tham khảo

[1] D. Rathi, “Byte Pair Encoding (BPE): From Data Compression to GPT-2 Tokenization | by Daksh Rathi | Medium.” Accessed: Nov. 18, 2026. [Online]. Available: https://medium.com/@dakshrathi/byte-pair-encoding-bpe-from-data-compression-to-gpt-2-tokenization-44e35be2fd58

[2] Phạm Đình Khánh, “PhoBERT: Pre-trained language models for Vietnamese,” Khanh’s blog. Accessed: Nov. 18, 2026. [Online]. Available: https://phamdinhkhanh.github.io

[3] Nguyễn Quốc Đạt and Nguyễn Tuấn Anh, PhoBERT: Pre-trained language models for Vietnamese. (2020). github. Accessed: Nov. 18, 2026. [Online]. Available: https://github.com/VinAIResearch/PhoBERT

[4] D. Jurafsky and J. H. Martin, Speech and Language Processing, Third Edition draft. 2024. Accessed: Nov. 18, 2026. [Online]. Available: https://web.stanford.edu/~jurafsky/slp3/ed3bookaug20_2024.pdf

[5] Protonx, “Chi tiết thuật toán tách token Byte-Pair Encoding (BPE).” Accessed: Nov. 18, 2026. [Online]. Available: https://protonx.io/news/chi-tiet-thuat-toan-tach-token-byte-pair-encoding-bpe-1702295274898

[6] X. Song, A. Salcianu, Y. Song, D. Dopson, and D. Zhou, “Fast WordPiece Tokenization,” in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, M.-F. Moens, X. Huang, L. Specia, and S. W. Yih, Eds., Online and Punta Cana, Dominican Republic: Association for Computational Linguistics, Oct. 2021, pp. 2089–2103. doi: 10.18653/v1/2021.emnlp-main.160.

[7] T. Kudo, “Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates,” in Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), I. Gurevych and Y. Miyao, Eds., Melbourne, Australia: Association for Computational Linguistics, Jul. 2018, pp. 66–75. doi: 10.18653/v1/P18-1007.