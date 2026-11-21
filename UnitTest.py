import os
import tempfile
import unicodedata
import unittest
from collections import Counter

try:
    from BPE_AIO import save_model, train_bpe
except ImportError:
    from BPE_AIO import save_model, train_bpe

try:
    from encoding import BPETokenizer
except ImportError:
    from encoding import BPETokenizer


class TestVietnameseBPE(unittest.TestCase):
    """Bộ kiểm thử đơn vị (Unit Test) cho huấn luyện BPE và tính chất decode(encode(x)) == x."""

    def setUp(self):
        # Ví dụ chạy tay trong báo cáo: "học" (5), "họ" (2), "hóa" (3)
        self.manual_word_counts = Counter({"học": 5, "họ": 2, "hóa": 3})
        self.expected_merges = [
            ("h", "ọ"),       # Bước 1: tần suất = 5 + 2 = 7
            ("họ", "c"),      # Bước 2: tần suất = 5 (ưu tiên thân từ trước ('c', '</w>'))
            ("học", "</w>"),  # Bước 3: tần suất = 5
        ]
        self.expected_base_vocab = {"<unk>", "</w>", "a", "c", "h", "ó", "ọ"}
        self.expected_merged_tokens = {"họ", "học", "học</w>"}

    def test_01_manual_dry_run_example_all_levels(self):
        """Tái hiện chính xác ví dụ chạy tay qua cả 4 mức tối ưu hóa (0, 1, 2, 3)."""
        results = {}
        for level in ("0", "1", "2", "3"):
            res = train_bpe(
                word_counts=self.manual_word_counts,
                num_merges=3,
                level=level,
                min_pair_freq=1,
                eow="</w>",
                log_interval=0,
            )
            results[level] = res

            # 1. Kiểm tra danh sách quy tắc gộp (merges) khớp từng bước chạy tay
            self.assertEqual(res["merges"], self.expected_merges, f"Sai merges ở mức {level}")

            # 2. Kiểm tra kích thước và tập phần tử từ vựng |V| = 7 + 3 = 10
            expected_vocab_keys = self.expected_base_vocab | self.expected_merged_tokens
            self.assertEqual(set(res["vocab"].keys()), expected_vocab_keys)
            self.assertEqual(len(res["vocab"]), 10)

            # 3. Kiểm tra phân tách token từng từ sau 3 bước gộp
            tok = BPETokenizer(res["vocab"], res["merges"])
            self.assertEqual(tok.encode("học", use_cache=False), ["học</w>"])
            self.assertEqual(tok.encode("họ", use_cache=False), ["họ", "</w>"])
            self.assertEqual(tok.encode("hóa", use_cache=False), ["h", "ó", "a", "</w>"])
            self.assertEqual(
                tok.encode("học họ hóa", use_cache=True),
                ["học</w>", "họ", "</w>", "h", "ó", "a", "</w>"],
            )

        # 4. Kiểm chứng chéo giữa 4 mức (0 == 1 == 2 == 3)
        for level in ("1", "2", "3"):
            self.assertEqual(results[level]["merges"], results["0"]["merges"])
            self.assertEqual(results[level]["vocab"], results["0"]["vocab"])

    def test_02_roundtrip_decode_encode_exact_preservation(self):
        """Kiểm tra bất biến decode(encode(x)) == x khi giữ khoảng trắng (keep_whitespace=True)."""
        corpus_counts = Counter({
            "Sinh": 6, "viên": 8, "Nha": 7, "Trang": 7, "nghiên": 6, "cứu": 6,
            "thuật": 9, "toán": 9, "BPE": 10, "cho": 8, "tiếng": 10, "Việt": 10,
            ",": 6, ".": 8, "!": 4, "(": 3, ")": 3, "2026": 5,
        })
        res = train_bpe(corpus_counts, num_merges=40, level="3", min_pair_freq=2, log_interval=0)
        tok = BPETokenizer(res["vocab"], res["merges"])

        test_sentences = [
            "học họ hóa",
            "Sinh viên Nha Trang nghiên cứu thuật toán BPE cho tiếng Việt.",
            "  Nhiều   khoảng   trắng\tvà\nxuống dòng (2026)!  ",
            "Từ mới chưa từng gặp lúc train: Khánh Hòa, Đà Nẵng, 100%!",
            "Chuỗi chứa ký tự đặc biệt </w> ngay trong văn bản gốc.",
        ]

        for text in test_sentences:
            for use_cache in (False, True):
                tokens = tok.encode(text, use_cache=use_cache, keep_whitespace=True)
                restored = tok.decode(tokens, has_whitespace=True)
                self.assertEqual(
                    restored,
                    text,
                    f"Vi phạm decode(encode(x)) == x (use_cache={use_cache}) với đầu vào: {text!r}",
                )

    def test_03_roundtrip_unicode_nfd_to_nfc_and_ids(self):
        """Kiểm tra chuẩn hóa NFD -> NFC và chu trình mã hóa/giải mã qua chỉ số nguyên (IDs)."""
        res = train_bpe(self.manual_word_counts, num_merges=3, level="2", min_pair_freq=1, log_interval=0)
        tok = BPETokenizer(res["vocab"], res["merges"])

        text_nfc = "học họ hóa"
        text_nfd = unicodedata.normalize("NFD", text_nfc)
        self.assertNotEqual(text_nfd, text_nfc)

        # Đầu vào NFD sau khi encode -> decode phải khớp dạng chuẩn NFC
        tokens_from_nfd = tok.encode(text_nfd, use_cache=True, keep_whitespace=True)
        self.assertEqual(tok.decode(tokens_from_nfd, has_whitespace=True), text_nfc)

        # Kiểm tra chu trình qua token IDs: decode_ids(encode_ids(x)) == x
        ids_no_cache = tok.encode_ids(text_nfc, use_cache=False)
        ids_cached = tok.encode_ids(text_nfc, use_cache=True)
        self.assertEqual(ids_no_cache, ids_cached)
        self.assertEqual(tok.decode_ids(ids_cached), text_nfc)

    def test_04_edge_cases_and_json_roundtrip(self):
        """Kiểm tra các điều kiện biên: chuỗi rỗng, chuỗi lặp ký tự, lưu/nạp tệp JSON."""
        # 1. Chuỗi rỗng & chỉ có khoảng trắng
        res = train_bpe(self.manual_word_counts, num_merges=3, level="1", min_pair_freq=1, log_interval=0)
        tok = BPETokenizer(res["vocab"], res["merges"])
        self.assertEqual(tok.encode("", use_cache=False), [])
        self.assertEqual(tok.encode("", use_cache=True), [])
        self.assertEqual(tok.decode([]), "")
        self.assertEqual(tok.decode(tok.encode("   \n\t  ", keep_whitespace=True), has_whitespace=True), "   \n\t  ")

        # 2. Chuỗi lặp ký tự & cặp liền kề trùng nhau (kiểm tra Doubly Linked List mức 3)
        rep_counts = Counter({"aaaa": 5, "aaaaa": 4, "abab": 6, "ababab": 5})
        res_0 = train_bpe(rep_counts, num_merges=20, level="0", min_pair_freq=1, log_interval=0)
        res_3 = train_bpe(rep_counts, num_merges=20, level="3", min_pair_freq=1, log_interval=0)
        self.assertEqual(res_0["merges"], res_3["merges"])
        self.assertEqual(res_0["vocab"], res_3["vocab"])

        tok_rep = BPETokenizer(res_3["vocab"], res_3["merges"])
        for s in ("aaaa aaaaa", "abab ababab", "aa ab a"):
            self.assertEqual(tok_rep.decode(tok_rep.encode(s, keep_whitespace=True), has_whitespace=True), s)

        # 3. Lưu ra JSON và nạp lại bằng BPETokenizer.from_json
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.json")
            save_model(model_path, res_3, preprocess_time=0.01)
            tok_loaded = BPETokenizer.from_json(model_path)
            self.assertEqual(
                tok_loaded.decode(tok_loaded.encode("aaaa ababab", keep_whitespace=True), has_whitespace=True),
                "aaaa ababab",
            )

if __name__ == "__main__":
    unittest.main(verbosity=2)