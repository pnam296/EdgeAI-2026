"""Xử lý văn bản cho intent: làm sạch câu, tách từ, từ điển token -> id.

VÒNG LOẠI: không dùng từ khóa, không dùng từ điển vật thể, không sinh thêm câu.
Mọi thông tin về từ đều do TextCNN tự học từ tập train BTC cung cấp.
Trên ESP32 phải viết lại đúng hàm normalize/tokenize này bằng C/C++.
"""
import re


def normalize(text):
    """Làm sạch chung: chữ thường, ký tự không phải a-z/0-9 thành dấu cách, gộp khoảng trắng."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return normalize(text).split()


class Vocab:
    """Từ điển token -> id, xây TỰ ĐỘNG từ câu train. 0 = PAD (đệm), 1 = OOV (từ chưa gặp)."""

    def __init__(self, texts, min_count=1):
        counts = {}
        for t in texts:
            for w in tokenize(t):
                counts[w] = counts.get(w, 0) + 1
        self.itos = ["<pad>", "<oov>"] + sorted(w for w, c in counts.items() if c >= min_count)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text, max_len):
        ids = [self.stoi.get(w, 1) for w in tokenize(text)][:max_len]
        return ids + [0] * (max_len - len(ids))
