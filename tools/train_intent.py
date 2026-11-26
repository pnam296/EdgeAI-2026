"""Train TextCNN ý định (v2) - tổng quát hơn cho câu hỏi chung kết (cách gọi đồ vật / viết tắt / gõ sai MỚI).

Giữ NGUYÊN kiến trúc của pipeline.ipynb (Embedding 32 + Conv1D 2/3/4 + GlobalMaxPool + Dense) -> cùng tập op TFLM.
Khác bản cũ ở cách sinh dữ liệu (thay từ theo TỪ ĐIỂN chỉ lúc train - đúng luật BTC):
  1. Tên vật lạ / gõ sai / token <oov> ở chỗ tên vật -> GIỮ ý định của câu (bản cũ ép thành OUT_OF_SCOPE, làm
     'dt', 'iphone', 'vo', 'chaii'... bị coi là ngoài phạm vi). OOS chỉ cho vật ĐÃ BIẾT ngoài 7 lớp, câu hỏi thuộc
     tính (màu, giá, chất liệu...) và chuyện ngoài lề.
  2. Nhiều từ đồng nghĩa / cách gọi cho từng lớp; câu hỏi nhiều vật; câu xác nhận vị trí ("... phai khong").
  3. Viết tắt kiểu chat cho từ chức năng (khong->ko/k/hok, bao nhieu->bn, vay->z...) và gõ sai ngẫu nhiên tên vật.
Bộ đề data/intent_exam.csv KHÔNG dùng để train; một số cách gọi trong đề cố tình không có trong từ điển (HELD_OUT).

    python tools/train_intent.py --out outputs/intent_v2
    python firmware/tools/export_intent.py --model outputs/intent_v2/intent_textcnn.keras \
        --vocab outputs/intent_v2/intent_vocab.txt --out outputs/intent_v2
    python firmware/tools/eval_qa.py --tflite outputs/intent_v2/textcnn_model.tflite --vocab outputs/intent_v2/vocab.cc
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
INTENTS = ["COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"]
MAX_LEN, EMB = 24, 32
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------- từ điển (chỉ dùng lúc TRAIN để sinh câu)
IN_SCOPE = {
    "bottle": ["chai", "chai nuoc", "chai nhua", "binh nuoc", "binh", "chai nuoc suoi", "chai lavie", "binh nhua",
               "chai thuy tinh", "chai coca", "chai nuoc ngot"],
    "cup": ["coc", "coc uong nuoc", "ly", "tach", "coc thuy tinh", "ly thuy tinh", "coc su", "ly su", "coc giay",
            "coc nhua", "ly nuoc", "tach tra", "tach ca phe", "coc nuoc", "ca phe ly", "mug"],
    "phone": ["dien thoai", "dien thoai di dong", "smartphone", "di dong", "dt", "dien thoai thong minh",
              "chiec dien thoai", "may dien thoai", "samsung", "dien thoai cam ung"],
    "book": ["sach", "cuon sach", "quyen sach", "vo", "quyen vo", "cuon vo", "so tay", "quyen so", "tap vo",
             "giao trinh", "sach giao khoa", "cuon so"],
    "mouse": ["chuot", "chuot may tinh", "con chuot may tinh", "chuot khong day", "con chuot", "chuot co day"],
    "box": ["hop", "cai hop", "hop giay", "thung giay", "thung carton", "hop nhua", "hop dung do an", "hop com",
            "hop khan giay", "goi khan giay", "hop qua", "hop banh", "hop sua", "thung"],
    "toy_car": ["oto do choi", "o to do choi", "xe oto do choi", "xe do choi", "xe", "o to mo hinh", "xe mo hinh",
                "oto", "xe hoi do choi", "xe dua do choi"],
}
# cách gọi có trong bộ đề nhưng CỐ TÌNH không đưa vào từ điển -> đo khả năng tổng quát thật
HELD_OUT = ["iphone", "dtdd", "truyen", "hot wheels", "binh giu nhiet"]
OUT_SCOPE = ["binh hoa", "cai thia", "den ban", "tai nghe", "cai vi", "cay but", "but bi", "cai remote", "cai dia",
             "banh mi", "con meo", "cuon giay ve sinh", "qua chuoi", "cai mu", "dong xu", "laptop", "may tinh xach tay",
             "cuc tay", "doi giay", "cai thuoc ke", "con cho", "cai bat", "cai keo", "kinh mat", "cai quat",
             "chia khoa", "con gau bong", "o cam dien", "may tinh bang", "cay nen", "qua bong", "cai o", "ban phim",
             "dong ho", "cai ghe", "em be", "chau cay", "con chim", "qua tao", "nguoi", "cai ban", "man hinh",
             "loa", "sac du phong", "day sac", "cuc sac", "may anh", "tivi", "cai luoc", "ba lo", "cap sach",
             "cai thot", "con dao", "cai nia", "khau trang", "binh chua chay", "cai den pin", "con ca", "bong den"]
FUNC = [(r"\bkhong\b", ["ko", "k", "hok", "hong", "khg", "kh"]), (r"\bbao nhieu\b", ["bn", "bnhieu", "bao nhiu"]),
        (r"\bvay\b", ["z", "v", "vay a"]), (r"\bgi\b", ["j"]), (r"\bduoc\b", ["dc", "duoc k"]),
        (r"\bminh\b", ["mk", "mik"]), (r"\bvoi\b", ["vs"]), (r"\bnhung\b", ["nhung"]), (r"\bban oi\b", ["b oi", "bn oi"]),
        (r"\bgiup\b", ["gium", "dum", "giup voi"]), (r"\bo dau\b", ["o dau z", "dau"]), (r"\bcai\b", ["", "chiec"])]
ALL_IN = sorted({n for v in IN_SCOPE.values() for n in v}, key=len, reverse=True)
IN_RE = re.compile(r"\b(" + "|".join(re.escape(n) for n in ALL_IN) + r")\b")
OOS_ATTR = ["{o} mau gi", "{o} nay mau gi", "mau cua {o} la gi", "{o} lam bang gi", "{o} lam bang chat lieu gi",
            "{o} dung de lam gi", "{o} hang gi", "{o} ten la gi", "{o} gia bao nhieu", "{o} nay gia bao nhieu tien",
            "{o} cua ai", "{o} nay cua ai", "{o} con moi khong", "{o} co dat khong", "{o} nang bao nhieu",
            "{o} cao bao nhieu", "{o} mua o dau", "{o} co dep khong", "nen mua {o} nao", "{o} con pin khong",
            "{o} con day khong", "{o} chay nhanh khong", "{o} co bi hong khong", "{o} san xuat o dau"]
OOS_GENERAL = ["hom nay an gi", "ten ban la gi", "ban dang lam gi", "day la gi", "may gio roi", "hom nay la thu may",
               "troi hom nay co mua khong", "chao ban", "xin chao", "cam on ban", "ban co khoe khong",
               "ke chuyen cuoi di", "1 cong 1 bang may", "thu do viet nam la gi", "dich sang tieng anh",
               "giai phuong trinh", "anh nay chup o dau", "ai chup buc anh nay", "anh chup luc may gio",
               "camera do phan giai bao nhieu", "bat den len", "tat den di", "hat mot bai di", "nen doc sach gi",
               "lam sao de hoc gioi", "cach nau com", "ban la ai", "ban bao nhieu tuoi", "ok", "hop ly khong",
               "ly do la gi", "co the giup toi khong", "co bao nhieu nguoi trong anh", "nguoi trong anh la ai"]
COUNT_T = ["co bao nhieu {o}", "co may {o}", "bao nhieu {o}", "dem so {o}", "so luong {o} la bao nhieu",
           "co tat ca may {o}", "tong cong co bao nhieu {o}", "trong anh co may {o}", "tren ban co bao nhieu {o}",
           "dem giup minh so {o}", "cho biet so {o}", "{o} co bao nhieu", "{o} co may cai", "may {o} vay"]
EXIST_T = ["co {o} khong", "co {o} nao khong", "trong anh co {o} khong", "tren ban co {o} khong",
           "co thay {o} khong", "co {o} hay khong", "lieu co {o} khong", "xem co {o} khong", "{o} co trong anh khong",
           "anh co chua {o} khong", "ban co nhin thay {o} khong", "co ton tai {o} khong"]
LOC_T = ["{o} o dau", "{o} nam o dau", "{o} o ben nao", "{o} nam phia nao", "vi tri cua {o}", "{o} o cho nao",
         "{o} dat o dau", "{o} nam ben trai hay ben phai", "{o} o giua hay o canh", "tim vi tri {o}",
         "xac dinh vi tri {o}", "{o} nam trai giua hay phai", "{o} nam ben trai phai khong", "{o} o giua dung khong",
         "{o} nam ben phai a", "{o} dang o phia nao"]
MULTI = {"COUNT": ["co bao nhieu {a} va {b}", "dem so {a} va {b}", "co may {a} may {b}", "so {a} va so {b} la bao nhieu"],
         "EXIST": ["co {a} va {b} khong", "co {a} hay {b} khong", "trong anh co {a} hoac {b} khong"],
         "LOCATION": ["{a} va {b} o dau", "{a} voi {b} nam o dau"]}
LIST_CORES = [
    "trong anh co gi", "trong hinh co gi", "tren ban co gi", "co gi trong anh", "co gi o day", "co nhung gi",
    "trong anh co nhung gi", "trong hinh co nhung thu gi", "anh nay co gi", "tren ban dang co nhung gi",
    "tren ban bay nhung gi", "co nhung mon gi", "liet ke cac do vat", "liet ke do vat trong anh", "liet ke tat ca",
    "liet ke vat the", "ke ten cac do vat", "ke ten nhung vat trong hinh", "ke ra cac mon do", "ke het do tren ban",
    "neu ten cac vat the", "nhung do vat nao co trong anh", "nhung vat nao xuat hien", "co cac loai do vat nao",
    "ban thay nhung gi trong anh", "ban thay gi trong anh", "ban nhin thay gi", "cho biet trong hinh co gi",
    "co may loai do vat", "trong anh co bao nhieu loai vat khac nhau", "anh nay gom nhung mon do nao",
    "mo ta cac vat tren ban", "cho minh danh sach do vat", "liet ke va dem tung loai", "cac vat the trong anh la gi",
    "trong buc anh co nhung doi tuong nao", "co nhung do dung gi", "tong hop cac loai do vat", "dem tung loai do vat",
    "tren ban co nhung mon gi", "trong khung hinh co nhung gi", "co do gi tren ban", "nhin thay nhung do vat nao"]
PREFIX = ["", "", "", "cho hoi ", "xin hoi ", "ban oi ", "cho minh hoi ", "hay cho biet ", "giup minh "]
SUFFIX = ["", "", "", " vay", " the", " nhi", " a", " nhe", " ban"]


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def typo(w, rng):
    """gõ sai 1 ký tự: lặp / bỏ / đổi chỗ / thay chữ bên cạnh trên bàn phím"""
    if len(w) < 3:
        return w
    i = rng.randrange(len(w))
    op = rng.random()
    if op < 0.3:
        return w[:i] + w[i] + w[i:]
    if op < 0.55:
        return w[:i] + w[i + 1:]
    if op < 0.8 and i < len(w) - 1:
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    near = {"a": "s", "s": "a", "o": "i", "i": "o", "e": "r", "n": "m", "m": "n", "h": "j", "c": "v", "t": "r"}
    return w[:i] + near.get(w[i], w[i]) + w[i + 1:]


def chat(t, rng, p=0.5):
    for pat, subs in FUNC:
        if re.search(pat, t) and rng.random() < p:
            t = re.sub(pat, rng.choice(subs), t, count=1)
    return normalize(t)


def wrap(q, rng):
    return normalize(rng.choice(PREFIX) + q + rng.choice(SUFFIX))


def obj(rng, mode):
    """tên vật cho chỗ trống: 'in' từ điển, 'typo' gõ sai, 'unk' từ lạ (chưa gặp) -> vẫn là vật hợp lệ"""
    o = rng.choice(ALL_IN)
    if mode == "typo":
        ws = o.split()
        k = rng.randrange(len(ws))
        ws[k] = typo(ws[k], rng)
        o = " ".join(ws)
    elif mode == "unk":
        o = rng.choice(["oovunseen", "oovunseen oovunseen", rng.choice(ALL_IN).split()[0] + " oovunseen"])
    return o


def build_data(train_df, seed):
    rng = random.Random(seed)
    T, Y = [], []

    def add(q, y):
        T.append(normalize(q))
        Y.append(y)

    # 1) câu gốc BTC + biến thể (thay tên vật bằng cách gọi khác / gõ sai / từ lạ: GIỮ ý định)
    for q, y in zip(train_df.question, train_df.intent):
        q = normalize(q)
        add(q, y)
        if rng.random() < 0.4:
            add(chat(q, rng), y)
        if y in ("COUNT", "EXIST", "LOCATION") and IN_RE.search(q):
            for mode in ("in", "in", "typo", "unk"):
                add(IN_RE.sub(lambda m: obj(rng, mode), q, count=1), y)
            add(IN_RE.sub(lambda m: rng.choice(OUT_SCOPE), q, count=1), "OUT_OF_SCOPE")   # vật ĐÃ BIẾT ngoài lớp
    # 2) mẫu câu sinh thêm
    for tmpls, y in ((COUNT_T, "COUNT"), (EXIST_T, "EXIST"), (LOC_T, "LOCATION")):
        for tm in tmpls:
            for _ in range(25):
                mode = rng.choices(["in", "typo", "unk"], [0.7, 0.2, 0.1])[0]
                q = wrap(tm.format(o=obj(rng, mode)), rng)
                add(chat(q, rng, 0.3), y)
            for _ in range(6):
                add(wrap(tm.format(o=rng.choice(OUT_SCOPE)), rng), "OUT_OF_SCOPE")
    for y, tmpls in MULTI.items():
        for tm in tmpls:
            for _ in range(30):
                add(chat(wrap(tm.format(a=obj(rng, "in"), b=obj(rng, "in")), rng), rng, 0.3), y)
    for tm in OOS_ATTR:
        for _ in range(15):
            o = rng.choice(ALL_IN + OUT_SCOPE)
            add(chat(wrap(tm.format(o=o), rng), rng, 0.3), "OUT_OF_SCOPE")
    for q in OOS_GENERAL:
        for _ in range(6):
            add(chat(wrap(q, rng), rng, 0.3), "OUT_OF_SCOPE")
    for _ in range(300):   # toàn từ lạ / rác -> OOS
        add(" ".join(["oovunseen"] * rng.randint(1, 5)), "OUT_OF_SCOPE")
    # LIST: câu LIST gốc + biến thể chat + mẫu câu tự nhiên (giữ LIST ngang các lớp khác)
    lst = [normalize(q) for q, y in zip(train_df.question, train_df.intent) if y == "LIST"]
    for q in rng.sample(lst, min(len(lst), 1200)):
        add(chat(wrap(q, rng), rng, 0.6), "LIST")
    for core in LIST_CORES:
        for _ in range(12):
            add(chat(wrap(core, rng), rng, 0.4), "LIST")
    return T, Y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(REPO / "data" / "intent_train.csv"))
    ap.add_argument("--exam", default=str(REPO / "data" / "intent_exam.csv"))
    ap.add_argument("--fasttext", default=str(REPO / "pipeline" / "weights" / "vi_fasttext_32d.npz"))
    ap.add_argument("--out", default=str(REPO / "outputs" / "intent_v2"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout", type=float, default=0.0,
                    help="giấu tỉ lệ câu BTC này khỏi train (kể cả khỏi bước sinh câu) rồi chấm + in câu sai")
    a = ap.parse_args()

    import tensorflow as tf
    tf.keras.utils.set_random_seed(a.seed)
    L = tf.keras.layers
    df = pd.read_csv(a.data)
    held = None
    if a.holdout > 0:
        idx = np.random.default_rng(a.seed).permutation(len(df))
        nh = int(len(df) * a.holdout)
        held, df = df.iloc[idx[:nh]].reset_index(drop=True), df.iloc[idx[nh:]].reset_index(drop=True)
    T, Y = build_data(df, a.seed)
    leak = [w for w in HELD_OUT if any(re.search(rf"\b{re.escape(w)}\b", t) for t in T)]
    print(f"{len(df)} câu gốc -> {len(T)} câu train | phân bố: {pd.Series(Y).value_counts().to_dict()}")
    print(f"cách gọi giữ lại để kiểm tra (không được có trong train): {HELD_OUT} | bị lọt: {leak or 'không'}")

    ft = np.load(a.fasttext)
    ft_words = ft["words"].tolist()
    words = {w for t in T for w in t.split()} | set(ft_words)
    words -= {w for h in HELD_OUT for w in h.split() if w not in {x for t in T for x in t.split()}}
    words.discard("oovunseen")   # luôn là <oov> (id 1)
    itos = ["<pad>", "<oov>"] + sorted(words)
    stoi = {w: i for i, w in enumerate(itos)}
    enc = lambda t: ([stoi.get(w, 1) for w in normalize(t).split()][:MAX_LEN] + [0] * MAX_LEN)[:MAX_LEN]
    X = np.array([enc(t) for t in T], np.int32)
    y = np.array([INTENTS.index(v) for v in Y], np.int32)
    ft_map = {w: ft["emb"][i] for i, w in enumerate(ft_words)}
    emb = np.random.default_rng(a.seed).normal(0, 0.15, (len(itos), EMB)).astype(np.float32)
    emb[0] = 0
    for i, w in enumerate(itos):
        if w in ft_map:
            emb[i] = ft_map[w]

    tin = L.Input((MAX_LEN,), dtype="int32", name="tokens")
    e = L.Embedding(len(itos), EMB, weights=[emb], trainable=True)(tin)
    br = [L.GlobalMaxPooling1D()(L.Conv1D(32, k, activation="relu", padding="same")(e)) for k in (2, 3, 4)]
    h = L.Dropout(0.3)(L.Concatenate()(br))
    m = tf.keras.Model(tin, L.Dense(len(INTENTS), activation="softmax")(h))
    m.compile(optimizer=tf.keras.optimizers.Adam(3e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cnt = np.bincount(y, minlength=len(INTENTS))
    cw = {i: len(y) / (len(INTENTS) * c) for i, c in enumerate(cnt) if c}
    m.fit(X, y, epochs=a.epochs, batch_size=64, class_weight=cw, verbose=2, shuffle=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m.save(out / "intent_textcnn.keras")
    (out / "intent_vocab.txt").write_text("\n".join(itos), encoding="utf-8")
    print(f"TextCNN {m.count_params():,} tham số | vocab {len(itos)} -> {out}")
    if held is not None:
        p = m.predict(np.array([enc(q) for q in held.question], np.int32), verbose=0)
        pi = p.argmax(1)
        ok = pi == held.intent.map(INTENTS.index).values
        print()
        print(f"BTC giấu {len(held)} câu: đúng {ok.mean():.4f} ({(~ok).sum()} câu sai)")
        for gi in INTENTS:
            mk = (held.intent == gi).values
            print(f"   {gi:13s} {ok[mk].mean():.3f} ({mk.sum()} câu)")
        for j in np.where(~ok)[0]:
            print(f"   SAI {held.question[j]!r}: {held.intent[j]} -> {INTENTS[pi[j]]} ({p[j].max():.2f})")
    if Path(a.exam).exists():   # chấm nhanh bản float trên bộ đề (không train trên đề)
        ex = pd.read_csv(a.exam)
        p = m.predict(np.array([enc(q) for q in ex.question.map(lambda s: __import__("unicodedata").normalize(
            "NFD", s).encode("ascii", "ignore").decode())], np.int32), verbose=0).argmax(1)
        ok = p == ex.intent.map(INTENTS.index).values
        print(f"bộ đề (float): ý định đúng {ok.mean():.3f}")
        for gi in INTENTS:
            mk = (ex.intent == gi).values
            print(f"   {gi:13s} {ok[mk].mean():.2f} ({mk.sum()} câu)")


if __name__ == "__main__":
    main()
