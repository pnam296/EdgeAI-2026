"""BƯỚC 1 - Khám phá dữ liệu detection.

Chạy:  python step1_explore.py
Mục tiêu: biết kích thước ảnh, lớp nào ít mẫu, vật nào sẽ "biến mất" khi resize nhỏ.
Kết quả lưu thêm ảnh vẽ box mẫu vào outputs/preview/ để kiểm tra nhãn bằng mắt.
"""
from collections import Counter, defaultdict

from PIL import Image, ImageDraw

from common import OUT, TEST_IMG, TRAIN_IMG, TRAIN_LBL, list_images, load_classes, read_yolo_label, yolo_to_xyxy

# Các kích thước đầu vào ứng viên cho mô hình chạy trên ESP32 (mỗi chiều <= 160)
CANDIDATE_INPUTS = [(96, 96), (128, 128), (160, 96), (160, 160)]
TINY_PX = 4  # box nhỏ hơn 4 px ở đầu vào mô hình thì gần như không thể phát hiện

classes = load_classes()
train_imgs = list_images(TRAIN_IMG)
test_imgs = list_images(TEST_IMG)
print(f"{len(classes)} lớp | {len(train_imgs)} ảnh train | {len(test_imgs)} ảnh test\n")

# --- Kích thước ảnh ---
sizes = Counter(Image.open(p).size for p in train_imgs + test_imgs)
print("Kích thước ảnh (W x H): số ảnh")
for (w, h), n in sizes.most_common():
    print(f"  {w} x {h}: {n}")

# --- Thống kê nhãn ---
n_box = Counter()
n_img_with = Counter()
rel_w, rel_h = defaultdict(list), defaultdict(list)
empty = 0
for img in train_imgs:
    boxes = read_yolo_label(TRAIN_LBL / (img.stem + ".txt"))
    if not boxes:
        empty += 1
    for c in set(b[0] for b in boxes):
        n_img_with[c] += 1
    for c, cx, cy, w, h in boxes:
        n_box[c] += 1
        rel_w[c].append(w)
        rel_h[c].append(h)

print(f"\nẢnh không có vật thể: {empty}")
print(f"\n{'id':>2} {'lớp':<24}{'box':>5}{'ảnh':>5}{'w nhỏ nhất':>12}{'h nhỏ nhất':>12}")
for c, name in enumerate(classes):
    if n_box[c]:
        print(f"{c:>2} {name:<24}{n_box[c]:>5}{n_img_with[c]:>5}{min(rel_w[c]):>12.3f}{min(rel_h[c]):>12.3f}")
    else:
        print(f"{c:>2} {name:<24}{0:>5}{0:>5}   (KHÔNG có mẫu!)")

# --- Box sẽ còn bao nhiêu pixel sau khi resize về kích thước đầu vào nhỏ? ---
print(f"\nTỉ lệ box có cạnh ngắn < {TINY_PX}px sau khi resize (càng cao càng khó):")
header = "".join(f"{f'{w}x{h}':>10}" for w, h in CANDIDATE_INPUTS)
print(f"{'lớp':<24}{header}")
for c, name in enumerate(classes):
    if not n_box[c]:
        continue
    row = ""
    for W, H in CANDIDATE_INPUTS:
        tiny = sum(min(w * W, h * H) < TINY_PX for w, h in zip(rel_w[c], rel_h[c]))
        row += f"{100 * tiny / n_box[c]:>9.0f}%"
    print(f"{name:<24}{row}")

# --- Vẽ box lên vài ảnh để kiểm tra nhãn bằng mắt ---
preview = OUT / "preview"
preview.mkdir(exist_ok=True)
colors = ["red", "lime", "blue", "yellow", "cyan", "magenta", "orange", "white", "purple", "brown"]
for img in train_imgs[:8]:
    im = Image.open(img).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)
    for c, cx, cy, w, h in read_yolo_label(TRAIN_LBL / (img.stem + ".txt")):
        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, W, H)
        col = colors[c % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=col, width=4)
        draw.text((x1 + 4, y1 + 4), classes[c], fill=col)
    im.save(preview / img.name)
print(f"\nĐã lưu ảnh có vẽ box vào {preview}")  
