"""Mô hình + hàm loss + bộ nạp dữ liệu cho detector tí hon (TensorFlow/Keras).

KIẾN TRÚC (~130k tham số, chỉ dùng op mà TFLite Micro hỗ trợ):

  ảnh 160x96x3
    └─ MobileNetV2 alpha=0.35 (pretrained ImageNet), CẮT ở block 13
         ├─ đặc trưng stride 8  (20x12, 96 kênh)  : chi tiết - tốt cho vật nhỏ
         └─ đặc trưng stride 16 (10x6, 192 kênh)  : ngữ cảnh - tốt cho vật lớn
    └─ Neck: 1x1 conv cả 2 nhánh -> phóng nhánh 16 lên x2 -> cộng -> 2 lớp conv tách rời 3x3
    └─ Head: heatmap (20x12xC, sigmoid)  +  box (20x12x4C)

Toàn bộ phần sau block 13 của MobileNetV2 (chiếm >70% tham số) bị bỏ đi.
"""
import numpy as np
import tensorflow as tf

from det_tiny_data import INPUT_H, INPUT_W, WH_LOG, augment, encode_targets, preprocess

NECK_CH = 64
# Độ rộng MobileNetV2: 0.35 -> ~130k tham số; 0.25 -> ~79k tham số (giữ nguyên độ sâu).
# Keras chỉ có trọng số ImageNet cho alpha 0.35/0.5/0.75/1.0 -> alpha 0.25 chỉ dùng được khi train từ đầu.
ALPHA = 0.25
L = tf.keras.layers


@tf.keras.utils.register_keras_serializable(package="edgeai")
class ResizeNearest(L.Layer):
    """Phóng to x2 kiểu nearest. Dùng thay UpSampling2D vì UpSampling2D của Keras 3 bị chuyển
    thành EXPAND_DIMS/TILE/RESHAPE; còn tf.image.resize -> RESIZE_NEAREST_NEIGHBOR (có trong TFLite Micro)."""

    def __init__(self, size, **kw):
        super().__init__(**kw)
        self.size = tuple(size)

    def call(self, x):
        return tf.image.resize(x, self.size, method="nearest")

    def get_config(self):
        return {**super().get_config(), "size": self.size}


def _conv_bn_relu(x, ch, k=1, name=None):
    x = L.Conv2D(ch, k, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = L.BatchNormalization(name=f"{name}_bn")(x)
    return L.ReLU(6.0, name=f"{name}_relu")(x)


def _sep_conv(x, ch, name):
    """Depthwise 3x3 + pointwise 1x1: rẻ hơn conv 3x3 thường ~8 lần."""
    x = L.DepthwiseConv2D(3, padding="same", use_bias=False, name=f"{name}_dw")(x)
    x = L.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = L.ReLU(6.0, name=f"{name}_dw_relu")(x)
    return _conv_bn_relu(x, ch, 1, name=f"{name}_pw")


def build_model(num_classes, pretrained=True):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(INPUT_H, INPUT_W, 3), alpha=ALPHA, include_top=False,
        weights="imagenet" if pretrained else None)
    c8 = base.get_layer("block_6_expand_relu").output    # stride 8
    c16 = base.get_layer("block_13_expand_relu").output  # stride 16

    p16 = _conv_bn_relu(c16, NECK_CH, name="lat16")
    p16 = ResizeNearest((INPUT_H // 8, INPUT_W // 8), name="up16")(p16)
    p8 = _conv_bn_relu(c8, NECK_CH, name="lat8")
    x = L.Add(name="fuse")([p8, p16])
    x = _sep_conv(x, NECK_CH, "neck1")
    x = _sep_conv(x, NECK_CH, "neck2")

    # bias -2.19 => sigmoid ~0.1 lúc đầu: đa số ô là nền, tránh loss bùng nổ ở epoch đầu
    heat = L.Conv2D(num_classes, 1, activation="sigmoid", name="heatmap",
                    bias_initializer=tf.keras.initializers.Constant(-2.19))(x)
    box = L.Conv2D(4 * num_classes, 1, name="box")(x)
    return tf.keras.Model(base.input, [heat, box], name="tiny_centernet")


# ------------------------------- LOSS -------------------------------
def focal_loss(y_true, y_pred):
    """Focal loss của CenterNet: tập trung vào ô khó, giảm phạt ô gần tâm (y_true gần 1)."""
    p = tf.clip_by_value(y_pred, 1e-4, 1 - 1e-4)
    pos = tf.cast(tf.equal(y_true, 1.0), tf.float32)
    pos_loss = tf.math.log(p) * tf.square(1 - p) * pos
    neg_loss = tf.math.log(1 - p) * tf.square(p) * tf.pow(1 - y_true, 4) * (1 - pos)
    n_pos = tf.maximum(tf.reduce_sum(pos, axis=[1, 2, 3]), 1.0)
    return -tf.reduce_sum(pos_loss + neg_loss, axis=[1, 2, 3]) / n_pos


def make_box_loss(num_classes, wh_weight=None, off_weight=1.0):
    C = num_classes
    if wh_weight is None:  # log-scale có giá trị nhỏ (-3..3) nên cần trọng số lớn hơn
        wh_weight = 1.0 if WH_LOG else 0.1

    def box_loss(y_true, y_pred):
        """L1 chỉ tại ô tâm của đúng lớp. y_true: 4C giá trị + C mask; y_pred: 4C."""
        shape = tf.shape(y_pred)
        tgt = tf.reshape(y_true[..., :4 * C], [shape[0], shape[1], shape[2], C, 4])
        pred = tf.reshape(y_pred, [shape[0], shape[1], shape[2], C, 4])
        mask = y_true[..., 4 * C:][..., None]
        err = tf.abs(pred - tgt) * mask
        n = tf.maximum(tf.reduce_sum(mask, axis=[1, 2, 3, 4]), 1.0)
        wh = tf.reduce_sum(err[..., :2], axis=[1, 2, 3, 4])
        off = tf.reduce_sum(err[..., 2:], axis=[1, 2, 3, 4])
        return (wh_weight * wh + off_weight * off) / n

    return box_loss


# ------------------------------- DỮ LIỆU -------------------------------
class DetDataset(tf.keras.utils.PyDataset):
    """Mỗi epoch: augment ngẫu nhiên lại toàn bộ ảnh -> mô hình gần như không thấy ảnh trùng lặp."""

    def __init__(self, samples, num_classes, batch_size=16, train=True, seed=0, **kw):
        super().__init__(**kw)
        self.samples, self.C, self.bs, self.train = samples, num_classes, batch_size, train
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(samples))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.bs))

    def on_epoch_end(self):
        if self.train:
            self.rng.shuffle(self.order)

    def __getitem__(self, i):
        X, H, B = [], [], []
        for j in self.order[i * self.bs:(i + 1) * self.bs]:
            s = self.samples[j]
            img, boxes = (augment(s["img"], s["boxes"], self.rng) if self.train
                          else (s["img"], s["boxes"]))
            h, b = encode_targets(boxes, self.C)
            X.append(preprocess(img))
            H.append(h)
            B.append(b)
        return np.stack(X), (np.stack(H), np.stack(B))
