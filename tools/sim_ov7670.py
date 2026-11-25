"""Mô phỏng ảnh camera OV7670 trên ảnh thường (dataset BTC, Roboflow...) để giảm lệch miền camera.

3 lệnh:

    # 1) Đo đặc điểm camera từ ảnh OV7670 thật (chụp bằng firmware/tools/client.py capture), so với ảnh tham chiếu
    python tools/sim_ov7670.py fit --camera data/ov7670 --ref data/train/images --out tools/ov7670_profile.json

    # 2) Xem thử: mỗi hàng = ảnh gốc | ảnh mô phỏng | 1 ảnh camera thật ngẫu nhiên
    python tools/sim_ov7670.py preview --profile tools/ov7670_profile.json --src data/train/images \
        --camera data/ov7670 --out outputs/sim_preview.png

    # 3) Sinh dataset mô phỏng (giữ nguyên nhãn YOLO, vì mô phỏng không dịch chuyển pixel)
    python tools/sim_ov7670.py apply --profile tools/ov7670_profile.json --src data/train \
        --out data/train_sim --copies 1

Các bước mô phỏng (theo thứ tự xảy ra trong camera):
  màu (chuyển thống kê Lab về thống kê của 1 ảnh camera thật ngẫu nhiên) -> mờ ống kính -> tối góc
  -> nhiễu cảm biến (mạnh hơn ở vùng tối) -> lượng tử RGB565 rồi mở rộng lại 8 bit đúng như firmware/main/cam.c.
Mọi tham số đều dao động ngẫu nhiên quanh giá trị đo được, vì camera thật cũng thay đổi theo ánh sáng.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

W, H = 320, 240
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ------------------------------------------------------------------ đọc / ghi ảnh (đường dẫn có dấu vẫn được)
def list_images(d: str | Path) -> list[Path]:
    d = Path(d)
    if d.is_file():
        return [d]
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT)


def load_rgb(p: Path) -> np.ndarray | None:
    buf = np.fromfile(str(p), np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if im is None:
        return None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if im.shape[:2] != (H, W):
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    return im


def save_rgb(p: Path, im: np.ndarray) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(p.suffix or ".png", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"không ghi được {p}")
    enc.tofile(str(p))


# ------------------------------------------------------------------ các phép đo
def to_lab(im: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(im.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)   # L 0..100, a/b ~ -128..127


def lab_stats(im: np.ndarray) -> tuple[list[float], list[float]]:
    lab = to_lab(im).reshape(-1, 3)
    return lab.mean(0).tolist(), lab.std(0).tolist()


def sharpness(im: np.ndarray) -> float:
    """Phương sai Laplacian sau median 3x3 (bớt ảnh hưởng của nhiễu, chỉ còn độ nét thật)."""
    g = cv2.medianBlur(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), 3)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def noise_sigma(im: np.ndarray) -> float:
    """Độ lệch chuẩn nhiễu, ước lượng trên vùng phẳng (gradient thấp) bằng MAD của phần dư sau median."""
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    res = g - cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    flat = mag < np.percentile(mag, 30)
    r = res[flat]
    if r.size < 100:
        return 0.0
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


_YY, _XX = np.mgrid[0:H, 0:W].astype(np.float32)
R2 = (((_XX - (W - 1) / 2) / ((W - 1) / 2)) ** 2 + ((_YY - (H - 1) / 2) / ((H - 1) / 2)) ** 2) / 2.0  # 0 tâm .. 1 góc


def radial_profile(im: np.ndarray, bins: int = 10) -> np.ndarray:
    lum = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lum /= max(lum.mean(), 1.0)
    idx = np.minimum((R2 * bins).astype(int), bins - 1)
    return np.array([lum[idx == i].mean() for i in range(bins)])


def vignette_k(p: np.ndarray) -> float:
    """Khớp độ sáng theo bán kính: lum = c0 + c1*r^2  ->  k = -c1/c0 (tối góc k*100 %)."""
    r2 = (np.arange(len(p)) + 0.5) / len(p)
    c1, c0 = np.polyfit(r2, p, 1)
    return float(-c1 / c0)


def add_noise(x: np.ndarray, sig: float, rng: np.random.Generator) -> np.ndarray:
    """Nhiễu từng kênh, mạnh hơn ở vùng tối (x float 0..255). Dùng chung cho fit và simulate."""
    if sig <= 0.1:
        return x
    lum = np.clip(x.mean(-1, keepdims=True) / 255.0, 0, 1)
    return x + rng.normal(0, 1, x.shape).astype(np.float32) * sig * (0.6 + 0.8 * (1 - lum))


def degrade(im: np.ndarray, blur: float, noise: float, rng: np.random.Generator) -> np.ndarray:
    x = im.astype(np.float32)
    if blur > 0.15:
        x = cv2.GaussianBlur(x, (0, 0), blur)
    return np.clip(add_noise(x, noise, rng), 0, 255).astype(np.uint8)


def _match(target: float, grid, measure) -> float:
    """Giá trị trong grid làm measure(v) gần target nhất (so theo log)."""
    errs = [abs(np.log(measure(v) + 1e-3) - np.log(target + 1e-3)) for v in grid]
    return float(grid[int(np.argmin(errs))])


# ------------------------------------------------------------------ fit
def fit(camera: str, ref: str, n_ref: int, out: str, seed: int) -> dict:
    rng = random.Random(seed)
    cam_paths = list_images(camera)
    if len(cam_paths) < 5:
        sys.exit(f"Cần ít nhất 5 ảnh camera trong {camera} (đang có {len(cam_paths)}). Nên có 30+ ảnh, nhiều mức sáng.")
    ref_paths = list_images(ref)
    if not ref_paths:
        sys.exit(f"Không thấy ảnh tham chiếu trong {ref}")
    ref_paths = rng.sample(ref_paths, min(n_ref, len(ref_paths)))

    cams = [im for im in (load_rgb(p) for p in cam_paths) if im is not None]
    refs = [im for im in (load_rgb(p) for p in ref_paths) if im is not None]
    print(f"camera: {len(cams)} ảnh | tham chiếu: {len(refs)} ảnh")

    cam_lab = [lab_stats(im) for im in cams]
    ref_lab = [lab_stats(im) for im in refs]

    # độ mờ + nhiễu: tìm (blur, noise) sao cho ảnh tham chiếu sau degrade() đo ra giống camera.
    # Hai đại lượng ảnh hưởng lẫn nhau (nhiễu làm ảnh "nét" hơn, blur làm nhiễu nhỏ đi) -> lặp vài vòng.
    cam_sharp = float(np.median([sharpness(im) for im in cams]))
    cam_noise = [noise_sigma(im) for im in cams]
    cam_noise_med = float(np.median(cam_noise))
    probe = refs[: min(80, len(refs))]
    ref_sharp = float(np.median([sharpness(im) for im in probe]))
    ref_noise = float(np.median([noise_sigma(im) for im in probe]))

    def med(fn, blur, noise):
        g = np.random.default_rng(seed)
        return float(np.median([fn(degrade(im, blur, noise, g)) for im in probe]))

    blur_grid, noise_grid = np.arange(0.0, 3.01, 0.1), np.arange(0.0, 20.01, 0.25)
    best_sigma, add_sig = 0.0, 0.0
    for _ in range(3):
        add_sig = _match(cam_noise_med, noise_grid, lambda n: med(noise_sigma, best_sigma, n))
        best_sigma = _match(cam_sharp, blur_grid, lambda b: med(sharpness, b, add_sig))
    # độ dao động nhiễu giữa các ảnh camera, quy về cùng thang với add_sig
    scale = add_sig / cam_noise_med if cam_noise_med > 1e-3 else 0.0
    noise_spread = float(np.subtract(*np.percentile(cam_noise, [75, 25])) * scale)

    # tối góc: tỉ số profile camera / tham chiếu (bỏ phần tối góc vốn có của ảnh tham chiếu)
    p_cam = np.mean([radial_profile(im) for im in cams], 0)
    p_ref = np.mean([radial_profile(im) for im in refs], 0)
    k_cam, k_ref = vignette_k(p_cam), vignette_k(p_ref)
    k_add = max(vignette_k(p_cam / p_ref), 0.0)

    prof = {
        "version": 1,
        "n_camera": len(cams),
        "n_ref": len(refs),
        "camera_lab": cam_lab,                       # [[mean L,a,b], [std L,a,b]] từng ảnh camera
        "ref_lab_mean": np.mean([m for m, _ in ref_lab], 0).tolist(),
        "camera_lab_mean": np.mean([m for m, _ in cam_lab], 0).tolist(),
        "sharpness": {"camera": cam_sharp, "ref": ref_sharp},
        "blur_sigma": best_sigma,
        "noise": {"camera_median": cam_noise_med, "ref_median": ref_noise,
                  "add_sigma": add_sig, "spread": noise_spread},
        "vignette": {"camera_k": k_cam, "ref_k": k_ref, "add_k": k_add},
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(prof, indent=1), encoding="utf-8")

    cm, rm = prof["camera_lab_mean"], prof["ref_lab_mean"]
    print(f"màu Lab trung bình  camera L={cm[0]:.1f} a={cm[1]:+.1f} b={cm[2]:+.1f} | "
          f"tham chiếu L={rm[0]:.1f} a={rm[1]:+.1f} b={rm[2]:+.1f}   (a>0 ám đỏ, b>0 ám vàng)")
    print(f"độ nét              camera {cam_sharp:.0f} | tham chiếu {ref_sharp:.0f} -> blur sigma {best_sigma:.1f}")
    print(f"nhiễu (đo ảnh xám)  camera {cam_noise_med:.2f} | tham chiếu {ref_noise:.2f} -> thêm nhiễu sigma {add_sig:.2f}/kênh")
    print(f"tối góc             camera {k_cam * 100:.0f}% | tham chiếu {k_ref * 100:.0f}% -> thêm {k_add * 100:.0f}%")
    print(f"đã ghi {out}")
    return prof


# ------------------------------------------------------------------ mô phỏng
def rgb565_roundtrip(im: np.ndarray) -> np.ndarray:
    """Giống cam.c: RGB565 -> RGB888 bằng (r<<3)|(r>>2), (g<<2)|(g>>4), (b<<3)|(b>>2)."""
    r = im[..., 0] >> 3
    g = im[..., 1] >> 2
    b = im[..., 2] >> 3
    return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1).astype(np.uint8)


def simulate(im: np.ndarray, prof: dict, rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    """im: RGB uint8 320x240 -> ảnh 'như OV7670 chụp'. strength 0 = chỉ lượng tử RGB565, 1 = đầy đủ."""
    x = im.astype(np.float32)

    # 1) màu: đưa thống kê Lab về thống kê của 1 ảnh camera thật ngẫu nhiên (Reinhard), trộn một phần
    if strength > 0 and prof.get("camera_lab"):
        mt, st = prof["camera_lab"][rng.integers(len(prof["camera_lab"]))]
        mt = np.array(mt, np.float32) + rng.normal(0, [3.0, 1.5, 1.5]).astype(np.float32)
        st = np.array(st, np.float32)
        lab = to_lab(np.clip(x, 0, 255).astype(np.uint8))
        ms = lab.reshape(-1, 3).mean(0)
        ss = lab.reshape(-1, 3).std(0) + 1e-3
        ratio = np.clip(st / ss, 0.5, 2.0)
        out = (lab - ms) * ratio + mt
        a = float(np.clip(rng.uniform(0.6, 1.0) * strength, 0, 1))
        lab = (1 - a) * lab + a * out
        lab[..., 0] = np.clip(lab[..., 0], 0, 100)
        x = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) * 255.0

    # 2) mờ ống kính
    sigma = prof.get("blur_sigma", 0.0) * rng.uniform(0.7, 1.3) * strength
    if sigma > 0.15:
        x = cv2.GaussianBlur(x, (0, 0), sigma)

    # 3) tối góc
    k = prof.get("vignette", {}).get("add_k", 0.0) * rng.uniform(0.5, 1.5) * strength
    if k > 0.005:
        x = x * (1.0 - k * R2)[..., None]

    # 4) nhiễu cảm biến, mạnh hơn ở vùng tối
    nz = prof.get("noise", {})
    sig = max(nz.get("add_sigma", 0.0) + rng.normal(0, nz.get("spread", 0.0) / 2 + 1e-6), 0.0) * strength
    x = add_noise(x, sig, rng)

    # 5) RGB565 như firmware
    return rgb565_roundtrip(np.clip(x, 0, 255).astype(np.uint8))


def load_profile(p: str) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def split_dirs(src: Path) -> tuple[Path, Path | None]:
    """src có thể là thư mục dataset (images/ + labels/) hoặc thẳng thư mục ảnh."""
    if (src / "images").is_dir():
        return src / "images", (src / "labels") if (src / "labels").is_dir() else None
    return src, None


def apply(profile: str, src: str, out: str, copies: int, strength: float, keep_orig: bool, seed: int) -> None:
    prof = load_profile(profile)
    rng = np.random.default_rng(seed)
    img_dir, lbl_dir = split_dirs(Path(src))
    out = Path(out)
    paths = list_images(img_dir)
    if not paths:
        sys.exit(f"Không thấy ảnh trong {img_dir}")
    n_lbl = 0
    for i, p in enumerate(paths):
        im = load_rgb(p)
        if im is None:
            print(f"bỏ qua (không đọc được): {p}")
            continue
        rel = p.relative_to(img_dir).with_suffix("")
        lbl = (lbl_dir / rel).with_suffix(".txt") if lbl_dir else None
        variants = [("", im)] if keep_orig else []
        variants += [(f"_sim{k}", simulate(im, prof, rng, strength)) for k in range(copies)]
        for suffix, v in variants:
            name = rel.parent / f"{rel.name}{suffix}"
            save_rgb(out / "images" / name.with_suffix(".png"), v)
            if lbl is not None and lbl.exists():
                dst = (out / "labels" / name).with_suffix(".txt")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(lbl, dst)
                n_lbl += 1
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(paths)}")
    print(f"xong: {len(paths)} ảnh gốc -> {out / 'images'} ({n_lbl} file nhãn)")


def preview(profile: str, src: str, camera: str | None, out: str, n: int, strength: float, seed: int) -> None:
    prof = load_profile(profile)
    rng = np.random.default_rng(seed)
    pick = random.Random(seed)
    img_dir, _ = split_dirs(Path(src))
    paths = pick.sample(list_images(img_dir), n)
    cams = list_images(camera) if camera else []
    rows = []
    for p in paths:
        im = load_rgb(p)
        cells = [im, simulate(im, prof, rng, strength)]
        if cams:
            cells.append(load_rgb(pick.choice(cams)))
        rows.append(np.concatenate([np.pad(c, ((2, 2), (2, 2), (0, 0)), constant_values=255) for c in cells], 1))
    grid = np.concatenate(rows, 0)
    save_rgb(Path(out), grid)
    cols = "gốc | mô phỏng" + (" | camera thật" if cams else "")
    print(f"đã ghi {out}  ({n} hàng: {cols})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="đo đặc điểm camera -> profile JSON")
    f.add_argument("--camera", required=True, help="thư mục ảnh OV7670 thật")
    f.add_argument("--ref", default="data/train/images", help="ảnh tham chiếu (miền nguồn)")
    f.add_argument("--n-ref", type=int, default=300)
    f.add_argument("--out", default="tools/ov7670_profile.json")

    a = sub.add_parser("apply", help="sinh dataset mô phỏng")
    a.add_argument("--profile", default="tools/ov7670_profile.json")
    a.add_argument("--src", required=True, help="thư mục dataset (images/ + labels/) hoặc thư mục ảnh")
    a.add_argument("--out", required=True)
    a.add_argument("--copies", type=int, default=1, help="số bản mô phỏng mỗi ảnh")
    a.add_argument("--keep-orig", action="store_true", help="chép cả ảnh gốc sang out")
    a.add_argument("--strength", type=float, default=1.0)

    v = sub.add_parser("preview", help="ảnh so sánh gốc | mô phỏng | camera thật")
    v.add_argument("--profile", default="tools/ov7670_profile.json")
    v.add_argument("--src", default="data/train/images")
    v.add_argument("--camera", default=None)
    v.add_argument("--out", default="outputs/sim_preview.png")
    v.add_argument("-n", type=int, default=6)
    v.add_argument("--strength", type=float, default=1.0)

    for s in (f, a, v):
        s.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.cmd == "fit":
        fit(args.camera, args.ref, args.n_ref, args.out, args.seed)
    elif args.cmd == "apply":
        apply(args.profile, args.src, args.out, args.copies, args.strength, args.keep_orig, args.seed)
    else:
        preview(args.profile, args.src, args.camera, args.out, args.n, args.strength, args.seed)


if __name__ == "__main__":
    main()
