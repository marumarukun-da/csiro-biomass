import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images_recursively(root: str):
    paths = []
    for d, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in EXTS:
                paths.append(os.path.join(d, f))
    return paths


def calc_stats(paths):
    # 累積（double精度）
    gray_sum = 0.0
    gray_sumsq = 0.0
    gray_count = 0

    rgb_sum = np.zeros(3, dtype=np.float64)
    rgb_sumsq = np.zeros(3, dtype=np.float64)
    rgb_count = 0

    for p in tqdm(paths, desc="Scanning"):
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        # [0,1] スケール
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32)

        # 自動判別
        if img.ndim == 2:  # 1ch (Gray)
            gray_sum += float(img.sum())
            gray_sumsq += float((img * img).sum())
            gray_count += int(img.size)
        elif img.ndim == 3:
            if img.shape[2] == 4:  # BGRA -> RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            elif img.shape[2] == 3:  # BGR -> RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                continue  # 想定外

            h, w, _ = img.shape
            r = img.reshape(-1, 3)
            rgb_sum += r.sum(axis=0).astype(np.float64)
            rgb_sumsq += (r * r).sum(axis=0).astype(np.float64)
            rgb_count += h * w
        else:
            # 想定外
            continue

    results = {}
    if gray_count > 0:
        g_mean = gray_sum / gray_count
        g_var = max(gray_sumsq / gray_count - g_mean**2, 0.0)
        g_std = np.sqrt(g_var)
        results["grayscale"] = ([float(g_mean)], [float(g_std)])

    if rgb_count > 0:
        r_mean = rgb_sum / rgb_count
        r_var = np.maximum(rgb_sumsq / rgb_count - r_mean**2, 0.0)
        r_std = np.sqrt(r_var)
        results["rgb"] = (r_mean.tolist(), r_std.tolist())

    return results


def main():
    ap = argparse.ArgumentParser(description="Recursively compute mean/std (auto-detect Gray/RGB).")
    ap.add_argument("image_dir", type=str, help="Root folder containing images")
    args = ap.parse_args()

    paths = list_images_recursively(args.image_dir)
    if not paths:
        raise SystemExit("画像が見つかりませんでした。拡張子とパスを確認してください。")

    stats = calc_stats(paths)

    if not stats:
        raise SystemExit("統計を計算できませんでした（対応外の画像のみの可能性）。")

    # 出力（混在時は両方表示）
    if "rgb" in stats:
        mean, std = stats["rgb"]
        m = np.round(mean, 4).tolist()
        s = np.round(std, 4).tolist()
        print("Detected: RGB (3ch)")
        print(f"  mean = {m}")
        print(f"  std  = {s}")
        print("\nAlbumentations:")
        print(f"  A.Normalize(mean={m}, std={s}, max_pixel_value=255.0)")
        print()

    if "grayscale" in stats:
        (m_g,), (s_g,) = stats["grayscale"]
        print("Detected: Grayscale (1ch)")
        print(f"  mean = ({m_g:.4f},)")
        print(f"  std  = ({s_g:.4f},)")
        print("\nAlbumentations:")
        print(f"  A.Normalize(mean=({m_g:.4f},), std=({s_g:.4f},), max_pixel_value=255.0)")


if __name__ == "__main__":
    main()
