"""Coverage calculation module for grass coverage rate estimation."""

import cv2
import numpy as np

# HSVパラメータ（固定値）
H_MIN = 35  # 黄緑
H_MAX = 85  # 深緑
S_MIN = 40  # 彩度下限（土・影除外）


def calculate_coverage(img_rgb: np.ndarray) -> tuple[float, float]:
    """
    画像から草の被覆率を計算する

    Args:
        img_rgb: RGB形式の画像 [H, W, 3]

    Returns:
        tuple: (coverage_raw, coverage_log)
            - coverage_raw: 被覆率 (0〜1)
            - coverage_log: log(coverage_raw + 1e-6)
    """
    # RGB -> HSV
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)

    # 「草っぽい緑」のマスク作成
    mask = ((H >= H_MIN) & (H <= H_MAX) & (S >= S_MIN)).astype(np.uint8)

    # ノイズ除去（モルフォロジー処理）
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 被覆率計算
    coverage_raw = mask.sum() / mask.size
    coverage_log = np.log(coverage_raw + 1e-6)

    return coverage_raw, coverage_log


def calculate_coverage_from_path(img_path: str) -> tuple[float, float]:
    """
    画像パスから草の被覆率を計算する

    Args:
        img_path: 画像ファイルのパス

    Returns:
        tuple: (coverage_raw, coverage_log)
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return calculate_coverage(img_rgb)
