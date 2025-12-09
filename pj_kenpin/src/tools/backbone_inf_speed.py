"""
PyTorch Backbone Inference Speed Benchmark

このスクリプトは、backbones.csvに記載されたモデルの推論速度を計測します。

計測条件:
- 入力サイズ: 224x224x3 (デフォルト)
  ※一部のモデル(maxvit_nano_rw_256, swinv2_*_256等)は256x256で計測
- バッチサイズ: 8
- デバイス: GPU (CUDA required)
- ウォームアップ: 5回
- 計測回数: 10回
- 推論速度: 10回の実行時間の中央値 (単位: ms)
- モデル: pretrained=False (ランダム初期化)

出力:
- backbones.csvの「Latency」列に推論速度(ms)を保存
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch

# 256x256の入力サイズが必要なモデルのリスト
MODELS_REQUIRE_256 = {
    "maxvit_nano_rw_256.sw_in1k",
    "swinv2_tiny_window8_256.ms_in1k",
    "swinv2_small_window8_256.ms_in1k",
    "swinv2_base_window8_256.ms_in1k",
}


def get_input_size(model_name):
    """
    モデル名に応じた入力サイズを返す

    Args:
        model_name: timmモデル名

    Returns:
        int: 入力画像サイズ (224 or 256)
    """
    return 256 if model_name in MODELS_REQUIRE_256 else 224


def measure_inference_speed(model_name, device, batch_size=8, img_size=224, warmup_runs=5, measure_runs=10):
    """
    モデルの推論速度を計測

    Args:
        model_name: timmモデル名
        device: torch.device
        batch_size: バッチサイズ
        img_size: 入力画像サイズ
        warmup_runs: ウォームアップ回数
        measure_runs: 計測回数

    Returns:
        float: 推論速度の中央値 (ms)
    """
    # モデルをロード
    model = timm.create_model(model_name, pretrained=False, in_chans=3, num_classes=1000).eval()
    model = model.to(device)

    # ダミー入力を作成
    x = torch.randn(batch_size, 3, img_size, img_size).to(device)

    # ウォームアップ
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(x)
            torch.cuda.synchronize()

    # 推論速度計測
    times = []
    with torch.no_grad():
        for _ in range(measure_runs):
            start = time.time()
            _ = model(x)
            torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)  # ms単位に変換

    # 中央値を返す
    return np.median(times)


def main():
    # GPU確認
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU is required for benchmarking.")

    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # CSVを読み込み
    csv_path = Path(__file__).parent / "backbones.csv"
    df = pd.read_csv(csv_path)

    # 列名を確認（BOM対策）
    model_col = df.columns[0]  # 1列目がモデル名列
    print(f"Found {len(df)} models to benchmark\n")

    # 各モデルの推論速度を計測
    failed_models = []

    for idx, row in df.iterrows():
        model_name = row[model_col]

        # モデルに応じた入力サイズを取得
        img_size = get_input_size(model_name)
        print(f"[{idx + 1}/{len(df)}] Benchmarking: {model_name} (input: {img_size}x{img_size})")

        try:
            latency = measure_inference_speed(
                model_name=model_name, device=device, batch_size=8, img_size=img_size, warmup_runs=5, measure_runs=10
            )
            df.at[idx, "Latency"] = round(latency, 2)
            print(f"  -> Latency: {latency:.2f} ms\n")

        except Exception as e:
            print(f"  -> ERROR: {str(e)}\n")
            failed_models.append((model_name, str(e)))
            df.at[idx, "Latency"] = None

    # 結果を保存
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # エラーサマリーを表示
    if failed_models:
        print(f"\n{'=' * 60}")
        print(f"Failed to benchmark {len(failed_models)} model(s):")
        print(f"{'=' * 60}")
        for model_name, error in failed_models:
            print(f"  - {model_name}")
            print(f"    Error: {error}")
    else:
        print("\nAll models benchmarked successfully!")


if __name__ == "__main__":
    main()
