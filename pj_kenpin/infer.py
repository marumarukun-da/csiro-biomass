import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data import SampleDataset, build_valid_transform_from_config
from src.model_factory import build_model

# ロギング設定
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def inference_with_tta(model: torch.nn.Module, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    バッチ単位でTTAを適用: オリジナル + 水平反転 + 垂直反転の平均

    Args:
        model: ArcFaceNetモデル
        images: torch.Tensor, shape (batch_size, C, H, W)
        device: torch.device

    Returns:
        torch.Tensor: 平均化されたlogits, shape (batch_size, num_classes)
    """
    model.eval()
    with torch.no_grad():
        # オリジナル
        logits_orig = model(images.to(device))

        # 水平反転（W方向）
        images_hflip = torch.flip(images, dims=[3])
        logits_hflip = model(images_hflip.to(device))

        # 垂直反転（H方向）
        images_vflip = torch.flip(images, dims=[2])
        logits_vflip = model(images_vflip.to(device))

        # 平均
        logits = (logits_orig + logits_hflip + logits_vflip) / 3.0

    return logits


def compute_anomaly_scores(logits: torch.Tensor) -> np.ndarray:
    """
    異常度スコアを計算（0_normalクラスへのコサイン類似度ベース）

    Args:
        logits: torch.Tensor, shape (batch_size, num_classes)
                値域 [-1, 1] のコサイン類似度

    Returns:
        np.ndarray: 異常度スコア, shape (batch_size,), 値域 [0, 1]
    """
    # 0番目クラス（0_normal）へのコサイン類似度
    cosine_sim_to_normal = logits[:, 0].cpu().numpy()

    # 異常度 = 1 - cosine_similarity
    # cosine_similarity が [-1, 1] なので、異常度は [0, 2]
    # これを [0, 1] に正規化
    anomaly_scores = (1 - cosine_sim_to_normal) / 2.0

    return anomaly_scores


def compute_metrics(true_labels: list[int], pred_labels: list[int], class_names: list[str]) -> dict[str, Any]:
    """
    分類メトリクスを計算

    Args:
        true_labels: 真のラベル（インデックス）
        pred_labels: 予測ラベル（インデックス）
        class_names: クラス名リスト

    Returns:
        dict: メトリクス辞書
    """
    # 全体の accuracy
    accuracy = accuracy_score(true_labels, pred_labels)

    # classification_report で詳細メトリクスを取得
    # labels パラメータで全クラスを明示的に指定（データに存在しないクラスも含む）
    num_classes = len(class_names)
    report = classification_report(
        true_labels,
        pred_labels,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy,
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
    }

    # クラスごとのメトリクス
    for class_name in class_names:
        if class_name in report:
            metrics[f"precision_{class_name}"] = report[class_name]["precision"]
            metrics[f"recall_{class_name}"] = report[class_name]["recall"]
            metrics[f"f1_{class_name}"] = report[class_name]["f1-score"]

    return metrics


# =============================================================================
# infer_single_run のヘルパー関数
# =============================================================================


def _validate_run_files(
    run_dir: Path,
    checkpoint: str,
) -> tuple[Path, dict[str, Any]] | None:
    """config.yamlと重みファイルの存在を確認し、設定を読み込む。

    Returns:
        (weight_path, cfg) または エラー時は None
    """
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        logger.warning(f"config.yaml not found in {run_dir}, skipping...")
        return None

    weight_path = run_dir / "weights" / f"{checkpoint}.pth"
    if not weight_path.exists():
        logger.warning(f"Weight file {weight_path} not found, skipping...")
        return None

    try:
        cfg = OmegaConf.load(str(config_path))
        cfg = OmegaConf.to_container(cfg, resolve=True)
    except Exception as e:
        logger.warning(f"Failed to load config.yaml in {run_dir}: {e}, skipping...")
        return None

    return weight_path, cfg


def _run_inference_loop(
    model: torch.nn.Module,
    dataloader: DataLoader,
    class_names: list[str],
    use_tta: bool,
    compute_anomaly: bool,
    device: torch.device,
    run_name: str,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    """推論ループを実行する。

    Returns:
        (results, true_labels, pred_labels)
    """
    results: list[dict[str, Any]] = []
    true_labels_all: list[int] = []
    pred_labels_all: list[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Inference [{run_name}]", leave=False):
            images = batch["image"]
            labels = batch["label"]
            image_paths = batch["image_path"]

            # 推論
            if use_tta:
                logits = inference_with_tta(model, images, device)
            else:
                logits = model(images.to(device))

            # 確率計算（softmax）
            probs = F.softmax(logits, dim=1).cpu().numpy()
            pred_labels = logits.argmax(dim=1).cpu().numpy()

            # 異常度計算
            anomaly_scores = compute_anomaly_scores(logits) if compute_anomaly else None

            # 結果を保存
            for i in range(len(images)):
                file_name = Path(image_paths[i]).name
                true_label = labels[i].item()
                pred_label = pred_labels[i].item()

                result_dict: dict[str, Any] = {
                    "file_name": file_name,
                    "true_label": class_names[true_label],
                    "pred_label": class_names[pred_label],
                }

                # 各クラスの確率
                for cls_idx, cls_name in enumerate(class_names):
                    result_dict[f"probability_{cls_name}"] = probs[i, cls_idx]

                # 異常度
                if compute_anomaly and anomaly_scores is not None:
                    result_dict["anomaly_score"] = anomaly_scores[i]

                results.append(result_dict)
                true_labels_all.append(true_label)
                pred_labels_all.append(pred_label)

    return results, true_labels_all, pred_labels_all


def _save_predictions(results: list[dict[str, Any]], output_path: Path) -> None:
    """予測結果をCSVに保存する。"""
    if not results:
        return

    fieldnames = list(results[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Predictions saved to: {output_path}")


# =============================================================================
# メイン関数
# =============================================================================


def infer_single_run(
    run_dir: Path,
    input_dir: Path,
    output_dir: Path,
    checkpoint: str,
    use_tta: bool,
    compute_anomaly: bool,
    device: torch.device,
) -> dict[str, Any] | None:
    """単一ランの推論を実行する。"""
    run_name = run_dir.name
    logger.info(f"Processing run: {run_name}")

    # 1. 設定・重みファイル検証
    validation = _validate_run_files(run_dir, checkpoint)
    if validation is None:
        return None
    weight_path, cfg = validation

    # 2. 設定の取得
    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    arcface_cfg = cfg.get("arcface", {})
    aug_cfg = cfg.get("augmentation", {})

    img_size = dataset_cfg.get("img_size")
    num_classes = dataset_cfg.get("num_classes")
    in_chans = model_cfg.get("in_chans", 3)

    if img_size is None:
        logger.warning(f"Missing img_size in config for {run_dir}, skipping...")
        return None
    if num_classes is None:
        logger.warning(f"Missing num_classes in config for {run_dir}, skipping...")
        return None

    # 3. データセット構築
    try:
        dataset = SampleDataset.from_directory(
            str(input_dir),
            transform=build_valid_transform_from_config(aug_cfg, img_size, in_chans),
            in_chans=in_chans,
        )
    except Exception as e:
        logger.warning(f"Failed to build dataset for {run_dir}: {e}, skipping...")
        return None

    if dataset.class_names is None or not dataset.class_names:
        logger.warning(f"No class names extracted from dataset for {run_dir}, skipping...")
        return None

    class_names = dataset.class_names
    if len(class_names) != num_classes:
        logger.warning(
            f"Class count mismatch: training used {num_classes} classes, but test data has {len(class_names)} classes."
        )

    # 4. anomaly_scoreの互換性チェック
    head_type = model_cfg.get("head", "arcface")
    if compute_anomaly and head_type == "simple":
        logger.warning(f"Anomaly score not supported for head_type='simple' in {run_name}.")
        compute_anomaly = False

    # 5. モデル構築・ロード
    try:
        net = build_model(model_cfg, arcface_cfg, num_classes, device, pretrained=False)
        net.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        net.eval()
    except Exception as e:
        logger.warning(f"Failed to build or load model for {run_dir}: {e}, skipping...")
        return None

    # 6. DataLoader作成
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # 7. 推論実行
    results, true_labels, pred_labels = _run_inference_loop(
        net, dataloader, class_names, use_tta, compute_anomaly, device, run_name
    )

    # 8. 結果保存
    run_output_dir = output_dir / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    _save_predictions(results, run_output_dir / "predictions.csv")

    # 9. メトリクス計算
    metrics = compute_metrics(true_labels, pred_labels, class_names)
    metrics["run_name"] = run_name
    metrics["checkpoint"] = checkpoint

    return metrics


def main() -> None:
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="Run inference on trained ArcFace models.")
    parser.add_argument(
        "--exp_dir", type=str, required=True, help="実験ディレクトリパス（例: output/20250108_120000_exp001）"
    )
    parser.add_argument("--input", type=str, required=True, help="推論画像フォルダパス（ラベルごとのサブフォルダ構成）")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_loss",
        choices=["best_loss", "last"],
        help="使用する重みファイル（デフォルト: best_loss）",
    )
    parser.add_argument("--anomaly_score", action="store_true", help="異常度スコアを計算する")
    parser.add_argument("--tta", action="store_true", help="Test Time Augmentation（水平反転・垂直反転）を適用する")
    args = parser.parse_args()

    # パスの検証
    exp_dir = Path(args.exp_dir).resolve()
    input_dir = Path(args.input).resolve()

    if not exp_dir.exists():
        raise FileNotFoundError(f"実験ディレクトリが存在しません: {exp_dir}")
    if not input_dir.exists():
        raise FileNotFoundError(f"入力ディレクトリが存在しません: {input_dir}")

    # 出力ディレクトリの作成
    # 入力フォルダ名を取得
    input_folder_name = input_dir.name

    # 出力フォルダ名のパーツを構築
    output_parts = [exp_dir.name, "infer", input_folder_name]

    # TTAオプションがあれば追加
    if args.tta:
        output_parts.append("with_tta")

    # 結合
    output_dir_name = "_".join(output_parts)
    output_dir = exp_dir.parent / output_dir_name

    # 例:
    # - input="data/test" の場合: "20250108_120000_exp001_infer_test"
    # - input="data/test" --tta の場合: "20250108_120000_exp001_infer_test_with_tta"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ランディレクトリの検出
    run_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and (d / "config.yaml").exists()])

    if not run_dirs:
        logger.error(f"No valid run directories found in {exp_dir}")
        return

    logger.info(f"Found {len(run_dirs)} run(s) to process")

    # 各ランの推論実行
    all_metrics = []
    for run_dir in run_dirs:
        metrics = infer_single_run(
            run_dir=run_dir,
            input_dir=input_dir,
            output_dir=output_dir,
            checkpoint=args.checkpoint,
            use_tta=args.tta,
            compute_anomaly=args.anomaly_score,
            device=device,
        )
        if metrics is not None:
            all_metrics.append(metrics)

    # サマリーCSV保存
    if all_metrics:
        summary_csv = output_dir / "inference_summary.csv"
        fieldnames = sorted({key for metrics in all_metrics for key in metrics.keys()})

        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)

        logger.info(f"Summary saved to: {summary_csv}")
        logger.info("Inference completed successfully!")
    else:
        logger.warning("No runs were processed successfully.")

    # 異常度スコア計算が有効な場合の注意喚起
    if args.anomaly_score:
        logger.info("")
        logger.info("=" * 80)
        logger.info("IMPORTANT NOTE: Anomaly Score Calculation")
        logger.info("=" * 80)
        logger.info("The anomaly scores are calculated based on the cosine distance to the FIRST class.")
        logger.info("Please ensure that the FIRST class in your training data is the 'normal' class.")
        logger.info("If the first class is not 'normal', the anomaly scores may not be meaningful.")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
