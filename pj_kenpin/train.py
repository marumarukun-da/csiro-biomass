import argparse
import csv
import json
import logging
from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from timm.utils import ModelEmaV3
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.data import SampleDataset, build_train_transform_from_config, build_valid_transform_from_config
from src.loss_function import compute_class_weights_from_counts
from src.model_factory import build_criterion, build_model
from src.seed import seed_everything

# リストのまま保持するキー（グリッドサーチ対象としない）
NON_SWEEP_PATHS: set[tuple[str, ...]] = {
    ("experiment", "notes"),
    ("arcface", "margins"),  # 予約
    ("trainer", "class_weighting", "weights"),  # 手動クラス重み
}


@dataclass(frozen=True)
class SweepParam:
    path: tuple[str, ...]
    values: Sequence[Any]


def load_yaml_with_includes(path: Path) -> dict[str, Any]:
    """__include__ に対応しながら YAML を再帰的に読み込む。"""
    cfg = OmegaConf.load(str(path))
    data = OmegaConf.to_container(cfg, resolve=True) or {}

    includes = data.pop("__include__", [])
    result: dict[str, Any] = {}

    if isinstance(includes, str):
        includes = [includes]

    for inc in includes or []:
        include_path = (path.parent / inc).resolve()
        if not include_path.exists():
            raise FileNotFoundError(f"Include 先のファイルが存在しません: {include_path}")
        include_cfg = load_yaml_with_includes(include_path)
        result = merge_dicts(result, include_cfg)

    result = merge_dicts(result, data)
    return result


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """辞書を再帰的にマージする。override 側が優先。"""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def collect_sweep_params(cfg: dict[str, Any]) -> list[SweepParam]:
    """YAMLからグリッドサーチ対象のパラメータを収集する。リスト値がグリッドサーチ対象となる。"""
    params: list[SweepParam] = []

    def _is_non_sweep_path(path: tuple[str, ...]) -> bool:
        """グリッドサーチ対象外のパスかどうかを判定"""
        # 完全一致での除外
        if path in NON_SWEEP_PATHS:
            return True

        # augmentation.normalize配下は全て除外（統計量はパラメータ値）
        if len(path) >= 2 and path[0] == "augmentation" and path[1] == "normalize":
            return True

        return False

    def _collect(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                _collect(v, path + (k,))
        elif isinstance(node, list):
            # 除外パスのチェック
            if _is_non_sweep_path(path):
                return

            if not node:
                raise ValueError(f"グリッドサーチ対象のリストが空です: {'/'.join(path)}")

            # *_range, *_limit などのaugmentationパラメータの特殊処理
            if path and path[-1].endswith(("_range", "_limit", "_limits")):
                # 最初の要素がリストかどうかで判定
                if node and isinstance(node[0], list):
                    # [[10, 30], [20, 40]] → グリッドサーチ対象
                    params.append(SweepParam(path=path, values=node))
                else:
                    # [10, 30] → パラメータ値として保持（グリッドサーチ対象外）
                    return
            else:
                # 通常のリスト: ネストがない場合のみグリッドサーチ対象
                if any(isinstance(item, (dict, list)) for item in node):
                    return
                params.append(SweepParam(path=path, values=node))

    _collect(cfg, tuple())
    return params


def apply_sweep_values(cfg: dict[str, Any], assignments: Sequence[tuple[tuple[str, ...], Any]]) -> dict[str, Any]:
    """グリッドサーチの組み合わせを設定に適用する。"""
    updated = deepcopy(cfg)
    for path, value in assignments:
        target = updated
        for key in path[:-1]:
            if key not in target:
                raise KeyError(f"設定内にキー {'/'.join(path)} が存在しません。")
            target = target[key]
        target[path[-1]] = value
    return updated


def make_run_descriptor(assignments: Sequence[tuple[tuple[str, ...], Any]]) -> dict[str, Any]:
    """実行時のパラメータを記述用の辞書に変換する。"""
    descriptor: dict[str, Any] = {}
    for path, value in assignments:
        descriptor["/".join(path)] = value
    return descriptor


def sanitize_name(name: str) -> str:
    """ファイル名として使える文字列に変換する。"""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def ensure_dir(path: Path) -> None:
    """ディレクトリが存在しなければ作成する。"""
    path.mkdir(parents=True, exist_ok=True)


def build_dataloader(
    dataset_cfg: dict[str, Any],
    trainer_cfg: dict[str, Any],
    aug_cfg: dict[str, Any],
    img_size: int,
    in_chans: int = 3,
) -> tuple[SampleDataset, SampleDataset, DataLoader, DataLoader]:
    """データセットとDataLoaderを構築する。"""
    train_dir = Path(dataset_cfg["train_dir"]).expanduser()
    val_dir = Path(dataset_cfg["val_dir"]).expanduser()

    if not train_dir.exists():
        raise FileNotFoundError(f"学習データディレクトリが存在しません: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"検証データディレクトリが存在しません: {val_dir}")

    train_dataset = SampleDataset.from_directory(
        str(train_dir),
        transform=build_train_transform_from_config(aug_cfg, img_size, in_chans),
        in_chans=in_chans,
    )
    valid_dataset = SampleDataset.from_directory(
        str(val_dir),
        transform=build_valid_transform_from_config(aug_cfg, img_size, in_chans),
        in_chans=in_chans,
    )

    # データセットから自動抽出されたクラス名を使用
    if train_dataset.class_names is None or not train_dataset.class_names:
        raise ValueError("学習データディレクトリからクラスを抽出できませんでした。")

    extracted_num_classes = len(train_dataset.class_names)

    # 設定ファイルの num_classes を使用
    num_classes = dataset_cfg.get("num_classes")
    if num_classes is None:
        raise ValueError("dataset.num_classes が設定されていません。")

    # バリデーション: データセットから抽出されたクラス数との整合性チェック
    if num_classes != extracted_num_classes:
        logging.warning(
            f"設定ファイルの num_classes ({num_classes}) と "
            f"データセットから抽出されたクラス数 ({extracted_num_classes}) が一致しません。"
        )

    train_batch_size = trainer_cfg.get("train_batch_size", 32)
    val_batch_size = trainer_cfg.get("val_batch_size", 64)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=trainer_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=trainer_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
    )

    if len(train_loader) == 0:
        raise RuntimeError("train_loader のバッチ数が 0 です。batch_size を見直してください。")

    return train_dataset, valid_dataset, train_loader, valid_loader


def compute_class_counts(labels: Sequence[int], num_classes: int) -> list[int]:
    """各クラスのサンプル数を計算する。"""
    counter = Counter(labels)
    counts = [counter.get(i, 0) for i in range(num_classes)]
    if any(c == 0 for c in counts):
        logging.warning("一部クラスのサンプル数が 0 です。ArcFace の学習が不安定になる可能性があります。")
    return counts


def plot_training_curves(
    history: dict[str, list[float]],
    output_dir: Path,
) -> None:
    """学習曲線をプロットして保存する。"""
    if plt is None:
        return

    ensure_dir(output_dir)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()


def configure_logger(log_path: Path) -> logging.Logger:
    """ランごとのログファイルを設定する。"""
    logger = logging.getLogger(f"train.{log_path.stem}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# =============================================================================
# train_single_run のヘルパー関数
# =============================================================================


def _setup_run_directories(run_dir: Path) -> None:
    """出力ディレクトリ（weights, logs, plots）を準備する。"""
    ensure_dir(run_dir)
    (run_dir / "weights").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)


def _compute_class_weights(
    class_counts: list[int],
    class_weighting_cfg: dict[str, Any],
    num_classes: int,
    logger: logging.Logger,
) -> np.ndarray | None:
    """class_weighting設定に基づいてクラス重みを計算する。

    Returns:
        クラス重みの配列、または重み付け無効時はNone
    """
    weighting_enabled = class_weighting_cfg.get("enabled", True)
    weighting_mode = class_weighting_cfg.get("mode", "auto")
    manual_weights = class_weighting_cfg.get("weights", [])

    if not weighting_enabled:
        logger.info("Class weighting: disabled")
        return None

    if weighting_mode == "auto":
        class_weights = compute_class_weights_from_counts(class_counts)
        logger.info("Class weighting: auto (based on sample counts)")
        return class_weights

    if weighting_mode == "manual":
        if len(manual_weights) != num_classes:
            raise ValueError(
                f"class_weighting.weights の長さ({len(manual_weights)})が num_classes({num_classes})と一致しません"
            )
        if any(w <= 0 for w in manual_weights):
            raise ValueError("class_weighting.weights に負の値またはゼロが含まれています")
        logger.info(f"Class weighting: manual (weights={manual_weights})")
        return np.array(manual_weights)

    raise ValueError(f"Unknown class_weighting.mode: {weighting_mode}. Must be 'auto' or 'manual'")


def _setup_optimizer_scheduler_ema(
    model: torch.nn.Module,
    optimization_cfg: dict[str, Any],
    trainer_cfg: dict[str, Any],
    total_steps: int,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[torch.optim.Optimizer, Any, GradScaler, ModelEmaV3 | None]:
    """オプティマイザ、スケジューラ、AMP scaler、EMAを設定する。"""
    # オプティマイザ
    lr = optimization_cfg.get("lr", 1e-3)
    weight_decay = optimization_cfg.get("weight_decay", 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # スケジューラ
    warmup_rate = optimization_cfg.get("warmup_rate", 0.1)
    warmup_steps = int(total_steps * warmup_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # AMP scaler
    use_amp = bool(trainer_cfg.get("use_amp", torch.cuda.is_available() and device.type == "cuda"))
    scaler = GradScaler(device="cuda", enabled=use_amp)

    # EMA
    use_ema = optimization_cfg.get("use_ema", True)
    ema = None
    if use_ema:
        ema_decay = optimization_cfg.get("ema_decay", 0.995)
        ema_start_ratio = optimization_cfg.get("ema_start_ratio", 0.1)
        update_after_step = int(total_steps * ema_start_ratio)
        ema = ModelEmaV3(model, decay=ema_decay, update_after_step=update_after_step)
        logger.info(f"EMA enabled: decay={ema_decay}, start_ratio={ema_start_ratio}")
    else:
        logger.info("EMA disabled")

    return optimizer, scheduler, scaler, ema


def _run_train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler,
    ema: ModelEmaV3 | None,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    global_step: int,
    use_amp: bool,
) -> tuple[float, int]:
    """1エポックの学習を実行する。

    Returns:
        (train_loss, updated_global_step)
    """
    model.train()
    train_loss_sum = 0.0
    train_samples = 0

    train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]", leave=False)
    for batch in train_iter:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            train_loss = criterion(logits, labels)

        scaler.scale(train_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        global_step += 1
        if ema is not None:
            ema.update(model, step=global_step)

        batch_size = images.size(0)
        train_samples += batch_size
        train_loss_sum += float(train_loss.item()) * batch_size

    train_loss_epoch = train_loss_sum / max(train_samples, 1)
    return train_loss_epoch, global_step


def _run_valid_epoch(
    model: torch.nn.Module,
    valid_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> float:
    """1エポックの検証を実行する。

    Returns:
        val_loss
    """
    model.eval()
    val_loss_sum = 0.0
    val_samples = 0

    with torch.no_grad():
        val_iter = tqdm(valid_loader, desc=f"Epoch {epoch}/{num_epochs} [val]", leave=False)
        for batch in val_iter:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            val_loss = criterion(logits, labels)

            val_loss_sum += float(val_loss.item()) * images.size(0)
            val_samples += images.size(0)

    return val_loss_sum / max(val_samples, 1)


# =============================================================================
# メイン関数
# =============================================================================


def train_single_run(
    cfg: dict[str, Any],
    run_dir: Path,
    run_descriptor: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """単一の設定で学習を実行する。EMAモデルで評価・保存を行う。"""
    # 1. 出力ディレクトリの準備
    _setup_run_directories(run_dir)
    OmegaConf.save(config=OmegaConf.create(cfg), f=str(run_dir / "config.yaml"))

    logger = configure_logger(run_dir / "logs" / "train.log")
    logger.info("==== Run parameters ====")
    logger.info(json.dumps(run_descriptor, ensure_ascii=False, indent=2))

    # 2. 設定の取得
    experiment_cfg = cfg.get("experiment", {})
    trainer_cfg = cfg.get("trainer", {})
    dataset_cfg = cfg.get("dataset", {})
    optimization_cfg = cfg.get("optimization", {})
    arcface_cfg = cfg.get("arcface", {})
    model_cfg = cfg.get("model", {})
    aug_cfg = cfg.get("augmentation", {})

    # 3. シード設定
    seed = experiment_cfg.get("seed", 42)
    seed_everything(seed)
    logger.info("Seed set to %s", seed)

    # 4. データ準備
    img_size = dataset_cfg.get("img_size")
    if img_size is None:
        raise ValueError("dataset.img_size が指定されていません。")
    if isinstance(img_size, (list, tuple)):
        raise ValueError("グリッドサーチ適用後も dataset.img_size がリストのままです。設定を見直してください。")
    img_size = int(img_size)
    in_chans = model_cfg.get("in_chans", 3)

    train_dataset, _, train_loader, valid_loader = build_dataloader(
        dataset_cfg=dataset_cfg,
        trainer_cfg=trainer_cfg,
        aug_cfg=aug_cfg,
        img_size=img_size,
        in_chans=in_chans,
    )

    # 5. クラス情報の取得
    num_classes = dataset_cfg.get("num_classes")
    if num_classes is None:
        raise ValueError("dataset.num_classes が設定されていません。")
    class_counts = compute_class_counts(train_dataset.labels, num_classes)

    # 6. クラス重み計算
    class_weighting_cfg = trainer_cfg.get("class_weighting", {})
    class_weights = _compute_class_weights(class_counts, class_weighting_cfg, num_classes, logger)

    # 7. モデル・損失関数構築
    head_type = model_cfg.get("head", "arcface")
    net = build_model(model_cfg, arcface_cfg, num_classes, device, pretrained=model_cfg.get("pretrained", True))
    criterion = build_criterion(head_type, arcface_cfg, trainer_cfg, class_counts, class_weights, num_classes, device)
    logger.info(f"Using head type: {head_type}")

    # 8. オプティマイザ・スケジューラ・EMA設定
    num_epochs = trainer_cfg.get("num_epochs", 10)
    total_steps = len(train_loader) * num_epochs
    optimizer, scheduler, scaler, ema = _setup_optimizer_scheduler_ema(
        net, optimization_cfg, trainer_cfg, total_steps, device, logger
    )
    use_amp = scaler.is_enabled()

    # 9. 学習ループ
    metrics_path = run_dir / "logs" / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as csv_fp:
        csv_writer = csv.writer(csv_fp)
        csv_writer.writerow(["epoch", "train_loss", "val_loss"])

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_loss_epoch = -1
        global_step = 0

        for epoch in range(1, num_epochs + 1):
            # 訓練フェーズ
            train_loss_epoch, global_step = _run_train_epoch(
                net,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                scaler,
                ema,
                device,
                epoch,
                num_epochs,
                global_step,
                use_amp,
            )

            # 検証フェーズ（EMAモデルまたは通常モデル）
            eval_model = ema.module if ema is not None else net
            val_loss_epoch = _run_valid_epoch(eval_model, valid_loader, criterion, device, epoch, num_epochs)

            # 履歴記録
            history["train_loss"].append(train_loss_epoch)
            history["val_loss"].append(val_loss_epoch)
            csv_writer.writerow([epoch, train_loss_epoch, val_loss_epoch])
            csv_fp.flush()

            logger.info(
                "Epoch %d/%d [train] loss=%.5f [val] loss=%.5f", epoch, num_epochs, train_loss_epoch, val_loss_epoch
            )

            # ベストモデル保存
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                best_loss_epoch = epoch
                save_model = ema.module if ema is not None else net
                torch.save(save_model.state_dict(), run_dir / "weights" / "best_loss.pth")
                logger.info("best_loss.pth を更新しました (epoch=%d)", epoch)

        # 最終モデル保存
        save_model = ema.module if ema is not None else net
        torch.save(save_model.state_dict(), run_dir / "weights" / "last.pth")
        logger.info("最終モデルを保存しました。")

    # 10. 結果保存
    plot_training_curves(history, run_dir / "plots")

    summary = {"best_val_loss": best_val_loss, "best_val_loss_epoch": best_loss_epoch}
    summary.update(run_descriptor)
    return summary


def create_run_name(index: int, descriptor: dict[str, Any]) -> str:
    """グリッドサーチのパラメータからランの名前を生成する。"""
    backbone = descriptor.get("model/backbone")
    backbone_slug = sanitize_name(str(backbone)) if backbone else f"run{index:03d}"
    other_parts = [
        f"{sanitize_name(k.split('/')[-1])}-{sanitize_name(str(v))}"
        for k, v in descriptor.items()
        if k != "model/backbone"
    ]
    suffix = "__".join(other_parts) if other_parts else ""
    if suffix:
        return f"{index:03d}_{backbone_slug}__{suffix}"
    return f"{index:03d}_{backbone_slug}"


def main() -> None:
    """メイン実行関数。YAMLを読み込み、グリッドサーチを実行して全ランを学習する。"""
    parser = argparse.ArgumentParser(description="Train image classification models with/without ArcFace.")
    parser.add_argument("--config", type=str, required=True, help="実験設定 YAML のパス")
    args = parser.parse_args()

    # 設定ファイルの読み込み
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが存在しません: {config_path}")

    base_cfg = load_yaml_with_includes(config_path)
    sweep_params = collect_sweep_params(base_cfg)

    # グリッドサーチの組み合わせを生成
    assignments_list: list[list[tuple[tuple[str, ...], Any]]] = []
    if sweep_params:
        path_list = [param.path for param in sweep_params]
        value_list = [param.values for param in sweep_params]
        for combo in product(*value_list):
            assignments_list.append(list(zip(path_list, combo, strict=False)))
    else:
        assignments_list.append([])

    # 実験ディレクトリの準備（JST時刻を使用）
    jst = timezone(timedelta(hours=9))
    timestamp = datetime.now(jst).strftime("%Y%m%d_%H%M%S")
    experiment_cfg = base_cfg.get("experiment", {})
    experiment_name = experiment_cfg.get("name", "exp")
    output_root = Path(experiment_cfg.get("output_dir", "./output")).expanduser()
    experiment_dir = output_root / f"{timestamp}_{sanitize_name(str(experiment_name))}"
    ensure_dir(experiment_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 全ランの学習実行
    summaries: list[dict[str, Any]] = []

    for idx, assignments in enumerate(assignments_list, start=1):
        run_cfg = apply_sweep_values(base_cfg, assignments)
        descriptor = make_run_descriptor(assignments)
        run_name = create_run_name(idx, descriptor) if descriptor else f"{idx:03d}"
        run_dir = experiment_dir / run_name
        print(f"[Run {idx}/{len(assignments_list)}] {run_dir.name}")

        summary = train_single_run(run_cfg, run_dir, descriptor, device=device)
        summary["run_number"] = idx
        summary["run_name"] = run_dir.name
        summaries.append(summary)

    # ベストランの判定
    best_summary: dict[str, Any] | None = None
    if summaries:
        best_summary = min(summaries, key=lambda x: x["best_val_loss"])

    # サマリーの保存
    summary_csv = experiment_dir / "summary.csv"
    if summaries:
        # best_val_lossの昇順でソート
        summaries_sorted = sorted(summaries, key=lambda x: x["best_val_loss"])

        # カラム順序のカスタマイズ: 重要な指標を先頭、augmentationを最後に
        all_keys = {key for item in summaries_sorted for key in item.keys()}
        priority_keys = ["best_val_loss", "best_val_loss_epoch", "run_name", "run_number"]
        augmentation_keys = sorted([k for k in all_keys if k.startswith("augmentation/")])
        other_keys = sorted([k for k in all_keys if k not in priority_keys and not k.startswith("augmentation/")])
        fieldnames = [k for k in priority_keys if k in all_keys] + other_keys + augmentation_keys
        with summary_csv.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for row in summaries_sorted:
                writer.writerow(row)

        summary_payload: dict[str, Any] = {"runs": summaries}
        if best_summary is not None:
            summary_payload["best_run"] = {
                "run_name": best_summary.get("run_name"),
                "best_val_loss": best_summary.get("best_val_loss"),
                "best_val_loss_epoch": best_summary.get("best_val_loss_epoch"),
            }
        with (experiment_dir / "summary.json").open("w", encoding="utf-8") as fp:
            json.dump(summary_payload, fp, ensure_ascii=False, indent=2)

    # 結果の表示
    print("====================================")
    print(f"Experiment outputs saved to: {experiment_dir}")
    if summaries:
        print(f"Summary: {summary_csv}")
        if best_summary is not None:
            print(
                f"Best val_loss run: {best_summary.get('run_name')} "
                f"(val_loss={best_summary.get('best_val_loss'):.6f} at epoch {best_summary.get('best_val_loss_epoch')})"
            )


if __name__ == "__main__":
    main()
