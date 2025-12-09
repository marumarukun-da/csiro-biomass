"""モデルと損失関数の構築を行うファクトリ関数群。

train.py と infer.py で重複していたモデル構築ロジックを共通化。
"""

from typing import Any

import numpy as np
import torch
from torch import nn

from src.loss_function import ArcFaceLossAdaptiveMargin, WeightedCrossEntropyLoss
from src.model import ClassificationNet


def build_model(
    model_cfg: dict[str, Any],
    arcface_cfg: dict[str, Any],
    num_classes: int,
    device: torch.device,
    pretrained: bool = True,
) -> ClassificationNet:
    """設定からClassificationNetを構築する。

    Args:
        model_cfg: モデル設定（backbone, head, in_chans等）
        arcface_cfg: ArcFace設定（embedding_size, subcenter_num等）
        num_classes: クラス数
        device: 使用デバイス
        pretrained: 事前学習済み重みを使用するか（train: True, infer: False）

    Returns:
        構築されたClassificationNet（deviceに配置済み）

    Raises:
        ValueError: 未知のhead_typeが指定された場合
    """
    model_name = model_cfg.get("backbone")
    if not isinstance(model_name, str):
        raise ValueError("model.backbone は単一の文字列で指定してください。")

    head_type = model_cfg.get("head", "arcface")
    embedding_size = arcface_cfg.get("embedding_size", 512)
    in_chans = model_cfg.get("in_chans", 3)

    if head_type == "arcface":
        net = ClassificationNet(
            model_name=model_name,
            n_classes=num_classes,
            head_type="arcface",
            embedding_size=embedding_size,
            pretrained=pretrained,
            in_chans=in_chans,
            subcenter_num=arcface_cfg.get("subcenter_num", 2),
        )
    elif head_type == "simple":
        net = ClassificationNet(
            model_name=model_name,
            n_classes=num_classes,
            head_type="simple",
            embedding_size=embedding_size,
            pretrained=pretrained,
            in_chans=in_chans,
        )
    else:
        raise ValueError(f"Unknown head type: {head_type}. Must be 'arcface' or 'simple'")

    net.to(device)
    return net


def build_criterion(
    head_type: str,
    arcface_cfg: dict[str, Any],
    trainer_cfg: dict[str, Any],
    class_counts: list[int],
    class_weights: np.ndarray | None,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    """head_typeに基づいて損失関数を構築する。

    Args:
        head_type: "arcface" または "simple"
        arcface_cfg: ArcFace設定（s, m_x, m_y等）
        trainer_cfg: 学習設定（label_smoothing等）
        class_counts: 各クラスのサンプル数
        class_weights: クラス重み（Noneの場合は重み付けなし）
        num_classes: クラス数
        device: 使用デバイス

    Returns:
        構築された損失関数（deviceに配置済み）

    Raises:
        ValueError: 未知のhead_typeが指定された場合
    """
    label_smoothing = trainer_cfg.get("label_smoothing", 0.0)

    if head_type == "arcface":
        # Adaptive Margin の計算
        tmp = np.sqrt(1 / np.sqrt(np.array(class_counts) + 1e-12))
        init_margins = (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-12)
        init_margins = init_margins * arcface_cfg.get("m_x", 0.1) + arcface_cfg.get("m_y", 0.6)

        criterion = ArcFaceLossAdaptiveMargin(
            margins=init_margins,
            n_classes=num_classes,
            s=arcface_cfg.get("s", 30.0),
            device=device,
            label_smoothing=label_smoothing,
            class_weights=class_weights,
        )
    elif head_type == "simple":
        criterion = WeightedCrossEntropyLoss(
            label_smoothing=label_smoothing,
            device=device,
            class_weights=class_weights,
        )
    else:
        raise ValueError(f"Unknown head type: {head_type}. Must be 'arcface' or 'simple'")

    criterion.to(device)
    return criterion
