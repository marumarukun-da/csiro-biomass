import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def compute_class_weights_from_counts(class_sample_counts: list[int]) -> np.ndarray:
    """サンプル数から重みを計算する。

    各クラスのサンプル数の逆数に基づいて重みを計算し、
    重みの平均が1になるように正規化する。

    Args:
        class_sample_counts: 各クラスのサンプル数のリスト

    Returns:
        np.ndarray: 正規化されたクラス重み
    """
    counts = np.array(class_sample_counts)
    weights = 1.0 / (counts + 1e-6)
    weights = weights * (len(counts) / weights.sum())
    return weights


class DenseCrossEntropy(nn.Module):
    """ワンホットエンコードされたターゲットに対応したクロスエントロピー損失"""

    def forward(self, x, target, weights=None):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)

        if weights is not None:
            loss = loss * weights

        return loss.mean()


class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    """適応的マージンとクラス重み付きのArcFace損失関数"""

    def __init__(
        self,
        margins,
        n_classes: int,
        s: float = 30.0,
        device: torch.device | None = None,
        label_smoothing: float = 0.0,
        class_weights: np.ndarray | None = None,
    ):
        """
        Args:
            margins: クラス毎のマージン値
            n_classes: クラス数
            s: スケーリングパラメータ
            device: 使用デバイス
            label_smoothing: ラベルスムージングの割合
            class_weights: クラス重み（Noneの場合は重み付けなし）
        """
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.margins = margins
        self.out_dim = n_classes
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_smoothing = label_smoothing

        # クラス重みの設定（Noneの場合は重み付けなし）
        self.class_weights: torch.Tensor | None
        if class_weights is not None:
            self.class_weights = torch.from_numpy(class_weights).float().to(self.device)
        else:
            self.class_weights = None

    def forward(self, logits, labels):
        ms = self.margins[labels.detach().cpu().numpy()]
        ms = torch.from_numpy(ms).float().to(self.device)

        cos_m = torch.cos(ms)
        sin_m = torch.sin(ms)
        th = torch.cos(math.pi - ms)
        mm = torch.sin(math.pi - ms) * ms

        # ワンホット --> ラベルスムージング
        y = F.one_hot(labels, self.out_dim).float().to(self.device)
        if self.label_smoothing > 0:
            eps = self.label_smoothing
            y = y * (1.0 - eps) + eps / self.out_dim

        logits = logits.float()
        cosine = logits.clamp(-1.0, 1.0)  # 数値安定のために念のため
        sine = torch.sqrt((1.0 - cosine**2).clamp_min(0))
        phi = cosine * cos_m.view(-1, 1) - sine * sin_m.view(-1, 1)
        phi = torch.where(cosine > th.view(-1, 1), phi, cosine - mm.view(-1, 1))
        output = (y * phi) + ((1.0 - y) * cosine)
        output = output * self.s

        # 各サンプルの重みを取得（class_weightsがNoneの場合は重み付けなし）
        weights = self.class_weights[labels] if self.class_weights is not None else None

        loss = self.crit(output, y, weights)
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """クラス重み付き・ラベル平滑化対応のCrossEntropyLoss"""

    def __init__(
        self,
        label_smoothing: float = 0.0,
        device: torch.device | None = None,
        class_weights: np.ndarray | None = None,
    ):
        """
        Args:
            label_smoothing: ラベルスムージングの割合
            device: 使用デバイス
            class_weights: クラス重み（Noneの場合は重み付けなし）
        """
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # クラス重みの設定（Noneの場合は重み付けなし）
        self.class_weights: torch.Tensor | None
        if class_weights is not None:
            self.class_weights = torch.from_numpy(class_weights).float().to(self.device)
        else:
            self.class_weights = None

        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, labels)
