import math

import timm
import torch
import torch.nn.functional as F
from torch import nn


def load_model(model_path, net, device):
    """モデルを読み込み、評価モードに設定する"""
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)
    net.eval()
    return net


class ArcMarginProductSubcenter(nn.Module):
    """Subcenterを使用したArcMarginProductの定義"""

    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine


class GeM(nn.Module):
    """GeMプーリングの定義"""

    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super().__init__()
        if p_trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.pow(1.0 / self.p)
        return x


def _ensure_bchw(x: torch.Tensor, num_features: int | None) -> torch.Tensor:
    # [B, H, W, C] (Swin等) を [B, C, H, W] に
    if x.dim() == 4 and (num_features is not None) and x.shape[-1] == num_features:
        return x.permute(0, 3, 1, 2).contiguous()
    return x  # その他はそのまま返す


class ClassificationNet(nn.Module):
    """画像分類モデルの定義 (ArcFace / Simple分類ヘッド対応)"""

    def __init__(
        self,
        model_name: str,
        n_classes: int,
        head_type: str = "arcface",
        embedding_size: int = 512,
        pretrained: bool = False,
        in_chans: int = 3,
        subcenter_num: int = 3,
    ):
        """
        Args:
            model_name: timmのバックボーンモデル名
            n_classes: クラス数
            head_type: ヘッドタイプ ("arcface" or "simple")
            embedding_size: 埋め込みベクトルのサイズ
            pretrained: 事前学習済み重みを使用するか
            in_chans: 入力チャンネル数
            subcenter_num: ArcFaceのサブセンター数 (head_type="arcface"のみ)
        """
        super().__init__()
        self.head_type = head_type
        self.embedding_size = embedding_size
        self.n_classes = n_classes

        # backboneの定義
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
        )

        # pooling層の定義
        self.global_pool = GeM(p_trainable=True)

        # neckの定義 (ArcFace / Simple 両方で共通)
        self.num_features = self.backbone.num_features  # backboneの種類によって変数名が異なるので注意
        self.neck = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.num_features, self.embedding_size, bias=True),
            nn.BatchNorm1d(self.embedding_size),
            torch.nn.PReLU(),
        )

        # headの定義 (タイプに応じて切り替え)
        if head_type == "arcface":
            self.head = ArcMarginProductSubcenter(self.embedding_size, self.n_classes, k=subcenter_num)
        elif head_type == "simple":
            self.head = nn.Linear(self.embedding_size, self.n_classes)
        else:
            raise ValueError(f"Unknown head_type: {head_type}. Must be 'arcface' or 'simple'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """推論処理

        Args:
            x (torch.Tensor): 入力画像

        Returns:
            torch.Tensor: logits
        """
        batch_size = x.size(0)

        # backboneから特徴量を抽出
        features = self.backbone.forward_features(x)
        # BHWCならBCHWへ（BCHWならそのまま）
        features = _ensure_bchw(features, self.num_features)

        # global pooling -> neckでembeddingsに変換 -> headでlogitsに変換
        pooled_features = self.global_pool(features).view(batch_size, -1)
        embeddings = self.neck(pooled_features)
        logits = self.head(embeddings)

        return logits
