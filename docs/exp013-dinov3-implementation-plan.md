# Exp013 DINOv3 Dense Features 実装計画書

---

## AIエージェント向け実装プロンプト

以下のプロンプトをそのままコピーして、新しいAIエージェントに渡してください：

```
あなたはKaggleコンペティション「CSIRO Biomass Prediction」の実験コード（exp013）を実装するタスクを担当します。

## 背景
現在のexp013は ConvNeXtV2 ベースの実装ですが、これを DINOv3 Dense Features ベースのアーキテクチャに変更します。
参考となるアプローチは `docs/lb-066-dinov2-dense-features-architecture.md` に記載されています。

## あなたのタスク
`docs/exp013-dinov3-implementation-plan.md` に記載された実装計画に基づいて、以下のファイルを新規作成または変更してください：

### 新規作成ファイル
1. `experiments/013/src/backbone.py` - DINOv3バックボーン定義
2. `experiments/013/src/head_model.py` - CLS Gating + マルチヘッドモデル
3. `experiments/013/src/feature_dataset.py` - 事前抽出ベクトル用Dataset
4. `experiments/013/src/manifold_mixup.py` - Feature-level Mixup
5. `experiments/013/extract_features.py` - ベクトル事前抽出スクリプト
6. `experiments/013/train_head.py` - Headのみ学習スクリプト
7. `experiments/013/configs/exp/exp013_dinov3.yaml` - 設定ファイル

### 変更ファイル
1. `experiments/013/src/loss_function.py` - 補助ロス（Dead, Clover）追加
2. `experiments/013/inference.py` - バックボーン抽出込みの推論

## 重要な仕様
- バックボーン: `timm/vit_huge_plus_patch16_dinov3.lvd1689m` (Frozen)
- 入力サイズ: 960×960 (16の倍数制約)
- 出力構造: [CLS(1), レジスタ(4), パッチ(3600)] × 1280次元
- レジスタトークンは無視する
- CLSトークンはGating方式でパッチの重要度学習に使用
- Head設計: 共有表現 + マルチヘッド出力
- Mixup: Feature-level (ManifoldMixup)
- 学習時: 事前抽出ベクトルを使用（20パターン/画像）
- 推論時: バックボーンでリアルタイム抽出（TTA 4種）

詳細な実装仕様は `docs/exp013-dinov3-implementation-plan.md` を参照してください。
```

---

## 1. プロジェクト概要

### 1.1 目的
CSIRO Biomass Predictionコンペティションにおいて、DINOv3の Dense Features を活用した回帰モデルを構築する。

### 1.2 参考アプローチ
- LB 0.66 を達成した `notebook/csiro-dinov2-dense-features-lb-0-66.ipynb` のアプローチ
- 詳細解説: `docs/lb-066-dinov2-dense-features-architecture.md`

### 1.3 現行exp013との主な違い

| 項目 | 現行exp013 | 新実装 |
|------|-----------|--------|
| バックボーン | ConvNeXtV2-Base（学習あり） | DINOv3-Huge（Frozen） |
| 入力形式 | 左右分割 Dual Input | 単一画像 |
| 入力サイズ | 640×640 | 960×960 |
| 特徴抽出 | CNN + GeM Pooling | Dense Patch Features |
| CLSトークン | N/A | Gating方式で活用 |
| 学習対象 | 全モデル | Headのみ |
| 効率化 | なし | ベクトル事前抽出 |

---

## 2. 確定要件

### 2.1 バックボーン仕様

```python
モデル名: vit_huge_plus_patch16_dinov3.lvd1689m
入力サイズ: 960×960 (16の倍数制約)
パッチサイズ: 16×16
隠れ層次元: 1280
出力シーケンス: [1, 3605, 1280]
  - [0]: CLSトークン
  - [1:5]: レジスタトークン (4個) ← 無視
  - [5:]: パッチトークン (3600個 = 60×60)
```

### 2.2 タスク構成

| タスク | 出力次元 | 損失関数 | 重み |
|--------|---------|---------|------|
| **Main: Total** | 1 | Smooth L1 | 1.0 |
| **Main: GDM** | 1 | Smooth L1 | 0.6 |
| **Main: Green** | 1 | Smooth L1 | 0.3 |
| Aux: State | 4 (分類) | CrossEntropy | 0.1 |
| Aux: Height | 1 | Smooth L1 | 0.1 |
| Aux: Dead | 1 | Smooth L1 | 0.1 |
| Aux: Clover | 1 | Smooth L1 | 0.1 |

### 2.3 Augmentation戦略

**ベクトル抽出時（20パターン/画像）:**
- Flip系: 4種 (original, hflip, vflip, hvflip)
- 各Flip × 色変換: 4種 (brightness, contrast, hsv, gamma)
- → 4 + 4×4 = 20パターン

**学習時:**
- Feature-level Mixup (ManifoldMixup, alpha=1)
- 各エポックでランダムにAugパターンを選択

**推論時（TTA）:**
- 4種: original, hflip, vflip, hvflip

### 2.4 その他の仕様

- ベクトル保存形式: NumPy npz
- 後処理: なし（State予測に基づく補正なし）
- リサイズ方式: cv2.INTER_AREA

---

## 3. アーキテクチャ設計

### 3.1 全体フロー

```
┌─────────────────────────────────────────────────────────────────┐
│                 Phase 1: Feature Extraction (Offline)            │
│                                                                  │
│  Train画像 ─→ Augmentation(20種) ─→ DINOv3 Backbone ─→ .npz保存  │
│              (960×960)              (Frozen)                     │
│                                                                  │
│  保存内容: cls_token [1280], patch_tokens [3600, 1280]          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   Phase 2: Head Training (Online)                │
│                                                                  │
│  .npz ─→ ランダムAug選択 ─→ Feature Mixup ─→ Head ─→ Loss       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   Phase 3: Inference (Online)                    │
│                                                                  │
│  Test画像 ─→ TTA(4種) ─→ DINOv3 Backbone ─→ Head ─→ Ensemble    │
│                          (リアルタイム抽出)                       │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Head Network アーキテクチャ

```
Input:
  cls_token: [B, 1280]
  patch_tokens: [B, 3600, 1280]

┌──────────────────────────────────────────────────────────────┐
│                    CLS Gating Module                          │
│                                                               │
│  cls_token ─→ MLP(1280→320→3600) ─→ softmax ─→ attn_weights  │
│                                                               │
│  patch_tokens × attn_weights ─→ sum ─→ weighted_pool [B,1280]│
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                  Shared Representation                        │
│                                                               │
│  concat([cls_token, weighted_pool]) ─→ [B, 2560]             │
│           ↓                                                   │
│  Linear(2560→512) ─→ LayerNorm ─→ GELU ─→ Dropout(0.1)       │
│           ↓                                                   │
│  shared_features [B, 512]                                    │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                   Multi-Head Outputs                          │
│                                                               │
│  shared_features ─┬─→ main_head ─→ Softplus ─→ [Total,GDM,Green] │
│                   ├─→ state_head ─→ [logits×4]                │
│                   ├─→ height_head ─→ Softplus ─→ [height]     │
│                   └─→ aux_head ─→ Softplus ─→ [Dead,Clover]   │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. ファイル構成

```
experiments/013/
├── extract_features.py      # [新規] ベクトル事前抽出スクリプト
├── train_head.py            # [新規] Headのみ学習スクリプト
├── inference.py             # [変更] 推論スクリプト（バックボーン抽出込み）
├── train.py                 # [既存] 参考用（変更不要）
├── config.py                # [既存] パス設定
├── src/
│   ├── backbone.py          # [新規] DINOv3バックボーン定義
│   ├── head_model.py        # [新規] CLS Gating + MultiHead
│   ├── feature_dataset.py   # [新規] 事前抽出ベクトル用Dataset
│   ├── manifold_mixup.py    # [新規] Feature-level Mixup
│   ├── loss_function.py     # [変更] 補助ロス追加
│   ├── model.py             # [既存] 参考用（変更不要）
│   ├── data.py              # [既存] 参考用（変更不要）
│   ├── metric.py            # [既存] そのまま使用
│   ├── seed.py              # [既存] そのまま使用
│   └── ...
└── configs/
    └── exp/
        ├── exp013.yaml          # [既存] 現行設定
        └── exp013_dinov3.yaml   # [新規] DINOv3用設定
```

---

## 5. 各コンポーネント詳細仕様

### 5.1 backbone.py

```python
"""DINOv3 Backbone for feature extraction."""

import timm
import torch
from torch import nn


class DINOv3Backbone(nn.Module):
    """DINOv3 Backbone for extracting CLS and patch features.

    Output structure for 960x960 input:
        - Total sequence: [B, 3605, 1280]
        - CLS token: output[:, 0, :]  -> [B, 1280]
        - Register tokens: output[:, 1:5, :] -> [B, 4, 1280] (ignored)
        - Patch tokens: output[:, 5:, :] -> [B, 3600, 1280]
    """

    MODEL_NAME = "vit_huge_plus_patch16_dinov3.lvd1689m"
    PATCH_SIZE = 16
    HIDDEN_DIM = 1280
    NUM_REGISTERS = 4

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(
            self.MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @property
    def hidden_dim(self) -> int:
        return self.HIDDEN_DIM

    @property
    def num_registers(self) -> int:
        return self.NUM_REGISTERS

    def get_num_patches(self, img_size: int) -> int:
        """Calculate number of patches for given image size."""
        assert img_size % self.PATCH_SIZE == 0
        num_patches_per_side = img_size // self.PATCH_SIZE
        return num_patches_per_side * num_patches_per_side

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract CLS and patch features.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Tuple of:
            - cls_token: [B, 1280]
            - patch_tokens: [B, num_patches, 1280]
        """
        output = self.model(x)
        cls_token = output[:, 0, :]
        patch_tokens = output[:, 1 + self.NUM_REGISTERS:, :]
        return cls_token, patch_tokens

    def train(self, mode: bool = True):
        """Always keep in eval mode (frozen)."""
        return super().train(False)


def build_backbone(pretrained: bool = True, device: str = "cuda") -> DINOv3Backbone:
    backbone = DINOv3Backbone(pretrained=pretrained)
    return backbone.to(device)
```

### 5.2 head_model.py

```python
"""CLS Gating Head Model for DINOv3 features."""

import torch
import torch.nn.functional as F
from torch import nn


class CLSGatingHead(nn.Module):
    """Head model with CLS-based gating and multi-task outputs.

    Uses CLS token to compute attention weights over patch tokens,
    then combines CLS and weighted patch features for prediction.
    """

    def __init__(
        self,
        hidden_dim: int = 1280,
        num_patches: int = 3600,
        shared_hidden: int = 512,
        num_main_outputs: int = 3,  # Total, GDM, Green
        num_states: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches

        # CLS Gating: cls -> attention weights over patches
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_patches),
        )

        # Shared representation: concat(cls, weighted_pool) -> shared features
        self.shared_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, shared_hidden),
            nn.LayerNorm(shared_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Output heads
        self.main_head = nn.Linear(shared_hidden, num_main_outputs)  # Total, GDM, Green
        self.state_head = nn.Linear(shared_hidden, num_states)  # State classification
        self.height_head = nn.Linear(shared_hidden, 1)  # Height regression
        self.aux_head = nn.Linear(shared_hidden, 2)  # Dead, Clover

    def forward(
        self, cls_token: torch.Tensor, patch_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            cls_token: [B, hidden_dim]
            patch_tokens: [B, num_patches, hidden_dim]

        Returns:
            Tuple of:
            - main_out: [B, 3] (Total, GDM, Green) - non-negative
            - state_out: [B, 4] logits
            - height_out: [B, 1] - non-negative
            - aux_out: [B, 2] (Dead, Clover) - non-negative
        """
        # CLS Gating: compute attention weights
        attn_weights = F.softmax(self.gate_mlp(cls_token), dim=-1)  # [B, num_patches]

        # Weighted pooling of patch tokens
        weighted_pool = (patch_tokens * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, hidden_dim]

        # Shared representation
        combined = torch.cat([cls_token, weighted_pool], dim=-1)  # [B, hidden_dim * 2]
        shared = self.shared_mlp(combined)  # [B, shared_hidden]

        # Output heads with non-negative constraints
        main_out = F.softplus(self.main_head(shared))
        state_out = self.state_head(shared)  # raw logits
        height_out = F.softplus(self.height_head(shared))
        aux_out = F.softplus(self.aux_head(shared))

        return main_out, state_out, height_out, aux_out


def build_head_model(
    hidden_dim: int = 1280,
    num_patches: int = 3600,
    shared_hidden: int = 512,
    dropout: float = 0.1,
    device: str = "cuda",
) -> CLSGatingHead:
    model = CLSGatingHead(
        hidden_dim=hidden_dim,
        num_patches=num_patches,
        shared_hidden=shared_hidden,
        dropout=dropout,
    )
    return model.to(device)
```

### 5.3 feature_dataset.py

```python
"""Dataset for precomputed DINOv3 features."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# State mapping (same as existing)
STATE_TO_IDX = {"Tas": 0, "NSW": 1, "WA": 2, "Vic": 3}


class PrecomputedFeatureDataset(Dataset):
    """Dataset that loads precomputed CLS and patch features from .npz files.

    Each .npz file contains features for multiple augmentation patterns:
        - cls_0, cls_1, ..., cls_19: CLS tokens for each augmentation
        - patches_0, patches_1, ..., patches_19: Patch tokens for each augmentation

    During training, randomly selects one augmentation pattern per sample.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_dir: Path | str,
        num_aug_patterns: int = 20,
        target_cols: list[str] | None = None,
        aux_target_cols: list[str] | None = None,
        state_col: str = "State",
        height_col: str = "Height_Ave_cm",
        is_train: bool = True,
    ):
        """Initialize dataset.

        Args:
            df: DataFrame with image metadata and targets (Wide format).
            feature_dir: Directory containing .npz feature files.
            num_aug_patterns: Number of augmentation patterns per image.
            target_cols: Main target columns [Dry_Total_g, GDM_g, Dry_Green_g].
            aux_target_cols: Auxiliary target columns [Dry_Dead_g, Dry_Clover_g].
            state_col: Column name for state classification.
            height_col: Column name for height regression.
            is_train: Whether this is training data.
        """
        self.df = df.reset_index(drop=True)
        self.feature_dir = Path(feature_dir)
        self.num_aug_patterns = num_aug_patterns
        self.target_cols = target_cols or ["Dry_Total_g", "GDM_g", "Dry_Green_g"]
        self.aux_target_cols = aux_target_cols or ["Dry_Dead_g", "Dry_Clover_g"]
        self.state_col = state_col
        self.height_col = height_col
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_id = row["image_id"]

        # Load features from .npz
        feature_path = self.feature_dir / f"{image_id}.npz"
        data = np.load(feature_path)

        # Select augmentation pattern
        if self.is_train:
            aug_idx = random.randint(0, self.num_aug_patterns - 1)
        else:
            aug_idx = 0  # Use original for validation

        cls_token = torch.tensor(data[f"cls_{aug_idx}"], dtype=torch.float32)
        patch_tokens = torch.tensor(data[f"patches_{aug_idx}"], dtype=torch.float32)

        result = {
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
            "image_id": image_id,
        }

        # Add targets
        if self.is_train:
            # Main targets
            main_targets = [row[col] for col in self.target_cols]
            result["main_targets"] = torch.tensor(main_targets, dtype=torch.float32)

            # Auxiliary targets (Dead, Clover)
            aux_targets = [row[col] for col in self.aux_target_cols]
            result["aux_targets"] = torch.tensor(aux_targets, dtype=torch.float32)

            # State label
            if self.state_col in row:
                result["state_label"] = STATE_TO_IDX[row[self.state_col]]

            # Height value
            if self.height_col in row:
                result["height_value"] = float(row[self.height_col])

        return result


class InferenceFeatureDataset(Dataset):
    """Dataset for inference with on-the-fly feature extraction."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_dir: Path | str,
        tta_indices: list[int] | None = None,
    ):
        """Initialize dataset.

        Args:
            df: DataFrame with image metadata.
            feature_dir: Directory containing .npz feature files.
            tta_indices: Indices of TTA augmentations to use (default: [0,1,2,3]).
        """
        self.df = df.reset_index(drop=True)
        self.feature_dir = Path(feature_dir)
        self.tta_indices = tta_indices or [0, 1, 2, 3]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_id = row["image_id"]

        feature_path = self.feature_dir / f"{image_id}.npz"
        data = np.load(feature_path)

        # Load all TTA features
        cls_tokens = []
        patch_tokens_list = []
        for tta_idx in self.tta_indices:
            cls_tokens.append(data[f"cls_{tta_idx}"])
            patch_tokens_list.append(data[f"patches_{tta_idx}"])

        return {
            "cls_tokens": torch.tensor(np.stack(cls_tokens), dtype=torch.float32),
            "patch_tokens": torch.tensor(np.stack(patch_tokens_list), dtype=torch.float32),
            "image_id": image_id,
        }
```

### 5.4 manifold_mixup.py

```python
"""Feature-level Mixup (ManifoldMixup) implementation."""

import numpy as np
import torch
from torch import Tensor


def feature_mixup(
    cls_a: Tensor,
    patches_a: Tensor,
    targets_a: dict[str, Tensor],
    cls_b: Tensor,
    patches_b: Tensor,
    targets_b: dict[str, Tensor],
    alpha: float = 1.0,
) -> tuple[Tensor, Tensor, dict[str, Tensor], Tensor]:
    """Apply ManifoldMixup to precomputed features.

    Args:
        cls_a: CLS tokens for samples A [B, D]
        patches_a: Patch tokens for samples A [B, N, D]
        targets_a: Dict of target tensors for samples A
        cls_b: CLS tokens for samples B [B, D]
        patches_b: Patch tokens for samples B [B, N, D]
        targets_b: Dict of target tensors for samples B
        alpha: Beta distribution parameter

    Returns:
        Tuple of:
        - mixed_cls: Mixed CLS tokens [B, D]
        - mixed_patches: Mixed patch tokens [B, N, D]
        - mixed_targets: Dict of mixed target tensors
        - lam: Lambda values [B]
    """
    batch_size = cls_a.size(0)
    device = cls_a.device

    # Sample lambda from Beta distribution (per-sample)
    lam = np.random.beta(alpha, alpha, size=batch_size)
    lam = torch.tensor(lam, dtype=torch.float32, device=device)

    # Mix features
    lam_feat = lam.view(-1, 1)  # [B, 1]
    mixed_cls = lam_feat * cls_a + (1 - lam_feat) * cls_b

    lam_patch = lam.view(-1, 1, 1)  # [B, 1, 1]
    mixed_patches = lam_patch * patches_a + (1 - lam_patch) * patches_b

    # Mix targets
    mixed_targets = {}
    for key in targets_a:
        if targets_a[key].dim() == 1:
            # Scalar targets (e.g., height)
            mixed_targets[key] = lam * targets_a[key] + (1 - lam) * targets_b[key]
        else:
            # Vector targets (e.g., main_targets, aux_targets)
            lam_target = lam.view(-1, 1)
            mixed_targets[key] = lam_target * targets_a[key] + (1 - lam_target) * targets_b[key]

    return mixed_cls, mixed_patches, mixed_targets, lam


def mixup_batch(
    batch: dict[str, Tensor],
    alpha: float = 2.5,
) -> dict[str, Tensor]:
    """Apply ManifoldMixup to a batch (shuffle within batch).

    Args:
        batch: Dict containing cls_token, patch_tokens, and targets
        alpha: Beta distribution parameter

    Returns:
        Mixed batch dict
    """
    batch_size = batch["cls_token"].size(0)
    device = batch["cls_token"].device

    # Shuffle indices
    indices = torch.randperm(batch_size, device=device)

    # Get shuffled samples
    cls_b = batch["cls_token"][indices]
    patches_b = batch["patch_tokens"][indices]

    targets_a = {
        "main_targets": batch["main_targets"],
        "aux_targets": batch["aux_targets"],
    }
    targets_b = {
        "main_targets": batch["main_targets"][indices],
        "aux_targets": batch["aux_targets"][indices],
    }

    # Add optional targets
    if "height_value" in batch:
        targets_a["height_value"] = batch["height_value"]
        targets_b["height_value"] = batch["height_value"][indices]

    # Apply mixup
    mixed_cls, mixed_patches, mixed_targets, lam = feature_mixup(
        batch["cls_token"],
        batch["patch_tokens"],
        targets_a,
        cls_b,
        patches_b,
        targets_b,
        alpha=alpha,
    )

    # Handle state labels (soft labels for mixup)
    if "state_label" in batch:
        num_states = 4
        state_onehot = torch.zeros(batch_size, num_states, device=device)
        state_onehot.scatter_(1, batch["state_label"].unsqueeze(1), 1.0)
        state_onehot_b = state_onehot[indices]
        lam_cls = lam.view(-1, 1)
        mixed_targets["state_label"] = lam_cls * state_onehot + (1 - lam_cls) * state_onehot_b

    result = {
        "cls_token": mixed_cls,
        "patch_tokens": mixed_patches,
        **mixed_targets,
        "lam": lam,
        "indices": indices,
    }

    return result
```

### 5.5 loss_function.py (追加部分)

```python
"""Extended loss function with Dead/Clover auxiliary loss."""

# 既存のMultiTaskBiomassLossに追加して、新しいクラスを作成

class DINOv3MultiTaskLoss(nn.Module):
    """Multi-task loss for DINOv3 head with all auxiliary tasks.

    Tasks:
    - Main: Total, GDM, Green (Smooth L1, weighted)
    - Aux1: State classification (CrossEntropy)
    - Aux2: Height regression (Smooth L1)
    - Aux3: Dead, Clover regression (Smooth L1)
    """

    def __init__(
        self,
        beta: float = 1.0,
        main_weights: list[float] | None = None,
        state_weight: float = 0.1,
        height_weight: float = 0.1,
        aux_weight: float = 0.1,
    ):
        """Initialize loss function.

        Args:
            beta: Smooth L1 beta parameter.
            main_weights: Weights for [Total, GDM, Green]. Default: [1.0, 0.6, 0.3]
            state_weight: Weight for state classification loss.
            height_weight: Weight for height regression loss.
            aux_weight: Weight for Dead/Clover auxiliary loss.
        """
        super().__init__()
        self.beta = beta
        self.main_weights = main_weights or [1.0, 0.6, 0.3]
        self.state_weight = state_weight
        self.height_weight = height_weight
        self.aux_weight = aux_weight

    def forward(
        self,
        main_pred: torch.Tensor,
        state_pred: torch.Tensor,
        height_pred: torch.Tensor,
        aux_pred: torch.Tensor,
        main_target: torch.Tensor,
        state_target: torch.Tensor,
        height_target: torch.Tensor,
        aux_target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate combined multi-task loss.

        Args:
            main_pred: [B, 3] predictions for Total, GDM, Green
            state_pred: [B, 4] logits for State classification
            height_pred: [B, 1] prediction for Height
            aux_pred: [B, 2] predictions for Dead, Clover
            main_target: [B, 3] targets for Total, GDM, Green
            state_target: [B] indices or [B, 4] soft labels for State
            height_target: [B] or [B, 1] target for Height
            aux_target: [B, 2] targets for Dead, Clover

        Returns:
            Combined loss scalar.
        """
        # Main loss: weighted Smooth L1
        main_loss = 0.0
        for i, w in enumerate(self.main_weights):
            main_loss += w * F.smooth_l1_loss(
                main_pred[:, i], main_target[:, i], beta=self.beta
            )

        # State classification loss
        if state_target.dim() == 1:
            state_loss = F.cross_entropy(state_pred, state_target)
        else:
            # Soft labels (for Mixup)
            log_probs = F.log_softmax(state_pred, dim=1)
            state_loss = -(state_target * log_probs).sum(dim=1).mean()

        # Height regression loss
        height_target_flat = height_target.view(-1)
        height_pred_flat = height_pred.view(-1)
        height_loss = F.smooth_l1_loss(height_pred_flat, height_target_flat, beta=self.beta)

        # Auxiliary loss: Dead, Clover
        aux_loss = F.smooth_l1_loss(aux_pred, aux_target, beta=self.beta)

        # Combined loss
        total_loss = (
            main_loss
            + self.state_weight * state_loss
            + self.height_weight * height_loss
            + self.aux_weight * aux_loss
        )

        return total_loss
```

### 5.6 extract_features.py

```python
"""Script to extract and save DINOv3 features for all training images."""

import argparse
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Import after config setup
import config
from src.backbone import build_backbone
from src.data import convert_long_to_wide


def build_augmentations() -> list[tuple[str, A.Compose]]:
    """Build 20 augmentation patterns.

    Pattern structure:
    - 0: original
    - 1: hflip
    - 2: vflip
    - 3: hvflip
    - 4-7: original + color variants
    - 8-11: hflip + color variants
    - 12-15: vflip + color variants
    - 16-19: hvflip + color variants
    """
    # Base flip transforms
    flip_transforms = [
        ("original", A.Compose([])),
        ("hflip", A.Compose([A.HorizontalFlip(p=1.0)])),
        ("vflip", A.Compose([A.VerticalFlip(p=1.0)])),
        ("hvflip", A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)])),
    ]

    # Color transforms
    color_transforms = [
        ("brightness", A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=1.0)),
        ("contrast", A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2, p=1.0)),
        ("hsv", A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0)),
        ("gamma", A.RandomGamma(gamma_limit=(80, 120), p=1.0)),
    ]

    augmentations = []

    # First 4: flip only
    for flip_name, flip_transform in flip_transforms:
        augmentations.append((flip_name, flip_transform))

    # Remaining 16: flip + color
    for flip_name, flip_transform in flip_transforms:
        for color_name, color_transform in color_transforms:
            combined = A.Compose([
                *flip_transform.transforms,
                color_transform,
            ])
            augmentations.append((f"{flip_name}_{color_name}", combined))

    return augmentations


def extract_features_for_image(
    image_path: Path,
    backbone: torch.nn.Module,
    augmentations: list[tuple[str, A.Compose]],
    img_size: int = 960,
    device: torch.device = torch.device("cuda"),
) -> dict[str, np.ndarray]:
    """Extract features for a single image with all augmentations.

    Args:
        image_path: Path to image file.
        backbone: DINOv3 backbone model.
        augmentations: List of (name, transform) tuples.
        img_size: Target image size (must be divisible by 16).
        device: Device for inference.

    Returns:
        Dict with cls_i and patches_i for each augmentation i.
    """
    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to target size
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # Normalize parameters (ImageNet)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    features = {}

    for i, (aug_name, transform) in enumerate(augmentations):
        # Apply augmentation
        augmented = transform(image=image)["image"]

        # Normalize and convert to tensor
        img_normalized = (augmented / 255.0 - mean) / std
        img_tensor = torch.tensor(img_normalized, dtype=torch.float32)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        # Extract features
        cls_token, patch_tokens = backbone(img_tensor)

        # Store as numpy arrays
        features[f"cls_{i}"] = cls_token.cpu().numpy().squeeze(0)
        features[f"patches_{i}"] = patch_tokens.cpu().numpy().squeeze(0)

    return features


def main():
    parser = argparse.ArgumentParser(description="Extract DINOv3 features")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for features")
    parser.add_argument("--img_size", type=int, default=960, help="Image size")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (currently only 1 supported)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load backbone
    print("Loading DINOv3 backbone...")
    backbone = build_backbone(pretrained=True, device=device)

    # Build augmentations
    augmentations = build_augmentations()
    print(f"Using {len(augmentations)} augmentation patterns")

    # Load training data
    train_csv = config.get_train_csv_path()
    train_df = pd.read_csv(train_csv)
    train_df = convert_long_to_wide(train_df)

    image_dir = config.get_image_dir()

    # Extract features for each image
    print(f"Extracting features for {len(train_df)} images...")
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        image_id = row["image_id"]
        image_path = image_dir / row["image_path"]

        output_path = output_dir / f"{image_id}.npz"

        # Skip if already exists
        if output_path.exists():
            continue

        # Extract features
        features = extract_features_for_image(
            image_path=image_path,
            backbone=backbone,
            augmentations=augmentations,
            img_size=args.img_size,
            device=device,
        )

        # Save as compressed npz
        np.savez_compressed(output_path, **features)

    print(f"Features saved to {output_dir}")


if __name__ == "__main__":
    main()
```

### 5.7 train_head.py

```python
"""Training script for DINOv3 head model using precomputed features."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.model_selection import GroupKFold
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

import config
from src.data import convert_long_to_wide
from src.feature_dataset import PrecomputedFeatureDataset
from src.head_model import build_head_model
from src.loss_function import DINOv3MultiTaskLoss
from src.manifold_mixup import mixup_batch
from src.metric import weighted_r2_score_full
from src.seed import seed_everything


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    use_amp,
    use_mixup,
    mixup_alpha,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_samples = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        # Move to device
        cls_token = batch["cls_token"].to(device)
        patch_tokens = batch["patch_tokens"].to(device)
        main_targets = batch["main_targets"].to(device)
        aux_targets = batch["aux_targets"].to(device)
        state_labels = torch.tensor(batch["state_label"], device=device)
        height_values = torch.tensor(batch["height_value"], dtype=torch.float32, device=device)

        # Prepare batch dict for mixup
        batch_dict = {
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
            "main_targets": main_targets,
            "aux_targets": aux_targets,
            "state_label": state_labels,
            "height_value": height_values,
        }

        # Apply Mixup if enabled
        if use_mixup:
            batch_dict = mixup_batch(batch_dict, alpha=mixup_alpha)

        with autocast(device_type=device.type, enabled=use_amp):
            main_pred, state_pred, height_pred, aux_pred = model(
                batch_dict["cls_token"], batch_dict["patch_tokens"]
            )

            loss = criterion(
                main_pred,
                state_pred,
                height_pred,
                aux_pred,
                batch_dict["main_targets"],
                batch_dict["state_label"],
                batch_dict["height_value"],
                batch_dict["aux_targets"],
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        batch_size = cls_token.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

    return total_loss / num_samples


@torch.no_grad()
def validate(model, valid_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_samples = 0
    all_preds = []
    all_targets = []

    for batch in tqdm(valid_loader, desc="Validation", leave=False):
        cls_token = batch["cls_token"].to(device)
        patch_tokens = batch["patch_tokens"].to(device)
        main_targets = batch["main_targets"].to(device)
        aux_targets = batch["aux_targets"].to(device)
        state_labels = torch.tensor(batch["state_label"], device=device)
        height_values = torch.tensor(batch["height_value"], dtype=torch.float32, device=device)

        main_pred, state_pred, height_pred, aux_pred = model(cls_token, patch_tokens)

        loss = criterion(
            main_pred,
            state_pred,
            height_pred,
            aux_pred,
            main_targets,
            state_labels,
            height_values,
            aux_targets,
        )

        batch_size = cls_token.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

        all_preds.append(main_pred.cpu().numpy())
        all_targets.append(main_targets.cpu().numpy())

    val_loss = total_loss / num_samples
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    val_r2 = weighted_r2_score_full(all_targets, all_preds)

    return val_loss, val_r2


def main():
    parser = argparse.ArgumentParser(description="Train DINOv3 head model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Setup
    seed_everything(cfg.get("experiment", {}).get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ... (rest of training loop similar to existing train.py)
    # See existing train.py for full implementation pattern


if __name__ == "__main__":
    main()
```

### 5.8 exp013_dinov3.yaml

```yaml
experiment:
  name: exp013_dinov3
  seed: 42

backbone:
  model_name: vit_huge_plus_patch16_dinov3.lvd1689m
  pretrained: true
  freeze: true
  hidden_dim: 1280
  num_patches: 3600  # 60x60 for 960x960 input

dataset:
  img_size: 960
  resize_method: INTER_AREA
  feature_dir: /path/to/precomputed_features  # Update this path
  num_aug_patterns: 20
  target_cols:
    - Dry_Total_g
    - GDM_g
    - Dry_Green_g
  aux_target_cols:
    - Dry_Dead_g
    - Dry_Clover_g

head:
  shared_hidden: 512
  dropout: 0.1

loss:
  beta: 1.0
  main_weights:
    - 1.0  # Total
    - 0.6  # GDM
    - 0.3  # Green
  state_weight: 0.1
  height_weight: 0.1
  aux_weight: 0.1  # Dead, Clover

trainer:
  num_epochs: 200
  batch_size: 32  # Can be larger since only training head
  num_workers: 4
  use_amp: true
  n_folds: 5
  fold: null  # Train all folds
  group_col: site

optimization:
  lr: 1e-4
  weight_decay: 1e-2
  warmup_rate: 0.1
  use_ema: true
  ema_decay: 0.997
  ema_start_ratio: 0.025

augmentation:
  mixup:
    enabled: true
    alpha: 2.5
    disable_ratio: 0.2  # Disable in last 20% of training

inference:
  tta_indices:
    - 0  # original
    - 1  # hflip
    - 2  # vflip
    - 3  # hvflip
```

---

## 6. 推論スクリプトの変更点

### 6.1 inference.py の主な変更

```python
# 主な変更点:
# 1. DINOv3 backboneのロードと使用
# 2. リアルタイムでの特徴抽出
# 3. Head modelのロード
# 4. TTA処理の更新

def inference_with_backbone(
    test_df: pd.DataFrame,
    backbone: nn.Module,
    head_models: list[nn.Module],
    image_dir: Path,
    device: torch.device,
    img_size: int = 960,
) -> dict[str, np.ndarray]:
    """Run inference with backbone feature extraction."""

    predictions = {}
    tta_transforms = build_tta_transforms()  # 4種: original, hflip, vflip, hvflip

    for image_path in tqdm(test_df["image_path"].unique()):
        # Load and preprocess image
        image = cv2.imread(str(image_dir / image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)

        all_preds = []

        for tta_transform in tta_transforms:
            # Apply TTA
            aug_image = tta_transform(image=image)["image"]

            # Normalize and convert to tensor
            img_tensor = preprocess_image(aug_image).to(device)

            # Extract features
            cls_token, patch_tokens = backbone(img_tensor)

            # Predict with each fold model
            for head_model in head_models:
                main_pred, _, _, _ = head_model(cls_token, patch_tokens)
                all_preds.append(main_pred.cpu().numpy())

        # Average predictions
        pred_mean = np.mean(all_preds, axis=0)
        predictions[image_path] = pred_mean

    return predictions
```

---

## 7. 実行手順

### Step 1: 特徴抽出
```bash
cd experiments/013
python extract_features.py \
    --output_dir /path/to/features \
    --img_size 960
```

### Step 2: Head学習
```bash
python train_head.py \
    --config configs/exp/exp013_dinov3.yaml
```

### Step 3: 推論
```bash
python inference.py \
    --experiment_dir <experiment_dir> \
    --run_name <run_name> \
    --img_size 960
```

---

## 8. ディスク容量見積もり

### 特徴ファイルサイズ（1画像あたり）
- CLS: 1280 × 4 bytes × 20 patterns = 102 KB
- Patches: 3600 × 1280 × 4 bytes × 20 patterns = 352 MB
- 合計: 約 352 MB/画像

### 総容量（仮に1000画像の場合）
- 圧縮なし: 352 GB
- 圧縮あり (npz_compressed): 約 100-150 GB（推定）

→ 大容量ストレージが必要。Augmentationパターン数の削減も検討可能。

---

## 9. 注意事項

1. **入力サイズ制約**: 必ず16の倍数（960推奨）
2. **レジスタトークン**: 出力の[1:5]は無視すること
3. **メモリ使用量**: 960×960入力は高メモリ消費。バッチサイズ調整が必要
4. **特徴抽出時間**: 1画像あたり数秒×20パターン×画像数
5. **Kaggle推論**: バックボーンの重みをKaggle Datasetsにアップロード必要
