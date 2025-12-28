# Experiment 013: DINOv3 Dense Features

DINOv3 ViT-Huge+の事前学習済み特徴量を使用したTwo-Stage学習アーキテクチャ。

## アーキテクチャ概要

```
[Image 960x960] → [DINOv3 Backbone (Frozen)] → [CLS + 3600 Patches] → [Head Model] → [Predictions]
```

- **Backbone**: `vit_huge_plus_patch16_dinov3.lvd1689m` (1.2GB, Frozen)
- **入力サイズ**: 960×960 (16の倍数必須)
- **出力**: CLS token (1280次元) + Patch tokens (3600×1280次元)
- **Head**: CLS Gating + Multi-task outputs

## ディレクトリ構造

```
experiments/013/
├── README.md                    # このファイル
├── config.py                    # パス設定
├── code.ipynb                   # Kaggleアップロード・推論用
├── extract_features.py          # 特徴抽出スクリプト
├── train_head.py                # Head学習スクリプト
├── inference_dinov3.py          # 推論スクリプト
├── configs/
│   └── exp/
│       └── exp013_dinov3.yaml   # 学習設定
└── src/
    ├── backbone.py              # DINOv3 Backbone
    ├── head_model.py            # CLS Gating Head
    ├── feature_dataset.py       # 特徴量データセット
    ├── manifold_mixup.py        # Feature-level Mixup
    ├── loss_function.py         # Multi-task Loss
    └── ...

data/output/013/1/
├── backbone.pth                 # Backbone重み (特徴抽出時に保存)
├── extract_features/            # 抽出された特徴量
│   └── {image_id}.npz
└── {日付フォルダ}/              # 学習結果
    ├── backbone.pth             # アップロード時にコピー
    ├── {run_name}/
    │   ├── config.yaml
    │   └── weights/
    │       └── best_fold*.pth
    └── summary.csv
```

## 実行手順

### Step 1: 特徴抽出

DINOv3 Backboneを使用して全訓練画像の特徴量を抽出・保存。

```bash
cd /home/marumarukun/kaggle/csiro-biomass/experiments/013

python extract_features.py \
    --output_dir data/output/013/1/extract_features \
    --img_size 960 \
    --skip_existing
```

**オプション:**
- `--output_dir`: 特徴量の保存先 (必須)
- `--img_size`: 画像サイズ (デフォルト: 960、16の倍数)
- `--skip_existing`: 既存ファイルをスキップ

**出力:**
- `data/output/013/1/backbone.pth`: Backbone重み (Kaggle用)
- `data/output/013/1/extract_features/{image_id}.npz`: 各画像の特徴量

**所要時間:** 約1-2時間 (GPU依存)

### Step 2: Head学習

抽出した特徴量を使用してHead Modelを学習。

```bash
python train_head.py --config configs/exp/exp013_dinov3.yaml
```

**オプション:**
- `--config`: 設定ファイルパス (必須)
- `--fold`: 特定のfoldのみ学習 (例: `--fold 0`)

**出力:**
- `data/output/013/1/{日付}_{実験名}/{run_name}/weights/best_fold*.pth`
- `data/output/013/1/{日付}_{実験名}/summary.csv`

**所要時間:** 約30分-1時間 (5 folds)

### Step 3: ローカル推論 (オプション)

学習済みモデルでテストデータを推論。

```bash
python inference_dinov3.py \
    --experiment_dir {日付フォルダ} \
    --run_name {run_name} \
    --folds all \
    --weight_type best
```

**オプション:**
- `--experiment_dir`: 日付フォルダ名 (必須)
- `--run_name`: run名 (必須)
- `--folds`: 使用するfold (デフォルト: all)
- `--weight_type`: best or last (デフォルト: best)
- `--backbone_weights`: Backbone重みパス (オフライン用)

### Step 4: Kaggleアップロード

`code.ipynb`を開き、cell-2を実行。

1. `EXPERIMENT_DIR`を日付フォルダ名に変更
2. cell-2を実行

```python
EXPERIMENT_DIR = "20251228_123456_exp013"  # ← 実際の日付フォルダ名に変更
```

**アップロードされるもの:**
- Kaggle Model: `backbone.pth` + Head weights
- Kaggle Dataset: 実験コード

### Step 5: Kaggle提出

Kaggle Notebookで推論を実行。

1. code.ipynbをKaggleにアップロード
2. Data Sourcesに以下を追加:
   - `csiro-biomass` (コンペデータ)
   - `csiro-biomass-artifacts` (モデル重み)
   - `csiro-biomass-codes-013` (実験コード)
3. cell-1のコメントを外して実行

```python
from inference_dinov3 import kaggle_inference

RUN_NAME = "001_exp013_dinov3__lr-0_002__weight_decay-0_01"
submission_df = kaggle_inference(
    run_name=RUN_NAME,
    folds=None,  # all folds
    img_size=960,
    weight_type="best",
)
submission_df.to_csv("submission.csv", index=False)
```

## 設定ファイル (configs/exp/exp013_dinov3.yaml)

主要な設定項目:

```yaml
backbone:
  hidden_dim: 1280
  num_patches: 3600  # 60x60 for 960x960

dataset:
  feature_dir: data/output/013/1/extract_features
  num_aug_patterns: 20

trainer:
  num_epochs: 200
  batch_size: 32
  n_folds: 5

optimization:
  lr: [2e-3, 1e-3, 1e-4]  # Grid search
  weight_decay: [1e-2, 1e-3]
```

## Tips

### GPU メモリ不足の場合

- `batch_size`を減らす (16など)
- `gradient_accumulation_steps`を増やす

### 学習が遅い場合

- `num_workers`を増やす (4-8)
- 特徴量ファイルをSSDに配置

### 特定のfoldだけ再学習

```bash
python train_head.py --config configs/exp/exp013_dinov3.yaml --fold 2
```

## トラブルシューティング

### "backbone.pth not found"

特徴抽出を再実行するか、手動でbackbone重みを保存:

```python
from src.backbone import build_backbone, save_backbone_weights
backbone = build_backbone(pretrained=True, device="cuda")
save_backbone_weights(backbone, "data/output/013/1/backbone.pth")
```

### Kaggleでインターネット接続エラー

`backbone.pth`がアップロードされているか確認。
code.ipynb cell-2で正しくコピーされているか確認。

### Out of Memory (推論時)

TTAを減らすか、バッチサイズを1にして推論。
