# pj-kenpin

本リポジトリは、YAML で定義した設定に基づいて ArcFace を含む画像分類モデルを学習させるためのテンプレートです。以下の手順に沿えば、セットアップから学習実行まで一通り行えます。

## 1. セットアップ

1. 依存関係をuvでインストールし、仮想環境を有効化します
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync
   source .venv/bin/activate
   ```

2. 学習用データを配置します。`SampleDataset.from_directory` はクラスごとのフォルダ構成を想定しています（例: `tree` コマンド風の構造）。
   ```
   data/
   └── yourdataset/
       ├── train/
       │   ├── class_a/
       │   │   ├── img1.jpg
       │   │   └── ...
       │   └── class_b/
       │       ├── img10.jpg
       │       └── ...
       └── val/
           ├── class_a/
           │   ├── img101.jpg
           │   └── ...
           └── class_b/
               ├── img201.jpg
               └── ...
   ```

## 2. 設定ファイルを準備する

- 基本設定は `configs/base/exp_template.yaml` にあります。このファイルを参照しつつ、`configs/exp/exp001.yaml` のように実験用 YAML を作成してください。
- YAML には以下のブロックがあります。
  - `experiment`: 実験名 (`name`)、シード (`seed`)、出力先 (`output_dir`)、メモ (`notes`) など。
  - `dataset`: `train_dir`、`val_dir`、`class_names`、`img_size` などデータに関する設定。
  - `trainer`: バッチサイズ、エポック、ワーカ数、AMP 利用 (`use_amp`) 、ラベル平滑化（`label_smoothing`）など。
  - `optimization`: 学習率や Weight Decay、ウォームアップ率、EMA 設定 (`use_ema`) など。リストで複数値を指定するとグリッドサーチが走ります。
  - `model` / `arcface`: バックボーンや ArcFace 固有パラメータ。
  - `augmentation`: 正規化パラメータ (`normalize`)、トレーニング時の Data Augmentation（`train`）など。詳細は「3. データセット正規化パラメータの計算」を参照。
- バックボーン候補は `configs/backbone/*.yaml` でサイズごと（S/M/L/X）にまとめているので、適宜コピペで使用してください。

## 3. データセット正規化パラメータの計算（オプション）

データセット固有の平均値・標準偏差を計算し、正規化パラメータを最適化したい場合は `src/tools/calc_mean_std.py` を使用します。

### 3.1. 使用方法

データセットのディレクトリを指定して実行します。

```bash
python src/tools/calc_mean_std.py data/yourdataset/train
```

### 3.2. 機能

- データセット内の全画像を再帰的にスキャン
- グレースケール（1ch）とRGB（3ch）を自動判別
- ピクセル値を [0, 1] に正規化して統計を計算
- Albumentations の `A.Normalize` で使える形式で出力

### 3.3. 出力例

**RGB画像の場合:**
```
Detected: RGB (3ch)
  mean = [0.4852, 0.4562, 0.4062]
  std  = [0.2292, 0.2237, 0.2249]

Albumentations:
  A.Normalize(mean=[0.4852, 0.4562, 0.4062], std=[0.2292, 0.2237, 0.2249], max_pixel_value=255.0)
```

**グレースケール画像の場合:**
```
Detected: Grayscale (1ch)
  mean = (0.4852,)
  std  = (0.2292,)

Albumentations:
  A.Normalize(mean=(0.4852,), std=(0.2292,), max_pixel_value=255.0)
```

### 3.4. コードへの反映

計算された値をYAML設定ファイルの `augmentation.normalize` セクションに設定します。

設定ファイルは `configs/base/exp_template.yaml` または実験用の設定ファイル（例: `configs/exp/exp001.yaml`）を編集してください。

**RGB画像の場合** (`augmentation.normalize.rgb`)：
```yaml
augmentation:
  normalize:
    rgb:
      mean:
        - 0.4852
        - 0.4562
        - 0.4062
      std:
        - 0.2292
        - 0.2237
        - 0.2249
```

**グレースケール画像の場合** (`augmentation.normalize.grayscale`)：
```yaml
augmentation:
  normalize:
    grayscale:
      mean:
        - 0.4852
      std:
        - 0.2292
```

学習時・推論時ともに、`model.in_chans`の値（1: グレースケール、3: RGB）に応じて適切な正規化パラメータが自動的に選択されます。

**注意**: ImageNet の事前学習済みモデルを使用する場合、ImageNet の統計値（デフォルト値）を使った方が良い結果が得られることもあります。データセット固有の値は、ファインチューニングの度合いや対象ドメインに応じて使い分けてください。

## 4. 学習を実行する

リポジトリ直下で以下を実行します。

```bash
python train.py --config configs/exp/exp001.yaml
```

- グリッドサーチが有効な場合、指定されたパラメータの組み合わせ分だけ学習が走ります。
- 出力は `output/{日時}_{experiment.name}/` 以下に作成され、各ランに以下が保存されます。
  - `weights/best_loss.pth`, `weights/last.pth`
  - `logs/metrics.csv`（エポックごとの loss 推移）
  - `plots/loss_curve.png`
  - `config.yaml`（実行時に使用した設定のコピー）
- 実験ディレクトリ直下には `summary.csv` と `summary.json` が生成されます。`summary.json` には全結果と、最も val loss が小さかった組み合わせ (`best_run`) が記録されます。

## 5. 推論を実行する

学習済みモデルを使って推論を行うには、`infer.py` を使用します。

### 5.1. 推論用データの準備

推論用データは学習時と同様のフォルダ構成で配置してください。

```
data/
└── test/
    ├── class_a/
    │   ├── test_img1.jpg
    │   └── ...
    └── class_b/
        ├── test_img10.jpg
        └── ...
```

### 5.2. 推論の実行

リポジトリ直下で以下を実行します。

```bash
python infer.py \
  --exp_dir output/20250108_012805_exp001 \
  --input data/test
```

#### オプション

- `--exp_dir`: 学習済みモデルが格納された実験ディレクトリ（必須）
- `--input`: 推論対象の画像フォルダ（必須）
- `--checkpoint`: 使用する重みファイル（`best_loss` または `last`、デフォルト: `best_loss`）
- `--tta`: Test Time Augmentation を有効化（水平反転・垂直反転・オリジナルの3画像の平均）
- `--anomaly_score`: 異常度スコアを計算（1番目のクラスへのコサイン距離ベースの指標）

#### 実行例

```bash
# TTA + 異常度スコア付きで推論
python infer.py \
  --exp_dir output/20250108_012805_exp001 \
  --input data/test \
  --tta \
  --anomaly_score

# lastモデルで推論
python infer.py \
  --exp_dir output/20250108_012805_exp001 \
  --input data/test \
  --checkpoint last
```

### 5.3. 出力

推論結果は `{exp_dir}_infer/` ディレクトリに保存されます。

```
output/20250108_012805_exp001_infer/
├── 001_efficientnetv2_b0__lr-0.001/
│   └── predictions.csv
├── 002_efficientnetv2_b1__lr-0.001/
│   └── predictions.csv
└── inference_summary.csv
```

- **ランごとの `predictions.csv`**: 各画像の詳細な推論結果
  - `file_name`: 画像ファイル名
  - `true_label`: 真のラベル
  - `pred_label`: 予測ラベル
  - `probability_{class_name}`: 各クラスの予測確率
  - `anomaly_score`: 異常度スコア（`--anomaly_score` 指定時のみ）

- **`inference_summary.csv`**: 全ランのメトリクスをまとめたサマリー
  - `accuracy`: 全体の精度
  - `precision_macro`, `recall_macro`, `f1_macro`: マクロ平均のメトリクス
  - `precision_{class_name}`, `recall_{class_name}`, `f1_{class_name}`: クラスごとのメトリクス

**注意**: `--anomaly_score` を使用する場合、異常度は1番目のクラスへのコサイン距離を基準に計算されます。学習時に1番目のクラスが「正常クラス」であることを確認してください。

