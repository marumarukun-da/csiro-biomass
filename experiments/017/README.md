# Experiment 017: Multi-Head Regression

DINOv3特徴量に対して複数の回帰モデル（head）を比較評価するための実験。

## 特徴抽出

2つのモードをサポート：

### オンライン特徴抽出（デフォルト）

学習時にDINOv3バックボーンを使って画像から直接特徴量を抽出。

```yaml
dataset:
  img_size: 960
  # feature_dir を省略するとオンライン抽出
  target_cols:
    - Dry_Total_g
    - GDM_g
    - Dry_Green_g
```

### 事前計算された特徴量を使用

`.npz`ファイルから事前抽出された特徴量を読み込む。

```yaml
dataset:
  img_size: 960
  feature_dir: data/output/013/1/extract_features  # npzファイルのディレクトリ
  aug_idx: 0  # 使用するaugmentation index
  target_cols:
    - Dry_Total_g
    - GDM_g
    - Dry_Green_g
```

## 利用可能なHeadモデル

### 線形モデル

| Head | 説明 |
|------|------|
| `svr` | Support Vector Regression (RBFカーネル) |
| `ridge` | Ridge回帰 (L2正則化) |
| `lasso` | Lasso回帰 (L1正則化、スパース) |
| `elasticnet` | ElasticNet (L1+L2正則化) |
| `bayesian_ridge` | ベイズリッジ回帰 |
| `kernel_ridge` | カーネルリッジ回帰 |
| `gpr` | ガウス過程回帰 (計算コスト高) |

### GBDTモデル

| Head | 説明 |
|------|------|
| `gbdt` | sklearn GradientBoostingRegressor |
| `histgbdt` | sklearn HistGradientBoostingRegressor (高速) |
| `xgboost` | XGBoost (GPU対応) |
| `lightgbm` | LightGBM (GPU対応、高速) |
| `catboost` | CatBoost (GPU対応) |

### アンサンブルモデル

| Head | 説明 |
|------|------|
| `extratrees` | Extremely Randomized Trees |

## 使い方

### 学習

```bash
# 線形モデル
python train.py --config configs/exp/ridge_pca_pls.yaml
python train.py --config configs/exp/svr_pca_pls.yaml

# GBDTモデル
python train.py --config configs/exp/xgboost_pca_pls.yaml
python train.py --config configs/exp/lightgbm_pca_pls.yaml
python train.py --config configs/exp/catboost_pca_pls.yaml
python train.py --config configs/exp/gbdt_pca_pls.yaml
python train.py --config configs/exp/histgbdt_pca_pls.yaml

# アンサンブルモデル
python train.py --config configs/exp/extratrees_pca_pls.yaml
```

### 推論

```bash
# head種別は自動検出（PCA/PLSも自動適用）
python inference.py --experiment_dir <出力ディレクトリ名>

# 例
python inference.py --experiment_dir 20250101_123456_exp017_xgboost
```

## 設定ファイル

`configs/exp/` に各headの設定ファイルあり：

### 線形モデル
- `svr.yaml`, `ridge.yaml`, `lasso.yaml`, `elasticnet.yaml`
- `bayesian_ridge.yaml`, `kernel_ridge.yaml`, `gpr.yaml`
- `*_pca_pls.yaml` (PCA/PLS前処理付き)

### GBDT/アンサンブル
- `gbdt_pca_pls.yaml` - sklearn GradientBoosting
- `histgbdt_pca_pls.yaml` - sklearn HistGradientBoosting
- `xgboost_pca_pls.yaml` - XGBoost
- `lightgbm_pca_pls.yaml` - LightGBM
- `catboost_pca_pls.yaml` - CatBoost
- `extratrees_pca_pls.yaml` - ExtraTrees

### 設定例 (xgboost_pca_pls.yaml)

```yaml
head:
  type: xgboost
  tree_method: hist
  device: cpu          # 'cuda' for GPU
  n_jobs: -1
  random_state: 42
  # Grid search parameters (リスト = グリッドサーチ対象)
  n_estimators:
    - 500
    - 1000
    - 1500
  learning_rate:
    - 0.01
    - 0.05
    - 0.1
  max_depth:
    - 3
    - 5
    - 7
```

## 前処理 (PCA/PLS)

特徴量の次元削減にPCAとPLSを使用できます。

### 設定方法

```yaml
preprocessing:
  pca:
    n_components: 0.98  # 98%の分散を保持
  pls:
    n_components: 16     # 16個のPLS成分
```

### n_componentsの指定方法

| 値 | 説明 |
|----|------|
| `float (0-1)` | 累積寄与率（例: 0.95 = 95%の分散を保持） |
| `int` | 成分数（例: 100 = 100次元に削減） |
| `null` | 無効化 |

### 注意: PLSの成分数制限

PLSの成分数は `min(n_samples, n_features, n_targets)` が上限です。
ターゲットが3つの場合、`n_components: 16` を指定しても実際には3成分になります。

### 使い分け

- **PCA**: 教師なし次元削減。特徴量の分散を最大化する方向に射影
- **PLS**: 教師あり次元削減。ターゲット変数との相関を最大化する方向に射影
- **両方同時**: PCA成分とPLS成分を結合して特徴量として使用

## ファイル構成

```
experiments/017/
├── train.py                # 汎用学習スクリプト
├── inference.py            # 汎用推論スクリプト
├── src/
│   ├── backbone.py         # DINOv3バックボーン
│   ├── feature_extractor.py # オンライン特徴抽出
│   ├── feature_engine.py   # PCA/PLS前処理エンジン
│   ├── coverage.py         # 被覆率特徴
│   └── heads/
│       ├── base.py         # 基底クラス
│       ├── svr.py          # SVR
│       ├── ridge.py        # Ridge/Lasso/ElasticNet/BayesianRidge
│       ├── kernel_ridge.py
│       ├── gpr.py
│       ├── gbdt.py         # GradientBoosting/HistGradientBoosting
│       ├── xgboost_head.py
│       ├── lightgbm_head.py
│       ├── catboost_head.py
│       └── extratrees.py
└── configs/exp/
    └── *.yaml              # 各headの設定ファイル
```
