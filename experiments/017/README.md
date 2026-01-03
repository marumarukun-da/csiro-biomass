# Experiment 017: Multi-Head Regression

DINOv3特徴量に対して複数の回帰モデル（head）を比較評価するための実験。

## 利用可能なHeadモデル

| Head | 説明 |
|------|------|
| `svr` | Support Vector Regression (RBFカーネル) |
| `ridge` | Ridge回帰 (L2正則化) |
| `lasso` | Lasso回帰 (L1正則化、スパース) |
| `elasticnet` | ElasticNet (L1+L2正則化) |
| `bayesian_ridge` | ベイズリッジ回帰 |
| `kernel_ridge` | カーネルリッジ回帰 |
| `gpr` | ガウス過程回帰 (計算コスト高) |

## 使い方

### 学習

```bash
# Ridge で学習
python train.py --config configs/exp/ridge.yaml

# Lasso で学習
python train.py --config configs/exp/lasso.yaml

# SVR で学習
python train.py --config configs/exp/svr.yaml

# PCA/PLS前処理付きで学習
python train.py --config configs/exp/ridge_pca_pls.yaml
```

### 推論

```bash
# head種別は自動検出（PCA/PLSも自動適用）
python inference.py --experiment_dir <出力ディレクトリ名>

# 例
python inference.py --experiment_dir 20250101_123456_exp017_ridge
```

## 設定ファイル

`configs/exp/` に各headの設定ファイルあり：

- `svr.yaml`, `ridge.yaml`, `lasso.yaml`, `elasticnet.yaml`
- `bayesian_ridge.yaml`, `kernel_ridge.yaml`, `gpr.yaml`
- `ridge_pca_pls.yaml`, `svr_pca_pls.yaml` (PCA/PLS前処理付き)

### 設定例 (ridge.yaml)

```yaml
head:
  type: ridge           # head種別
  fit_intercept: true   # 固定パラメータ
  alpha:                # リスト = グリッドサーチ対象
    - 0.01
    - 0.1
    - 1.0
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

### 使い分け

- **PCA**: 教師なし次元削減。特徴量の分散を最大化する方向に射影
- **PLS**: 教師あり次元削減。ターゲット変数との相関を最大化する方向に射影
- **両方同時**: PCA成分とPLS成分を結合して特徴量として使用

### 設定例

```yaml
# PCAのみ（95%分散保持）
preprocessing:
  pca:
    n_components: 0.95
  pls:
    n_components: null

# PLSのみ（8成分）
preprocessing:
  pca:
    n_components: null
  pls:
    n_components: 8

# 両方（推奨）
preprocessing:
  pca:
    n_components: 0.95
  pls:
    n_components: 8
```

## ファイル構成

```
experiments/017/
├── train.py              # 汎用学習スクリプト
├── inference.py          # 汎用推論スクリプト
├── src/
│   ├── feature_engine.py # PCA/PLS前処理エンジン
│   └── heads/
│       ├── base.py       # 基底クラス
│       ├── svr.py        # SVR
│       ├── ridge.py      # Ridge/Lasso/ElasticNet/BayesianRidge
│       ├── kernel_ridge.py
│       └── gpr.py
└── configs/exp/
    └── *.yaml            # 各headの設定ファイル
```
