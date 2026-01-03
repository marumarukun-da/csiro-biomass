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
```

### 推論

```bash
# head種別は自動検出
python inference.py --experiment_dir <出力ディレクトリ名>

# 例
python inference.py --experiment_dir 20250101_123456_exp017_ridge
```

## 設定ファイル

`configs/exp/` に各headの設定ファイルあり：

- `svr.yaml`, `ridge.yaml`, `lasso.yaml`, `elasticnet.yaml`
- `bayesian_ridge.yaml`, `kernel_ridge.yaml`, `gpr.yaml`

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

## ファイル構成

```
experiments/017/
├── train.py              # 汎用学習スクリプト
├── inference.py          # 汎用推論スクリプト
├── src/
│   └── heads/
│       ├── base.py       # 基底クラス
│       ├── svr.py        # SVR
│       ├── ridge.py      # Ridge/Lasso/ElasticNet/BayesianRidge
│       ├── kernel_ridge.py
│       └── gpr.py
└── configs/exp/
    └── *.yaml            # 各headの設定ファイル
```
