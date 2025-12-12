# csiro-biomass

## Quick Start Checklist

初めてこのプロジェクトを使う場合は、以下のチェックリストに従ってセットアップしてください。

- [x] `.env.sample` をコピーして `.env` を作成
- [x] `.env` に `KAGGLE_USERNAME` と `KAGGLE_KEY` を設定
- [x] Docker コンテナを起動 (`docker compose up -d --build gpu` or `cpu`)
- [x] VSCode「コンテナーで再度開く」でコンテナに接続
- [x] 推奨拡張機能をインストール
- [x] Kaggle上でコンペティションへの参加登録
- [x] `sh scripts/download_competition.sh` でデータをダウンロード

---

## Setup

<details>
<summary>1. 環境変数の設定</summary>

`.env.sample`をコピーして`.env`を作成し、Kaggle認証情報を設定します：

```bash
cp .env.sample .env
```

`.env` ファイルを編集:
```bash
KAGGLE_USERNAME=your_username  # あなたのKaggleユーザー名
KAGGLE_KEY=your_key           # あなたのKaggle API key
```

> **Kaggle API Key の取得方法**:
> 1. [Kaggle](https://www.kaggle.com) にログイン
> 2. 右上のアイコン → Settings → API → Create New Token
> 3. ダウンロードされた `kaggle.json` から値をコピー

</details>

<details>
<summary>2. Docker環境の起動</summary>

GPU環境の場合：
```bash
docker compose up -d --build gpu
```

CPU環境の場合：
```bash
docker compose up -d --build cpu
```

</details>

<details>
<summary>3. VSCode接続と拡張機能</summary>

1. VSCodeの左下の「><」アイコンをクリック
2. 「コンテナーで再度開く」を選択してコンテナに接続
3. 推奨される拡張機能をインストール：
   - 右下に表示される「推奨される拡張機能をインストール」の通知をクリック
   - または、拡張機能タブ（Ctrl+Shift+X）から「推奨」セクションを確認

</details>

---

## Download Competition Dataset

```bash
# 1. Kaggle上でコンペティションへの参加登録を行う（必須）

# 2. データセットをダウンロード
sh scripts/download_competition.sh
```

---

## Submission Flow

### Step 1: 実験フォルダを作成

```bash
# 新しい実験フォルダを作成（テンプレートからコピー）
sh scripts/new_exp.sh                    # tabularテンプレート → experiments/002/
sh scripts/new_exp.sh --template image   # 画像系テンプレート
```

作成されるファイル：
| ファイル | 役割 | 編集が必要か |
|---------|------|------------|
| `config.py` | パス設定（自動で設定される） | 基本的に不要 |
| `code.ipynb` | 学習用ノートブック | **必須** |
| `inference.py` | Kaggle上で実行される推論コード | **必須** |

### Step 2: 実験を行う

`experiments/001/code.ipynb` を編集・実行してモデルを学習します。

<details>
<summary>学習コードの例（code.ipynb）</summary>

```python
import config
import joblib
import polars as pl
from xgboost import XGBClassifier

# データ読み込み
train_df = pl.read_csv(config.COMP_DATASET_DIR / "train.csv")

# モデル学習
model = XGBClassifier(n_estimators=100, random_state=0)
model.fit(
    X=train_df.select(["Age", "VIP", "VRDeck"]).to_numpy(),
    y=train_df["Transported"].to_numpy()
)

# モデル保存（OUTPUT_DIR に保存）
model_dir = config.OUTPUT_DIR / "models"
model_dir.mkdir(exist_ok=True, parents=True)
joblib.dump(model, model_dir / "model.joblib")
```

</details>

### Step 3: 推論コードを作成

`experiments/001/inference.py` を編集します。これはKaggle上で実行されるコードです。

<details>
<summary>推論コードの例（inference.py）</summary>

```python
import config
import joblib
import polars as pl

# データ読み込み
test_df = pl.read_csv(config.COMP_DATASET_DIR / "test.csv")
sub_df = pl.read_csv(config.COMP_DATASET_DIR / "sample_submission.csv")

# モデル読み込み（ARTIFACT_EXP_DIR から読み込む）
model_dir = config.ARTIFACT_EXP_DIR(config.EXP_NAME) / "models"
model = joblib.load(model_dir / "model.joblib")

# 推論
test_pred = model.predict(test_df.select(["Age", "VIP", "VRDeck"]).to_numpy())

# 提出ファイル作成（OUTPUT_DIR に保存）
sub_df.with_columns(pl.Series("Transported", test_pred)).write_csv(
    config.OUTPUT_DIR / "submission.csv"
)
```

</details>

> **Note: パスの使い分け**
> - `OUTPUT_DIR`: 当該実験の出力先（学習時のモデル保存先）
> - `ARTIFACT_EXP_DIR(exp_name)`: Kaggle上でアップロードされたモデルの参照先（推論時）

### Step 4: 依存パッケージを設定（初回のみ）

Kaggle環境にないパッケージを使う場合、`deps/code.ipynb` を編集します。

<details>
<summary>deps/code.ipynb の例</summary>

```python
# 必要なパッケージをダウンロード・インストール
!pip download -d /kaggle/working loguru polars
!pip install /kaggle/working/*.whl \
    --force-reinstall \
    --root-user-action ignore \
    --no-deps \
    --no-index \
    --find-links /kaggle/working
```

</details>

```bash
# 依存パッケージをKaggleにアップロード
sh scripts/push_deps.sh
```

### Step 5: 提出設定を確認

`sub/code.ipynb` と `sub/kernel-metadata.json` を確認・編集します。

<details>
<summary>sub/code.ipynb の確認ポイント</summary>

```python
# inference.py のパスが正しいか確認
!PYTHONPATH=/kaggle/input/csiro-biomass-codes \
  python /kaggle/input/csiro-biomass-codes/experiments/001/inference.py
#                                                                          ^^^
#                                                            実験番号を確認
```

</details>

<details>
<summary>sub/kernel-metadata.json の確認ポイント</summary>

```json
{
  "model_sources": [
    "marumarukun/csiro-biomass-artifacts/other/001/1"
  ]  // ← 実験番号が正しいか確認
}
```

</details>

### Step 6: 検証と提出

```bash
# 1. 提出前の検証（問題がないかチェック）
sh scripts/validate.sh 001

# 2. 提出（検証 → アップロード → Kernel push を一括実行）
sh scripts/submit.sh 001

# 確認だけしたい場合（実際にはpushしない）
sh scripts/submit.sh 001 --dry-run

# 3. 提出状況の確認
sh scripts/status.sh
```

<details>
<summary>手動で個別に実行する場合</summary>

```bash
# コード・モデルをアップロード
sh scripts/push_experiment.sh 001

# 提出Kernelをpush
sh scripts/push_sub.sh
```

</details>

---

## Scripts Reference

| Script | Description |
|--------|-------------|
| `scripts/validate.sh <exp>` | 提出前の検証（ファイル存在確認、設定チェック） |
| `scripts/submit.sh <exp>` | 一括提出（validate → upload → push） |
| `scripts/status.sh` | 最新の提出状況を確認 |
| `scripts/new_exp.sh [--template TYPE]` | 新しい実験フォルダを作成 |
| `scripts/download_competition.sh` | コンペティションデータをダウンロード |
| `scripts/push_experiment.sh <exp>` | 実験コード・モデルをアップロード |
| `scripts/push_deps.sh` | 依存パッケージをアップロード |
| `scripts/push_sub.sh` | 提出Kernelをpush |

<details>
<summary>submit.sh のオプション</summary>

```bash
sh scripts/submit.sh <experiment_name> [options]

Options:
  --dry-run        実際にpushせずに何が起こるか確認
  --skip-validate  検証をスキップ
  --skip-codes     コードDatasetのアップロードをスキップ
  --skip-artifacts モデルアップロードをスキップ
```

</details>

---

## Project Structure

```
csiro-biomass/
├── experiments/           # 実験フォルダ
│   ├── 001/              # 実験番号ごとのフォルダ
│   │   ├── config.py     # パス設定、コンペ情報
│   │   ├── code.ipynb    # 学習用ノートブック
│   │   └── inference.py  # 推論コード
│   └── templates/        # 実験テンプレート
├── sub/                   # 提出用Kernel
│   ├── code.ipynb        # 提出時に実行されるノートブック
│   └── kernel-metadata.json
├── deps/                  # 依存パッケージ管理
├── src/                   # ユーティリティ
│   ├── upload.py         # アップロード処理
│   ├── download.py       # ダウンロード処理
│   └── kaggle_utils/     # Kaggle API ラッパー
├── scripts/              # シェルスクリプト
├── data/
│   ├── input/            # ダウンロードしたデータ
│   └── output/           # 実験の出力
└── docker/               # Dockerfile
```

<details>
<summary>データフロー図</summary>

```
[ローカル環境]                              [Kaggle]

experiments/001/
  ├─ code.ipynb (学習)
  ├─ config.py
  └─ inference.py ──────────────────────→ Dataset: {comp}-codes

data/output/001/1/
  └─ models/*.pkl ──────────────────────→ Model: {comp}-artifacts/other/001/1

deps/
  └─ code.ipynb ────────────────────────→ Kernel: {comp}-deps (依存パッケージ)

sub/
  └─ code.ipynb ────────────────────────→ Kernel: {comp}-sub
      │                                       │
      │ (inference.pyを呼び出し)               ↓
      └───────────────────────────────→ submission.csv
```

</details>

---

## パス環境の詳細

ローカル環境とKaggle環境でのパス構造の対応関係です。

### ローカル環境（学習時）

```
/workspace/ (プロジェクトルート)
│
├── experiments/{exp}/            # 実験コード
│   ├── config.py
│   ├── train.py (or code.ipynb)
│   ├── inference.py
│   └── src/
│
├── data/
│   ├── input/{competition}/      # コンペデータ
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── train/, test/ (画像等)
│   │
│   └── output/{exp}/1/           # ★学習結果の出力先★
│       └── {timestamp}_{exp_name}/
│           └── {run_name}/       # ← これをKaggleにアップロード
│               ├── weights/
│               │   └── *.pth
│               └── config.yaml
```

### Kaggle環境（推論時）

```
/kaggle/
│
├── working/                      # ★コードを配置★
│   ├── config.py
│   ├── inference.py
│   └── src/
│
└── input/
    ├── {competition}/            # コンペデータ（自動で利用可能）
    │   ├── test.csv
    │   └── test/
    │
    └── {competition}-artifacts/  # ★アップロードしたモデル★
        └── other/
            └── {exp}/
                └── 1/
                    └── {run_name}/
                        ├── weights/
                        └── config.yaml
```

### config.py での変数対応

| 変数 | ローカル | Kaggle |
|------|----------|--------|
| `INPUT_DIR` | `/workspace/data/input` | `/kaggle/input` |
| `OUTPUT_DIR` | `/workspace/data/output/{exp}/1` | `/kaggle/working` |
| `COMP_DATASET_DIR` | `INPUT_DIR/{competition}` | `/kaggle/input/{competition}` |
| `ARTIFACT_DIR` | `/workspace/data/output` | `/kaggle/input/{competition}-artifacts/other` |

### アップロード時の注意

ローカルの出力構造:
```
data/output/{exp}/1/{timestamp}_{exp_name}/{run_name}/
                    ~~~~~~~~~~~~~~~~~~~~~~ ~~~~~~~~~~
                    中間ディレクトリ        ← これをアップロード
```

**`{run_name}/` ディレクトリをアップロード**します。`{timestamp}_{exp_name}/`は中間ディレクトリなので含めません。

Kaggle Datasets上の構造:
```
{competition}-artifacts/other/{exp}/1/{run_name}/
```

---

## Troubleshooting

<details>
<summary>よくあるエラーと対処法</summary>

### "KAGGLE_USERNAME is not set"

**原因**: `.env` ファイルが存在しないか、設定が正しくない

**対処法**:
1. `.env.sample` を `.env` にコピー
2. `KAGGLE_USERNAME` と `KAGGLE_KEY` を正しく設定
3. 値が `your_username` や `your_key` のままになっていないか確認

---

### "Experiment directory not found"

**原因**: 指定した実験番号のフォルダが存在しない

**対処法**:
```bash
# 新しい実験フォルダを作成
sh scripts/new_exp.sh
# または手動で作成
mkdir -p experiments/001
cp experiments/templates/tabular/* experiments/001/
```

---

### "Output directory not found"

**原因**: 実験を実行していない、または出力パスが間違っている

**対処法**:
1. `experiments/001/code.ipynb` を実行してモデルを学習・保存
2. `config.py` の `OUTPUT_DIR` を確認
3. `data/output/001/1/` にファイルが出力されているか確認

---

### Kaggle Kernel が "Error" で失敗

**対処法**:
1. `sh scripts/status.sh` でエラーログを確認
2. ログを確認: `kaggle kernels output <kernel_slug>`
3. よくある原因:
   - 依存パッケージの不足 → `deps/code.ipynb` を更新して `sh scripts/push_deps.sh`
   - パスの間違い → `sub/code.ipynb` の inference.py パスを確認
   - モデルファイルの参照エラー → `sub/kernel-metadata.json` の `model_sources` を確認

---

### "403 - Forbidden" エラー

**原因**: コンペティションの規約に同意していない

**対処法**:
1. Kaggle Web で該当コンペティションのページを開く
2. "Join Competition" または "I Understand and Accept" をクリック
3. 再度ダウンロードを試行

---

### Model/Dataset が Kaggle 上に表示されない

**原因**: アップロード処理が完了していない、またはプライベート設定

**対処法**:
1. アップロード時のエラーメッセージを確認
2. Kaggle Web の "Your Work" → "Datasets" / "Models" で確認
3. 初回アップロード後、反映に数分かかることがある

</details>

---

## Reference
- [kaggle code competition 用のテンプレート作ってみた](https://osushinekotan.hatenablog.com/entry/2024/12/24/193145)
- [効率的なコードコンペティションの作業フロー](https://ho.lc/blog/kaggle_code_submission/)
- [Kaggleコンペ用のVScode拡張を開発した](https://ho.lc/blog/vscode_kaggle_extension/)
