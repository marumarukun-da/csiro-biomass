# 補助タスク（Auxiliary Tasks）の検討

## 概要

本ドキュメントでは、バイオマス予測タスクにおける補助タスク（マルチタスク学習）の候補を検討した結果をまとめる。
testデータにはSpecies等のメタデータが含まれないため、これらを直接入力特徴量として使用できない。
代わりに、補助タスクとしてモデルに予測させることで、有用な表現学習を促進することを目指す。

---

## 用語説明

### ICC（Intraclass Correlation Coefficient：級内相関係数）

ICCは、カテゴリ変数（例：Species）が連続変数（例：Dry_Total_g）の変動をどの程度説明できるかを測定する指標。

**計算方法:**
$$
ICC = \frac{\sigma^2_{between}}{\sigma^2_{between} + \sigma^2_{within}}
$$

- $\sigma^2_{between}$: グループ間分散（カテゴリ間のばらつき）
- $\sigma^2_{within}$: グループ内分散（同一カテゴリ内でのばらつき）

**解釈:**
- **ICC = 0**: カテゴリによる違いがない（グループ内分散のみ）
- **ICC = 1**: 同一カテゴリ内では完全に同じ値（グループ間分散のみ）
- **ICC > 0.5**: カテゴリによる説明力が比較的高い

本ドキュメントでは、SpeciesのICC=0.611は「植物種によってバイオマス量の約61%が説明できる」ことを意味し、Speciesが補助タスクとして有効である根拠となっている。

---

## train.csvに含まれるメタデータ

| カラム名 | データ型 | ユニーク数 | 説明 |
|---------|---------|-----------|------|
| Sampling_Date | string | 28 | 撮影日（2015/1/15 ~ 2015/11/10） |
| State | string | 4 | 州（Tas, Vic, NSW, WA） |
| Species | string | 15 | 植物種 |
| Pre_GSHH_NDVI | float | 65 | 植生指数（0.16 ~ 0.91） |
| Height_Ave_cm | float | 81 | 平均高さ（1.0 ~ 70.0 cm） |

---

## 各メタデータの詳細分析

### 1. Species（植物種）

#### 分布

| Species | 画像数 | 割合 |
|---------|-------|------|
| Ryegrass_Clover | 98 | 27.5% |
| Ryegrass | 62 | 17.4% |
| Phalaris_Clover | 42 | 11.8% |
| Clover | 41 | 11.5% |
| Fescue | 28 | 7.8% |
| Lucerne | 22 | 6.2% |
| その他9種 | 64 | 17.9% |

#### ターゲット（Dry_Total_g）との関係

- **ANOVA F値**: 13.58 (p < 0.001)
- **ICC**: 0.611（全メタデータ中最高）

| Species | 平均Dry_Total_g | 標準偏差 |
|---------|----------------|---------|
| Phalaris | 100.26 | 31.05 |
| Fescue | 86.32 | 38.41 |
| Lucerne | 54.65 | 28.98 |
| Phalaris_Clover | 51.41 | 21.56 |
| Ryegrass_Clover | 38.89 | 23.22 |
| Clover | 34.59 | 17.00 |

→ 植物種によってバイオマス量が大きく異なる（Phalaris: 100g vs Clover: 35g）

---

### 2. Height_Ave_cm（平均高さ）

#### 統計量

| 指標 | 値 |
|------|-----|
| 平均 | 7.60 cm |
| 標準偏差 | 10.29 cm |
| 最小 | 1.0 cm |
| 最大 | 70.0 cm |

#### ターゲットとの関係

- **相関係数**: r = 0.497（中〜強い正の相関）

→ 草の高さが高いほどバイオマス量が多い傾向

---

### 3. Pre_GSHH_NDVI（植生指数）

#### 統計量

| 指標 | 値 |
|------|-----|
| 平均 | 0.657 |
| 標準偏差 | 0.152 |
| 最小 | 0.160 |
| 最大 | 0.910 |

#### ターゲットとの関係

- **相関係数**: r = 0.361（中程度の正の相関）

→ NDVIが高い（緑が濃い）ほどバイオマス量が多い傾向

---

### 4. State（州）

#### 分布

| State | 画像数 | 平均Dry_Total_g |
|-------|-------|----------------|
| NSW | 75 | 70.90 |
| Vic | 112 | 42.67 |
| Tas | 138 | 36.80 |
| WA | 32 | 31.39 |

#### ターゲットとの関係

- **ANOVA F値**: 36.56 (p < 0.001)

→ 州によって平均バイオマス量が大きく異なる（NSW: 71g vs WA: 31g）

---

### 5. Month（撮影月）

Sampling_Dateから月を抽出して分析。

#### 分布と平均ターゲット

| 月 | 画像数 | 平均Dry_Total_g |
|----|-------|----------------|
| 1月 | 17 | 58.34 |
| 2月 | 24 | 79.92 |
| 4月 | 10 | 42.16 |
| 5月 | 42 | 44.89 |
| 6月 | 53 | 30.44 |
| 7月 | 41 | 25.61 |
| 8月 | 37 | 41.52 |
| 9月 | 67 | 45.06 |
| 10月 | 29 | 67.61 |
| 11月 | 37 | 48.17 |

#### ターゲットとの関係

- **ANOVA F値**: 14.09 (p < 0.001)
- **相関係数**: r = -0.087（ほぼ無相関）

→ 月による変動はあるが、線形関係ではない（季節的な周期性）

---

## 補助タスクとしての適性評価

### 評価基準

1. **ターゲットとの関係**: 補助タスクがメインタスク（バイオマス予測）に関連しているか
2. **画像からの予測可能性**: 画像の視覚的特徴から予測できるか（予測できないものは補助タスクとして意味がない）

### 評価結果

| メタデータ | タスク種類 | ターゲット関連 | 画像から予測可能 | 推奨度 |
|-----------|-----------|---------------|-----------------|--------|
| **Species** | 分類（15クラス） | ★★★ ICC=0.611 | ★★★ 草の形状・色・テクスチャ | ◎ 強く推奨 |
| **Height_Ave_cm** | 回帰 | ★★☆ r=0.497 | ★★☆ 草の密度・立体感 | ○ 推奨 |
| **Pre_GSHH_NDVI** | 回帰 | ★☆☆ r=0.361 | ★★★ 緑色の濃さ | ○ 推奨 |
| State | 分類（4クラス） | ★★☆ F=36.56 | ★☆☆ 地域差が不明 | △ 検討の余地あり |
| Month | 分類（10クラス） | ★☆☆ F=14.09 | ★☆☆ 季節感が不明 | △ 検討の余地あり |

---

## 各補助タスクの詳細評価

### Species分類（◎ 強く推奨）

**推奨理由:**
- ターゲットとの関連が最も強い（ICC=0.611）
- 画像から視覚的に判別可能（草の形状、葉の形、色合い）
- バイオマス量は草の種類に大きく依存するため、種を認識する能力は直接的に有用

**注意点:**
- クラス不均衡が激しい（Ryegrass_Clover: 98枚 vs Mixed: 2枚）
- Focal LossやClass Weightingの導入を検討

---

### Height_Ave_cm回帰（○ 推奨）

**推奨理由:**
- バイオマス量と物理的に関連（高い草=バイオマス多い）
- 草の密度や立体感を捉える学習を促進

**注意点:**
- 単眼画像からの高さ推定は難しい場合がある
- 相対的な高さ（画像内での比較）は推定しやすいが、絶対的な高さは困難

---

### Pre_GSHH_NDVI回帰（○ 推奨）

**推奨理由:**
- NDVIは「緑の濃さ」を表す指標で、RGB画像から推測しやすい
- 生育状態の理解に寄与
- Height_Ave_cmと組み合わせると効果的な可能性

**注意点:**
- ターゲットとの相関は中程度（r=0.361）
- 単独での効果は限定的かもしれない

---

### State分類（△ 検討の余地あり）

**懸念点:**
- 画像から「どの州か」を判別するのは困難な可能性
- 土壌の色や植生パターンに地域差があれば有効だが、不明

**試す価値:**
- 州による平均値の差は大きい（NSW: 70.9g vs WA: 31.4g）
- もし画像から判別できれば強力な補助タスクになる

---

### Month分類（△ 検討の余地あり）

**懸念点:**
- 画像から撮影月を判別するのは難しい
- 草の成長段階や色合いの季節変化が見えれば有効だが、不明

**試す価値:**
- 月による変動はある（2月: 79.9g vs 6月: 30.4g）
- 季節的なパターンを学習できれば有用

---

## 推奨する補助タスクの組み合わせ

### 優先度順

1. **Species分類**（最優先）
   - 15クラス分類タスク
   - CrossEntropyLoss + Class Weighting

2. **Height_Ave_cm回帰**（推奨）
   - 連続値回帰タスク
   - MSELoss or SmoothL1Loss

3. **Pre_GSHH_NDVI回帰**（オプション）
   - 連続値回帰タスク
   - MSELoss or SmoothL1Loss

---

## 実装例

### モデル構造

```python
import torch
import torch.nn as nn

class MultiTaskModel(nn.Module):
    def __init__(self, backbone, feat_dim=512, num_species=15):
        super().__init__()
        self.backbone = backbone

        # メインタスク: バイオマス予測（3値: Dry_Total_g, GDM_g, Dry_Green_g）
        self.head_biomass = nn.Linear(feat_dim, 3)

        # 補助タスク1: Species分類
        self.head_species = nn.Linear(feat_dim, num_species)

        # 補助タスク2: Height回帰
        self.head_height = nn.Linear(feat_dim, 1)

        # 補助タスク3: NDVI回帰
        self.head_ndvi = nn.Linear(feat_dim, 1)

    def forward(self, x):
        feat = self.backbone(x)

        return {
            'biomass': self.head_biomass(feat),
            'species': self.head_species(feat),
            'height': self.head_height(feat),
            'ndvi': self.head_ndvi(feat),
        }
```

### 損失関数

```python
def compute_loss(outputs, targets, lambda_species=0.2, lambda_height=0.1, lambda_ndvi=0.1):
    """
    マルチタスク損失関数

    Args:
        outputs: モデルの出力 dict
        targets: ターゲット dict
        lambda_*: 各補助タスクの重み

    Returns:
        total_loss: 合計損失
        loss_dict: 各損失の内訳
    """
    # メインタスク: バイオマス予測
    loss_biomass = F.mse_loss(outputs['biomass'], targets['biomass'])

    # 補助タスク1: Species分類
    loss_species = F.cross_entropy(
        outputs['species'],
        targets['species'],
        weight=class_weights  # クラス不均衡対策
    )

    # 補助タスク2: Height回帰
    loss_height = F.smooth_l1_loss(outputs['height'], targets['height'])

    # 補助タスク3: NDVI回帰
    loss_ndvi = F.smooth_l1_loss(outputs['ndvi'], targets['ndvi'])

    # 合計損失
    total_loss = (
        loss_biomass
        + lambda_species * loss_species
        + lambda_height * loss_height
        + lambda_ndvi * loss_ndvi
    )

    return total_loss, {
        'biomass': loss_biomass.item(),
        'species': loss_species.item(),
        'height': loss_height.item(),
        'ndvi': loss_ndvi.item(),
    }
```

### ハイパーパラメータの目安

| パラメータ | 推奨範囲 | 備考 |
|-----------|---------|------|
| lambda_species | 0.1 ~ 0.3 | 最も重要な補助タスク |
| lambda_height | 0.05 ~ 0.1 | |
| lambda_ndvi | 0.05 ~ 0.1 | |

---

## 注意事項

### 補助タスクの効果検証

補助タスクが必ずしもメインタスクの性能を向上させるとは限らない。以下の比較実験を推奨:

1. **ベースライン**: 補助タスクなし
2. **Species only**: Species分類のみ追加
3. **All aux tasks**: 全補助タスク追加

CVスコアの改善を確認しながら、有効な組み合わせを選択する。

### GroupKFoldとの整合性

先の分析で確認した通り、サイト（State × Sampling_Date）とSpeciesはほぼ1対1の関係にある。
GroupKFold by siteを使用することで、「見たことないSpecies」への汎化性能もCVで評価できる。

---

## 参考

- マルチタスク学習の一般的な効果: 関連タスクを同時に学習することで、共有表現の質が向上
- 本タスクでの期待: 「草の種類を認識する」能力が「バイオマス量を予測する」能力に転移
