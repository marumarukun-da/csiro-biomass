## **データの概要：**

- このデータセットは、牧草地の画像と対応する実測バイオマスデータを用いて、以下の5種類の牧草構成要素を予測するために設計されています。
    1. **Dry_Green_g**：クローバーを除いた乾燥した緑草のバイオマス量
    2. **Dry_Dead_g**：枯死植物体のバイオマス量
    3. **Dry_Clover_g**：クローバー部分の乾燥バイオマス量
    4. **GDM_g**：Green Dry Matter（緑乾物量）
    5. **Dry_Total_g**：総乾燥バイオマス量（上記を含む全体）
- これらの指標を正確に推定することで、農家や研究者は牧草の成長状況を把握し、放牧時期や飼料供給の最適化、持続的な畜産管理に役立てることができます。
- **上記指標は下記のバイオマス構成式を満たすため、予測はDry_Total_g, GDM_g, Dry_Green_gの3つのみで良い**
    - $\text{Dry\_Total\_g} \approx \text{GDM\_g} + \text{Dry\_Dead\_g}$
    - $\text{GDM\_g} \approx \text{Dry\_Green\_g} + \text{Dry\_Clover\_g}$
    - $\text{Dry\_Total\_g} \approx \text{Dry\_Green\_g} + \text{Dry\_Clover\_g} + \text{Dry\_Dead\_g}$

---

## **各ファイルの詳細な説明：**

- **`train/`**
    
    トレーニング用の牧草画像（JPEG形式）を格納したディレクトリ。
    
    `train.csv` の `image_path` カラムから参照されます。
    
    全357枚
        
- **`test/`**
    
    テスト用画像を格納するディレクトリ。スコアリング時にのみ利用可能で、`test.csv` の `image_path` に対応します（通常は非公開）。
    
- **`train.csv`**
    - **データ例**
        
        
        | sample_id | image_path | Sampling_Date | State | Species | Pre_GSHH_NDVI | Height_Ave_cm | target_name | target |
        | --- | --- | --- | --- | --- | --- | --- | --- | --- |
        | ID1011485656__Dry_Clover_g | train/ID1011485656.jpg | 2015/9/4 | Tas | Ryegrass_Clover | 0.62 | 4.6667 | Dry_Clover_g | 0 |
        | ID1011485656__Dry_Dead_g | train/ID1011485656.jpg | 2015/9/4 | Tas | Ryegrass_Clover | 0.62 | 4.6667 | Dry_Dead_g | 31.9984 |
        | ID1011485656__Dry_Green_g | train/ID1011485656.jpg | 2015/9/4 | Tas | Ryegrass_Clover | 0.62 | 4.6667 | Dry_Green_g | 16.2751 |
        | ID1011485656__Dry_Total_g | train/ID1011485656.jpg | 2015/9/4 | Tas | Ryegrass_Clover | 0.62 | 4.6667 | Dry_Total_g | 48.2735 |
        | ID1011485656__GDM_g | train/ID1011485656.jpg | 2015/9/4 | Tas | Ryegrass_Clover | 0.62 | 4.6667 | GDM_g | 16.275 |
        | ID1012260530__Dry_Clover_g | train/ID1012260530.jpg | 2015/4/1 | NSW | Lucerne | 0.55 | 16 | Dry_Clover_g | 0 |
        | ID1012260530__Dry_Dead_g | train/ID1012260530.jpg | 2015/4/1 | NSW | Lucerne | 0.55 | 16 | Dry_Dead_g | 0 |
        | ID1012260530__Dry_Green_g | train/ID1012260530.jpg | 2015/4/1 | NSW | Lucerne | 0.55 | 16 | Dry_Green_g | 7.6 |
        | ID1012260530__Dry_Total_g | train/ID1012260530.jpg | 2015/4/1 | NSW | Lucerne | 0.55 | 16 | Dry_Total_g | 7.6 |
        | ID1012260530__GDM_g | train/ID1012260530.jpg | 2015/4/1 | NSW | Lucerne | 0.55 | 16 | GDM_g | 7.6 |
        | ID1025234388__Dry_Clover_g | train/ID1025234388.jpg | 2015/9/1 | WA | SubcloverDalkeith | 0.38 | 1 | Dry_Clover_g | 6.05 |
        | ID1025234388__Dry_Dead_g | train/ID1025234388.jpg | 2015/9/1 | WA | SubcloverDalkeith | 0.38 | 1 | Dry_Dead_g | 0 |
        | ID1025234388__Dry_Green_g | train/ID1025234388.jpg | 2015/9/1 | WA | SubcloverDalkeith | 0.38 | 1 | Dry_Green_g | 0 |
        | ID1025234388__Dry_Total_g | train/ID1025234388.jpg | 2015/9/1 | WA | SubcloverDalkeith | 0.38 | 1 | Dry_Total_g | 6.05 |
        | ID1025234388__GDM_g | train/ID1025234388.jpg | 2015/9/1 | WA | SubcloverDalkeith | 0.38 | 1 | GDM_g | 6.05 |
    
    学習データのメタ情報およびターゲット（実測バイオマス値）が含まれています。
    
    各カラムの意味は以下の通りです：
    
    - **sample_id**：各サンプルのユニークID
    - **image_path**：画像への相対パス（例：`images/ID1098771283.jpg`）
    - **Sampling_Date**：サンプル採取日
    - **State**：採取地のオーストラリア州名
    - **Species**：牧草の種構成（バイオマス比順にアンダースコア区切りで記載）
    - **Pre_GSHH_NDVI**：NDVI（正規化植生指数）測定値（GreenSeekerによる）
    - **Height_Ave_cm**：平均牧草高（cm）
    - **target_name**：バイオマス成分名（Dry_Green_g / Dry_Dead_g / Dry_Clover_g / GDM_g / Dry_Total_g）
    - **target**：実測バイオマス値（単位：グラム）
- **`test.csv`**
    - **実際のデータ**
        
        
        | sample_id | image_path | target_name |
        | --- | --- | --- |
        | ID1001187975__Dry_Clover_g | test/ID1001187975.jpg | Dry_Clover_g |
        | ID1001187975__Dry_Dead_g | test/ID1001187975.jpg | Dry_Dead_g |
        | ID1001187975__Dry_Green_g | test/ID1001187975.jpg | Dry_Green_g |
        | ID1001187975__Dry_Total_g | test/ID1001187975.jpg | Dry_Total_g |
        | ID1001187975__GDM_g | test/ID1001187975.jpg | GDM_g |
    
    テストデータのメタ情報をまとめたファイル。予測すべき全ての `(image_path, target_name)` の組み合わせを含みます。
    
    - **sample_id**：各予測行を識別するユニークID
    - **image_path**：画像への相対パス（例：`test/ID1001187975.jpg`）
    - **target_name**：予測対象のバイオマス成分名（上記5種類のいずれか）
    
    ※テストセットには800枚以上の画像が含まれます。
    
- **`sample_submission.csv`**
    - **実際のデータ**
        
        
        | sample_id | target |
        | --- | --- |
        | ID1001187975__Dry_Clover_g | 0 |
        | ID1001187975__Dry_Dead_g | 0 |
        | ID1001187975__Dry_Green_g | 0 |
        | ID1001187975__Dry_Total_g | 0 |
        | ID1001187975__GDM_g | 0 |
    
    提出フォーマットのサンプルです。
    
    - **sample_id**：`test.csv` と同一のID
    - **target**：予測したバイオマス値（単位：グラム）
    
    提出時は `sample_submission.csv` と同一形式で作成する必要があります（各画像につき5行）。
    

---

## **まとめ：**

このデータセットは「画像・NDVI・高さ・種構成」といった複数モダリティ情報を組み合わせて、牧草地の**バイオマスを定量推定するマルチターゲット回帰タスク**です。データ構造はシンプルながら、画像と数値データの両方を扱うマルチモーダル学習が求められる実践的な構成となっています。
