# Q1: 直接法による片側パワースペクトル推定

串本の潮位データ（2023年）から、直接法（FFTに基づく方法）によりパワースペクトルを推定し、95%・99%信頼区間とともにlog-logプロットで図示します。

## ファイル構成

```
Q1/
├── spectrum_analysis.py      # メイン実行スクリプト
├── README.md                  # このファイル
├── src/                       # モジュール群
│   ├── __init__.py           # パッケージ初期化
│   ├── data_loader.py        # データ読み込み
│   ├── preprocessing.py      # 前処理（テーパリング）
│   ├── spectrum.py           # スペクトル推定
│   ├── smoothing.py          # 平滑化処理
│   ├── confidence.py         # 信頼区間計算
│   └── plotting.py           # プロット作成
└── output/                    # 出力ディレクトリ
    └── power_spectrum_*.png  # 生成されたスペクトル図
```

### モジュールの役割

- **data_loader.py**: 潮位データファイルの読み込み
- **preprocessing.py**: Cos20テーパーの適用
- **spectrum.py**: FFTによる片側スペクトル推定
- **smoothing.py**: 移動平均による平滑化
- **confidence.py**: χ²分布に基づく信頼区間の計算
- **plotting.py**: log-logプロットの作成

## 実行方法

### 1. プロジェクトルートでuv環境をセットアップ（初回のみ）

```bash
cd /path/to/4a_NumAnalysis_PWD_EOF
uv sync
```

### 2. スクリプトの実行

#### デフォルトパラメータで実行

```bash
uv run Q1/spectrum_analysis.py
```

#### コマンドライン引数で実行（例：window_size=15）

```bash
uv run Q1/spectrum_analysis.py --window_size 15
```

#### 利用可能なコマンドライン引数

```bash
uv run Q1/spectrum_analysis.py --help
```

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--input_file` | `../input/tide_Kushimoto_hourly_2023.txt` | 入力データパス |
| `--output_dir` | `output` | 出力ディレクトリ |
| `--output_filename` | 自動命名 | 出力ファイル名（未指定時は窓サイズに応じて命名） |
| `--window_size` | `7` | 平滑化窓サイズ（奇数推奨） |
| `--remove_trend` | `False` | 線形トレンド除去を有効化 |
| `--no-remove_mean` | - | 平均除去を無効化（デフォルトは有効） |
| `--no-smoothing` | - | 平滑化を無効化（デフォルトは有効） |
| `--confidence_levels` | `0.95,0.99` | カンマ区切りで信頼水準を指定 |
| `--nu0` | `1.792` | Cos20テーパーの基本等価自由度 |

実行すると、`Q1/output/` ディレクトリに以下のファイルが生成されます：
- `power_spectrum_cos20_smooth{window_size}.png` - パワースペクトル図（log-logプロット、信頼区間付き）

## データ条件

- **データファイル**: `input/tide_Kushimoto_hourly_2023.txt`
- **データ長**: N = 8760（2023年1月1日〜12月31日の毎時データ）
- **サンプリング間隔**: Δt = 1.0 hour
- **サンプリング周波数**: fs = 1.0 Hz（1/hour）
- **ナイキスト周波数**: 0.5 Hz（1回/2時間）

## メイン関数: `estimate_power_spectrum()`

### 関数シグネチャ

```python
def estimate_power_spectrum(
    input_file,
    output_dir,
    output_filename="power_spectrum.png",
    remove_mean=True,
    remove_trend=False,
    taper_type="cos20",
    smoothing=True,
    smoothing_method="moving_average",
    window_size=7,
    confidence_levels=[0.95, 0.99],
    nu0=1.792,
    figsize=(10, 6),
    dpi=300
)
```

### 引数の説明

#### 必須引数

| 引数 | 型 | 説明 |
|------|-----|------|
| `input_file` | `str` | 入力データファイルのパス（例: `"../input/tide_Kushimoto_hourly_2023.txt"`） |
| `output_dir` | `str` | 出力先ディレクトリのパス（例: `"output"`） |

#### オプション引数

| 引数 | 型 | デフォルト値 | 説明 |
|------|-----|-------------|------|
| `output_filename` | `str` | `"power_spectrum.png"` | 出力画像ファイル名 |
| `remove_mean` | `bool` | `True` | 平均除去を行うかどうか |
| `remove_trend` | `bool` | `False` | 線形トレンド除去を行うかどうか |
| `taper_type` | `str` | `"cos20"` | テーパーの種類（現在は`"cos20"`のみ対応） |
| `smoothing` | `bool` | `True` | 平滑化を行うかどうか |
| `smoothing_method` | `str` | `"moving_average"` | 平滑化方法（現在は移動平均のみ対応） |
| `window_size` | `int` | `7` | 平滑化の窓サイズ（奇数推奨: 5, 7, 9, 11など） |
| `confidence_levels` | `list` | `[0.95, 0.99]` | 表示する信頼水準のリスト |
| `nu0` | `float` | `1.792` | Cos20テーパーの基本等価自由度（与条件） |
| `figsize` | `tuple` | `(10, 6)` | 図のサイズ（インチ） |
| `dpi` | `int` | `300` | 図の解像度（dots per inch） |

### 戻り値

```python
{
    'frequency': ndarray,           # 周波数軸（Hz）
    'power': ndarray,              # 平滑化後のパワースペクトル
    'power_raw': ndarray,          # 平滑化前のパワースペクトル
    'confidence_intervals': dict,   # 信頼区間（{0.95: (lower, upper), 0.99: (lower, upper)}）
    'N': int,                      # データ長
    'dt': float,                   # サンプリング間隔
    'nu': float,                   # 等価自由度（平滑化後）
    'nu0': float,                  # 基本等価自由度（平滑化前）
    'window_size': int,            # 平滑化窓サイズ
    'taper_type': str              # テーパー種類
}
```

## 解析手順

### 1. データ読み込みと前処理

- 潮位データ（第2列）を読み込み
- 欠損値がある場合は線形補間で補完
- **平均除去**: `x ← x - mean(x)` を実行（`remove_mean=True`）
- 線形トレンド除去はデフォルトでは行わない（`remove_trend=False`）

### 2. テーパリング（Cos20テーパー）

端の10%を cosine 窓で立ち上げ/立ち下げ、中央80%を1とする：

```
W(n) = 0.5 * [1 - cos(10π(n-0.5)/N)]  (n < N/10 + 0.5)
     = 1                               (N/10 + 0.5 ≤ n ≤ 9N/10 + 0.5)
     = 0.5 * [1 - cos(10π(N-n+0.5)/N)] (9N/10 + 0.5 < n)
```

**与条件**: Cos20テーパーの等価自由度（平滑化なし）= **ν₀ = 1.792**

### 3. FFTによる片側スペクトル推定

- `numpy.fft.rfft()` を使用して実数FFTを実行
- 正の周波数成分のみを抽出
- **片側化**: DC成分とナイキスト成分を除き、正の周波数成分を2倍
  - 理由: 全パワーを正の周波数だけで表すための表示上の換算
- DC成分（周波数0）は対数プロットでエラーになるため除外

### 4. 平滑化（移動平均）

スペクトルの分散を低減し、特徴を見やすくするために平滑化を実行。

- **方法**: 等重み移動平均（Daniell型）
- **窓サイズ**: `window_size = 7`（デフォルト）
  - 奇数推奨: 5, 7, 9, 11 など
  - 小さい値から試し、ピーク形状が不自然に潰れない範囲で採用
- **トレードオフ**:
  - 窓サイズが大きい → 分散は減少、周波数分解能は低下
  - 窓サイズが小さい → 周波数分解能は保持、分散は大きいまま

#### パラメータ選定理由（window_size = 7）

- `m = 5`: 平滑化が不十分で、スペクトルのばらつきが目立つ
- `m = 7`: 適度な平滑化で、ピークを保ちつつ分散を低減 **（採用）**
- `m = 9, 11`: 過度な平滑化で、鋭いピークが潰れる可能性

### 5. 等価自由度の計算

平滑化による等価自由度の増加を考慮：

```
ν ≈ ν₀ / Σw_i²
```

移動平均（等重み `w_i = 1/m`）の場合：

```
Σw_i² = m × (1/m)² = 1/m

ν ≈ ν₀ / (1/m) = ν₀ × m
```

**デフォルト値での計算**:
```
ν = 1.792 × 7 = 12.54
```

### 6. 信頼区間の計算（χ²分布に基づく）

スペクトル推定値 Ŝ(f) が真値 S(f) に対し、

```
ν · Ŝ(f) / S(f) ~ χ²_ν
```

とみなせるとき、100(1-α)% 信頼区間は以下で与えられる：

```
[ν · Ŝ(f) / χ²_{1-α/2, ν}, ν · Ŝ(f) / χ²_{α/2, ν}]
```

**デフォルトで計算される信頼区間**:
- 95%信頼区間: α = 0.05
- 99%信頼区間: α = 0.01

### 7. 図示（log-logプロット）

- **横軸**: 周波数（Hz）- 対数スケール
- **縦軸**: パワー（任意単位）- 対数スケール
- **表示内容**:
  - 推定スペクトル曲線（青線）
  - 95%信頼区間（緑の破線と半透明領域）
  - 99%信頼区間（赤の点線と半透明領域）
- **注記**: データ長、サンプリング条件、テーパー、平滑化パラメータ、等価自由度

## カスタマイズ例

### 平滑化窓サイズを変更

```python
results = estimate_power_spectrum(
    input_file=input_file,
    output_dir=output_dir,
    window_size=9,  # より強い平滑化
    output_filename="power_spectrum_smooth9.png"
)
```

### 平滑化なしで実行

```python
results = estimate_power_spectrum(
    input_file=input_file,
    output_dir=output_dir,
    smoothing=False,  # 平滑化を無効化
    output_filename="power_spectrum_no_smooth.png"
)
```

### トレンド除去を有効化

```python
results = estimate_power_spectrum(
    input_file=input_file,
    output_dir=output_dir,
    remove_trend=True,  # 線形トレンド除去を実行
    output_filename="power_spectrum_detrend.png"
)
```

## 出力例

実行すると、以下のような情報がコンソールに表示されます：

```
============================================================
直接法によるパワースペクトル推定
============================================================

[1] データ読み込み
  データ長: N = 8760
  サンプリング間隔: Δt = 1.0 hour
  サンプリング周波数: fs = 1.0 Hz (= 1/hour)
  ナイキスト周波数: 0.500000 Hz

[2] 前処理
  平均除去を実行（平均値: 196.41）

[3] テーパリング
  Cos20テーパーを適用
  等価自由度（平滑化なし）: ν₀ = 1.792

[4] FFTによる片側スペクトル推定
  周波数点数: 4380
  周波数範囲: 0.000114 ~ 0.500000 Hz

[5] 平滑化
  moving_average（窓サイズ m = 7）を適用
  等価自由度: ν ≈ ν₀ × m = 1.792 × 7 = 12.54

[6] 信頼区間の計算
  95%信頼区間を計算
  99%信頼区間を計算

[7] プロット作成
スペクトル図を保存しました: output/power_spectrum_cos20_smooth7.png

============================================================
解析完了
============================================================
```

## 結果の解釈

生成されたスペクトル図から、以下のような特徴が観察されます：

### 主要なピーク

- **日周潮（K1, O1）**: 周波数 ~ 0.04 Hz付近（周期 ~ 24時間）
- **半日周潮（M2, S2）**: 周波数 ~ 0.08 Hz付近（周期 ~ 12時間）

これらは潮汐現象の主要な周期成分を示しています。

### スペクトルの傾き

- 低周波数帯: 潮汐の主要成分による強いピーク
- 中周波数帯: パワーが周波数に対して減少
- 高周波数帯: より急な傾きで減衰（高周波ノイズの影響）

### 信頼区間の解釈

- 平滑化により等価自由度が増加（ν = 12.54）
- 信頼区間の幅が狭くなり、推定精度が向上
- 95%CIと99%CIの範囲から、スペクトル推定の不確実性を評価可能

## 参考

- 課題詳細: [instruction_Q1.md](../instruction_Q1.md)
- 入力データ: [tide_Kushimoto_hourly_2023.txt](../input/tide_Kushimoto_hourly_2023.txt)
