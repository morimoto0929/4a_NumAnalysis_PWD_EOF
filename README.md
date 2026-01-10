# 4a_NumAnalysis_PWD_EOF

本リポジトリは数値解析の課題用コードをまとめています。
- Q1: 串本潮位（2023年）の直接法による片側パワースペクトル推定
- Q2: 問２の処理のソースコードと出力

## 環境セットアップ
- Python 3.13 以上
- 依存関係は `pyproject.toml` / `uv.lock` で管理。初回はプロジェクトルートで `uv sync` を実行してください。

## ディレクトリ構造
```
.
├── Q1/                     # パワースペクトル解析（詳細は Q1/README.md）
│   ├── spectrum_analysis.py
│   ├── src/
│   └── output/             # 回答出力画像
├── Q2/                     # PCA/EOF 解析（詳細は Q2/README.md）
│   ├── pca_analysis.py
│   ├── src/
│   └── output/             # 回答出力CSV・図
├── pyproject.toml          # 依存関係定義
└── uv.lock                 # ロックファイル
```

## 実行例
input/ に入力となる該当ファイルを置いた後、
- Q1: `uv run Q1/spectrum_analysis.py`
- Q2: `uv run Q2/pca_analysis.py`
（その他のオプションは各ディレクトリ内の README を参照してください）

