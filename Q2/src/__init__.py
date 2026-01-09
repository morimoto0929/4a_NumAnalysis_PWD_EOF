"""
Q2: PCA/EOF Analysis Package

15地点の秋季平均気温の年々変動に対する主成分分析（PCA/EOF解析）

モジュール構成:
- data_loader: データ読み込み
- seasonal_average: 季節平均の計算
- pca_eof: PCA/EOF解析の実行
- visualization: 結果の可視化
"""

__version__ = "1.0.0"

# 各モジュールから主要な関数・クラスをインポート
from .data_loader import load_temperature_data
from .seasonal_average import calculate_seasonal_average
from .pca_eof import perform_pca_eof, compute_north_error_bars
from .visualization import (
    plot_pc_timeseries,
    plot_eof_pattern,
    plot_eof_map,
    plot_variance_fraction_with_error_bars,
    print_north_interpretation
)

__all__ = [
    "load_temperature_data",
    "calculate_seasonal_average",
    "perform_pca_eof",
    "compute_north_error_bars",
    "plot_pc_timeseries",
    "plot_eof_pattern",
    "plot_eof_map",
    "plot_variance_fraction_with_error_bars",
    "print_north_interpretation",
]
