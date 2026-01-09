"""
15地点・秋季（9-11月平均）気温の年々変動に対する主成分分析（PCA/EOF解析）

気象庁の15地点の気温データを用いて、秋季平均気温の年々変動に対してPCA（EOF解析）を行う。
第1〜第3モードについて、以下を求めて提示する:
- 有次元化した因子負荷量（EOFパターン；単位つき）
- 規格化（標準化）された主成分時系列（PC時系列；無次元）
- 各モードの寄与率（%）
- 各モードのPC時系列のグラフ
- EOF（因子負荷量）を数値（15地点の表）または地図で表示

参考: instruction_Q2.md
"""

import argparse
import os
from pathlib import Path

# 各モジュールをインポート
from src.data_loader import load_temperature_data, get_station_info
from src.seasonal_average import (
    calculate_seasonal_average,
    remove_incomplete_cases,
    compute_anomalies
)
from src.pca_eof import perform_pca_eof, save_pca_results
from src.visualization import (
    plot_pc_timeseries,
    plot_eof_pattern,
    plot_eof_map,
    print_eof_table,
    print_variance_table
)


def run_pca_eof_analysis(
    input_file,
    output_dir,
    season='SON',
    n_modes=3,
    min_months=3,
    remove_incomplete=True,
    remove_trend=False,
    adjust_sign=True,
    save_csv=True,
    plot_pc=True,
    plot_eof_bar=True,
    plot_eof_map_flag=True,
    separate_pc_plots=False,
    figsize_pc=(12, 8),
    figsize_eof_bar=(10, 6),
    figsize_eof_map=(15, 5),
    dpi=300
):
    """
    PCA/EOF解析のメイン処理

    Parameters
    ----------
    input_file : str
        入力データファイルのパス（MonthlyMeanTemp15points.txt）
    output_dir : str
        出力ディレクトリ
    season : str
        季節の指定（'SON', 'DJF', 'MAM', 'JJA'）
    n_modes : int
        計算するモード数
    min_months : int
        季節平均を計算するための最低月数
    remove_incomplete : bool
        欠損がある年を除去するか
    remove_trend : bool
        線形トレンドを除去するか
    adjust_sign : bool
        EOFとPCの符号を調整するか
    save_csv : bool
        結果をCSVで保存するか
    plot_pc : bool
        PC時系列をプロットするか
    plot_eof_bar : bool
        EOFパターンを棒グラフでプロットするか
    plot_eof_map_flag : bool
        EOFパターンを地図でプロットするか
    separate_pc_plots : bool
        PC時系列を各モードで別々にプロットするか
    figsize_pc : tuple
        PC時系列の図のサイズ
    figsize_eof_bar : tuple
        EOFパターン棒グラフの図のサイズ
    figsize_eof_map : tuple
        EOFパターン地図の図のサイズ
    dpi : int
        図の解像度

    Returns
    -------
    results : dict
        PCA/EOF解析の結果
    """
    print("=" * 70)
    print("15地点・秋季平均気温の年々変動に対するPCA/EOF解析")
    print("=" * 70)

    # 出力ディレクトリの作成
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. データ読み込み
    print("\n[1] データ読み込み")
    df, metadata = load_temperature_data(input_file)

    # 観測所情報の表示
    station_info = get_station_info(metadata)
    print("\n観測所情報:")
    print(station_info.to_string(index=False))

    # 2. 季節平均の計算
    print("\n" + "=" * 70)
    print("[2] 季節平均の計算")
    print("=" * 70)
    seasonal_avg = calculate_seasonal_average(df, season=season, min_months=min_months)

    # 3. 欠損値の処理（完全ケースの抽出）
    if remove_incomplete:
        print("\n" + "=" * 70)
        print("[3] 欠損値の処理（完全ケース抽出）")
        print("=" * 70)
        seasonal_avg = remove_incomplete_cases(seasonal_avg)
    else:
        print("\n[3] 欠損値の処理: スキップ（欠損を含む年も使用）")

    # 4. 年々偏差（アノマリー）の計算
    print("\n" + "=" * 70)
    print("[4] 年々偏差（アノマリー）の計算")
    print("=" * 70)
    anomalies, climatology = compute_anomalies(seasonal_avg, remove_trend=remove_trend)

    # 気候値の表示
    print("\n気候値（時間平均）:")
    for station, value in climatology.items():
        print(f"  {station:15s}: {value:6.2f} °C")

    # 5. PCA/EOF解析
    print("\n" + "=" * 70)
    print("[5] PCA/EOF解析")
    print("=" * 70)
    results = perform_pca_eof(anomalies, n_modes=n_modes, adjust_sign=adjust_sign)

    # 6. 結果の保存
    print("\n" + "=" * 70)
    print("[6] 結果の保存")
    print("=" * 70)

    if save_csv:
        save_pca_results(results, output_dir, prefix='pca_eof')

    # 7. 可視化
    print("\n" + "=" * 70)
    print("[7] 可視化")
    print("=" * 70)

    # PC時系列のプロット
    if plot_pc:
        plot_pc_timeseries(
            results,
            output_dir,
            filename='PC_timeseries.png',
            separate_plots=separate_pc_plots,
            figsize=figsize_pc,
            dpi=dpi
        )

    # EOFパターンの棒グラフ
    if plot_eof_bar:
        plot_eof_pattern(
            results,
            metadata,
            output_dir,
            filename='EOF_pattern_bar.png',
            figsize=figsize_eof_bar,
            dpi=dpi
        )

    # EOFパターンの地図
    if plot_eof_map_flag:
        plot_eof_map(
            results,
            metadata,
            output_dir,
            filename_prefix='EOF_map',
            figsize=figsize_eof_map,
            dpi=dpi
        )

    # 8. 結果の表示
    print("\n" + "=" * 70)
    print("[8] 結果の表示")
    print("=" * 70)

    # 寄与率の表示
    print_variance_table(results)

    # EOFパターンの数値表示
    print_eof_table(results, metadata)

    print("\n" + "=" * 70)
    print("解析完了")
    print("=" * 70)
    print(f"\n結果は以下のディレクトリに保存されました:")
    print(f"  {output_dir.absolute()}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="15地点の秋季平均気温に対するPCA/EOF解析",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # スクリプトのディレクトリとデフォルトパス
    script_dir = Path(__file__).parent
    default_input = script_dir.parent / "input" / "MonthlyMeanTemp15points.txt"
    default_output = script_dir / "output"

    # 入出力パラメータ
    parser.add_argument(
        "--input_file",
        type=str,
        default=str(default_input),
        help="入力データファイルのパス"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(default_output),
        help="出力ディレクトリ"
    )

    # 解析パラメータ
    parser.add_argument(
        "--season",
        type=str,
        default="SON",
        choices=["SON", "DJF", "MAM", "JJA"],
        help="季節の指定（SON: 9-11月, DJF: 12-2月, MAM: 3-5月, JJA: 6-8月）"
    )
    parser.add_argument(
        "--n_modes",
        type=int,
        default=3,
        help="計算するモード数"
    )
    parser.add_argument(
        "--min_months",
        type=int,
        default=3,
        help="季節平均を計算するための最低月数（3: 完全ケースのみ）"
    )
    parser.add_argument(
        "--keep_incomplete",
        action="store_true",
        help="欠損がある年も使用する（デフォルト: 完全ケースのみ）"
    )
    parser.add_argument(
        "--remove_trend",
        action="store_true",
        help="線形トレンドを除去する"
    )
    parser.add_argument(
        "--no_adjust_sign",
        action="store_true",
        help="EOFとPCの符号調整を無効化"
    )

    # 出力オプション
    parser.add_argument(
        "--no_save_csv",
        action="store_true",
        help="CSV保存を無効化"
    )
    parser.add_argument(
        "--no_plot_pc",
        action="store_true",
        help="PC時系列のプロットを無効化"
    )
    parser.add_argument(
        "--no_plot_eof_bar",
        action="store_true",
        help="EOFパターン棒グラフのプロットを無効化"
    )
    parser.add_argument(
        "--no_plot_eof_map",
        action="store_true",
        help="EOFパターン地図のプロットを無効化"
    )
    parser.add_argument(
        "--separate_pc_plots",
        action="store_true",
        help="PC時系列を各モードで別々にプロット"
    )

    # 図のパラメータ
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="図の解像度"
    )

    args = parser.parse_args()

    # 解析の実行
    results = run_pca_eof_analysis(
        input_file=args.input_file,
        output_dir=args.output_dir,
        season=args.season,
        n_modes=args.n_modes,
        min_months=args.min_months,
        remove_incomplete=not args.keep_incomplete,
        remove_trend=args.remove_trend,
        adjust_sign=not args.no_adjust_sign,
        save_csv=not args.no_save_csv,
        plot_pc=not args.no_plot_pc,
        plot_eof_bar=not args.no_plot_eof_bar,
        plot_eof_map_flag=not args.no_plot_eof_map,
        separate_pc_plots=args.separate_pc_plots,
        dpi=args.dpi
    )
