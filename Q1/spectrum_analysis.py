"""
直接法による片側パワースペクトル推定（テーパ＋平滑化＋95%/99%信頼区間）

串本の潮位データ（2023年）からパワースペクトルを推定し、
95%・99%信頼区間とともにlog-logプロットで図示する。

参考: instruction_Q1.md
"""

import argparse
import os
import numpy as np
from scipy.signal import detrend

# src モジュールから各関数をインポート
from src import (
    load_tide_data,
    apply_cos20_taper,
    compute_onesided_spectrum,
    smooth_spectrum,
    compute_confidence_intervals,
    plot_spectrum
)


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
):
    """
    直接法による片側パワースペクトル推定と信頼区間付きプロット

    Parameters
    ----------
    input_file : str
        入力データファイルパス
    output_dir : str
        出力ディレクトリパス
    output_filename : str
        出力画像ファイル名
    remove_mean : bool
        平均除去の有無
    remove_trend : bool
        線形トレンド除去の有無
    taper_type : str
        テーパーの種類（"cos20"）
    smoothing : bool
        平滑化の有無
    smoothing_method : str
        平滑化方法（"moving_average"）
    window_size : int
        平滑化窓サイズ（奇数推奨）
    confidence_levels : list
        信頼水準（例: [0.95, 0.99]）
    nu0 : float
        Cos20テーパーの基本等価自由度（与条件: 1.792）
    figsize : tuple
        図のサイズ
    dpi : int
        解像度

    Returns
    -------
    results : dict
        解析結果（周波数、スペクトル、信頼区間、パラメータなど）
    """
    print("=" * 60)
    print("直接法によるパワースペクトル推定")
    print("=" * 60)

    # 1. データ読み込み
    print("\n[1] データ読み込み")
    data = load_tide_data(input_file)
    N = len(data)
    dt = 1.0  # サンプリング間隔: 1時間
    print(f"  データ長: N = {N}")
    print(f"  サンプリング間隔: Δt = {dt} hour")
    print(f"  サンプリング周波数: fs = {1/dt} Hz (= 1/hour)")
    print(f"  ナイキスト周波数: {1/(2*dt):.6f} Hz")

    # 欠損値チェック
    if np.any(np.isnan(data)):
        print(f"  警告: {np.sum(np.isnan(data))} 個の欠損値を検出しました")
        # 線形補間で欠損を埋める
        mask = np.isnan(data)
        data[mask] = np.interp(np.flatnonzero(mask),
                               np.flatnonzero(~mask),
                               data[~mask])
        print(f"  → 線形補間で欠損値を埋めました")

    # 2. 前処理
    print("\n[2] 前処理")
    x = data.copy()

    if remove_trend:
        # 線形トレンド除去
        x = detrend(x, type='linear')
        print("  線形トレンド除去を実行")

    if remove_mean:
        # 平均除去
        mean_val = np.mean(x)
        x = x - mean_val
        print(f"  平均除去を実行（平均値: {mean_val:.2f}）")

    # 3. テーパリング
    print("\n[3] テーパリング")
    if taper_type == "cos20":
        x_tapered, W = apply_cos20_taper(x)
        print(f"  Cos20テーパーを適用")
        print(f"  等価自由度（平滑化なし）: ν₀ = {nu0}")
    else:
        raise ValueError(f"未対応のテーパー: {taper_type}")

    # 4. FFTによるスペクトル推定
    print("\n[4] FFTによる片側スペクトル推定")
    freq, power_raw = compute_onesided_spectrum(x_tapered, dt)
    print(f"  周波数点数: {len(freq)}")
    print(f"  周波数範囲: {freq[0]:.6f} ~ {freq[-1]:.6f} Hz")

    # 5. 平滑化
    print("\n[5] 平滑化")
    if smoothing and window_size > 1:
        power = smooth_spectrum(power_raw, window_size)
        print(f"  {smoothing_method}（窓サイズ m = {window_size}）を適用")

        # 等価自由度の計算
        # 移動平均（等重み w_i = 1/m）の場合: Σw_i^2 = 1/m
        # ν ≈ ν₀ / Σw_i^2 = ν₀ * m
        nu = nu0 * window_size
        print(f"  等価自由度: ν ≈ ν₀ × m = {nu0} × {window_size} = {nu:.2f}")
    else:
        power = power_raw
        nu = nu0
        print(f"  平滑化なし（ν = {nu}）")

    # 6. 信頼区間の計算
    print("\n[6] 信頼区間の計算")
    intervals = compute_confidence_intervals(power, nu, confidence_levels)
    for conf_level in confidence_levels:
        print(f"  {int(conf_level*100)}%信頼区間を計算")

    # 7. プロット
    print("\n[7] プロット作成")
    output_path = f"{output_dir}/{output_filename}"
    plot_spectrum(freq, power, intervals, nu, N, dt, window_size, output_path)

    # 結果の返却
    results = {
        'frequency': freq,
        'power': power,
        'power_raw': power_raw,
        'confidence_intervals': intervals,
        'N': N,
        'dt': dt,
        'nu': nu,
        'nu0': nu0,
        'window_size': window_size,
        'taper_type': taper_type
    }

    print("\n" + "=" * 60)
    print("解析完了")
    print("=" * 60)

    return results


if __name__ == "__main__":

    def parse_conf_levels(text):
        # カンマ区切り文字列を浮動小数に変換
        return [float(x) for x in text.split(',') if x.strip()]

    parser = argparse.ArgumentParser(
        description="直接法による潮位データのパワースペクトル推定"
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(script_dir, "../input/tide_Kushimoto_hourly_2023.txt")
    default_output_dir = os.path.join(script_dir, "output")

    parser.add_argument("--input_file", default=default_input,
                        help="入力データパス（デフォルト: 2023年串本潮位データ）")
    parser.add_argument("--output_dir", default=default_output_dir,
                        help="出力ディレクトリ")
    parser.add_argument("--output_filename", default=None,
                        help="出力ファイル名（未指定なら窓サイズに合わせ自動命名）")
    parser.add_argument("--window_size", type=int, default=7,
                        help="平滑化窓サイズ (奇数推奨)")
    parser.add_argument("--remove_trend", action="store_true", default=False,
                        help="線形トレンド除去を有効化")
    parser.add_argument("--no-remove_mean", dest="remove_mean",
                        action="store_false", default=True,
                        help="平均除去を無効化")
    parser.add_argument("--no-smoothing", dest="smoothing",
                        action="store_false", default=True,
                        help="平滑化を無効化")
    parser.add_argument("--confidence_levels", type=parse_conf_levels, default=[0.95, 0.99],
                        help="カンマ区切りで信頼水準を指定 (例: 0.9,0.95)")
    parser.add_argument("--nu0", type=float, default=1.792,
                        help="Cos20テーパーの基本等価自由度")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 出力ファイル名が未指定なら窓サイズで自動命名
    output_filename = args.output_filename
    if output_filename is None:
        output_filename = f"power_spectrum_cos20_smooth{args.window_size}.png"

    estimate_power_spectrum(
        input_file=args.input_file,
        output_dir=args.output_dir,
        output_filename=output_filename,
        remove_mean=args.remove_mean,
        remove_trend=args.remove_trend,
        taper_type="cos20",
        smoothing=args.smoothing,
        smoothing_method="moving_average",
        window_size=args.window_size,
        confidence_levels=args.confidence_levels,
        nu0=args.nu0,
        figsize=(10, 6),
        dpi=300
    )
