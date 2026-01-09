"""
スペクトル推定モジュール
"""

import numpy as np


def compute_onesided_spectrum(x, dt):
    """
    FFTによる片側スペクトルを計算

    Parameters
    ----------
    x : ndarray
        入力データ（テーパー適用済み）
    dt : float
        サンプリング間隔

    Returns
    -------
    freq : ndarray
        正の周波数（f > 0）
    power : ndarray
        片側パワースペクトル（正の周波数のみ、2倍化済み）

    Notes
    -----
    - rfft（実数FFT）を使用して正の周波数成分のみを取得
    - 片側スペクトルとして表示するため、DC成分とナイキスト成分を除き、
      正の周波数成分を2倍する（全パワーを正の周波数だけで表すため）
    """
    N = len(x)

    # FFT実行（実数FFT）
    X = np.fft.rfft(x)

    # 周波数軸
    freq = np.fft.rfftfreq(N, d=dt)

    # パワースペクトル |X(f)|^2
    power = np.abs(X) ** 2

    # 片側化：DC成分（freq=0）とナイキスト成分を除き、2倍する
    # DC成分: freq[0] = 0
    # ナイキスト成分: freq[-1]（Nが偶数の場合）
    power[1:-1] *= 2  # 正の周波数成分を2倍

    # DC成分（freq=0）は解析対象外なので除外（対数プロットでエラーになるため）
    freq = freq[1:]
    power = power[1:]

    return freq, power
