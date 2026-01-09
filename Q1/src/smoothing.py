"""
平滑化モジュール
"""

import numpy as np


def smooth_spectrum(power, window_size):
    """
    移動平均によるスペクトル平滑化

    Parameters
    ----------
    power : ndarray
        元のパワースペクトル
    window_size : int
        移動平均の窓サイズ（奇数推奨: 5, 7, 9, 11など）

    Returns
    -------
    power_smoothed : ndarray
        平滑化後のパワースペクトル

    Notes
    -----
    - 等重み移動平均: w_i = 1/m
    - 両端は窓サイズを調整して計算（numpy.convolveのmodeオプション）
    """
    if window_size == 1:
        return power

    # 等重み移動平均
    window = np.ones(window_size) / window_size

    # convolveで移動平均を計算（mode='same'で同じ長さを保つ）
    power_smoothed = np.convolve(power, window, mode='same')

    return power_smoothed
