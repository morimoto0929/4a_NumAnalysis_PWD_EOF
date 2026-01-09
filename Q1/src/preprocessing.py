"""
前処理モジュール（テーパリング等）
"""

import numpy as np


def apply_cos20_taper(x):
    """
    Cos20テーパーを適用

    端の10%をcosで立上げ/立下げ、中央80%は1とする。

    Parameters
    ----------
    x : ndarray
        入力データ（長さN）

    Returns
    -------
    x_tapered : ndarray
        テーパー適用後のデータ
    W : ndarray
        テーパー窓関数

    Notes
    -----
    Cos20テーパーの等価自由度（平滑化なし）: ν₀ = 1.792（与条件）

    窓関数の定義:
    W(n) = 0.5 * [1 - cos(10π(n-0.5)/N)]  for n < N/10 + 0.5
         = 1                               for N/10 + 0.5 ≤ n ≤ 9N/10 + 0.5
         = 0.5 * [1 - cos(10π(N-n+0.5)/N)] for 9N/10 + 0.5 < n

    ここで n = 1, 2, ..., N（1-indexed）
    """
    N = len(x)
    W = np.ones(N)

    # n = 1, 2, ..., N として計算（1-indexed）
    n = np.arange(1, N + 1)

    # 端の10%を特定（立ち上がり部分）
    taper_length = N / 10
    rising_mask = n < (taper_length + 0.5)
    W[rising_mask] = 0.5 * (1 - np.cos(10 * np.pi * (n[rising_mask] - 0.5) / N))

    # 端の10%を特定（立ち下がり部分）
    falling_mask = n > (9 * N / 10 + 0.5)
    W[falling_mask] = 0.5 * (1 - np.cos(10 * np.pi * (N - n[falling_mask] + 0.5) / N))

    # テーパー適用
    x_tapered = W * x

    return x_tapered, W
