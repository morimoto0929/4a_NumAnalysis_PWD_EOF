"""
データ読み込みモジュール
"""

import numpy as np


def load_tide_data(filepath):
    """
    潮位データを読み込む

    Parameters
    ----------
    filepath : str
        データファイルのパス

    Returns
    -------
    data : ndarray
        潮位データ（単位: cm）
    """
    # 第2列（潮位）のみを読み込む
    # skiprows=1: ヘッダー行がある場合はスキップ（実際のデータ構造に応じて調整）
    try:
        data = np.loadtxt(filepath, usecols=1)
    except:
        # ヘッダーなしの場合
        data = np.loadtxt(filepath, usecols=1, skiprows=0)

    return data
