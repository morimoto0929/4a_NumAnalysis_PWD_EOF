"""
信頼区間計算モジュール
"""

from scipy.stats import chi2


def compute_confidence_intervals(power, nu, confidence_levels=[0.95, 0.99]):
    """
    χ²分布に基づく信頼区間を計算

    Parameters
    ----------
    power : ndarray
        パワースペクトル推定値
    nu : float
        等価自由度
    confidence_levels : list
        信頼水準のリスト（例: [0.95, 0.99]）

    Returns
    -------
    intervals : dict
        各信頼水準に対する下限・上限
        キー: 信頼水準（例: 0.95）
        値: (lower, upper) のタプル

    Notes
    -----
    スペクトル推定値 Ŝ(f) が真値 S(f) に対し、

        ν * Ŝ(f) / S(f) ~ χ²_ν

    とみなせるとき、100(1-α)%信頼区間は

        [ν * Ŝ(f) / χ²_{1-α/2, ν}, ν * Ŝ(f) / χ²_{α/2, ν}]

    で与えられる。
    """
    intervals = {}

    for conf_level in confidence_levels:
        alpha = 1 - conf_level

        # χ²分布の分位点
        chi2_lower = chi2.ppf(alpha / 2, nu)      # χ²_{α/2, ν}
        chi2_upper = chi2.ppf(1 - alpha / 2, nu)  # χ²_{1-α/2, ν}

        # 信頼区間
        lower = nu * power / chi2_upper
        upper = nu * power / chi2_lower

        intervals[conf_level] = (lower, upper)

    return intervals
