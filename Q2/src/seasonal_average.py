"""
秋季（SON: 9-11月）平均気温の計算モジュール

各年の9月・10月・11月の平均気温を計算し、年×地点の行列を作成する。
"""

import pandas as pd
import numpy as np


def calculate_seasonal_average(df, season='SON', min_months=3):
    """
    季節平均気温を計算（デフォルトは秋季: 9-11月）

    Parameters
    ----------
    df : pd.DataFrame
        月平均気温データ（DatetimeIndex, columns=観測所名）
    season : str
        季節の指定（'SON': 9-11月、'DJF': 12-2月、'MAM': 3-5月、'JJA': 6-8月）
    min_months : int
        季節平均を計算するための最低月数（デフォルト: 3）
        - 3: 3ヶ月すべて揃っている場合のみ計算（完全ケース）
        - 2: 2ヶ月以上あれば計算
        - 1: 1ヶ月でも計算

    Returns
    -------
    seasonal_avg : pd.DataFrame
        季節平均気温（行: 年、列: 観測所名）
        単位: °C

    Notes
    -----
    - 秋季（SON）は同一年の 9月・10月・11月の平均
    - 冬季（DJF）は前年12月、当年1月・2月の平均（年をまたぐ）
    - 欠損がある場合、min_months 以上のデータがある場合のみ平均を計算
    - min_months を満たさない場合は NaN とする
    """
    # 季節の定義
    season_months = {
        'SON': [9, 10, 11],     # 秋季
        'DJF': [12, 1, 2],      # 冬季（年またぎ）
        'MAM': [3, 4, 5],       # 春季
        'JJA': [6, 7, 8]        # 夏季
    }

    if season not in season_months:
        raise ValueError(f"未対応の季節: {season}。'SON', 'DJF', 'MAM', 'JJA' のいずれかを指定してください。")

    months = season_months[season]

    print(f"\n[季節平均の計算]")
    print(f"  季節: {season} (月: {months})")
    print(f"  最低必要月数: {min_months}")

    # 年と月の列を追加
    df_with_ym = df.copy()
    df_with_ym['year'] = df_with_ym.index.year
    df_with_ym['month'] = df_with_ym.index.month

    # 冬季（DJF）の場合、12月は翌年にカウント
    if season == 'DJF':
        # 12月のデータを翌年として扱う
        df_with_ym.loc[df_with_ym['month'] == 12, 'year'] += 1

    # 指定された季節の月のデータを抽出
    seasonal_data = df_with_ym[df_with_ym['month'].isin(months)]

    # 観測所名のリスト
    stations = [col for col in df.columns if col not in ['year', 'month']]

    # 年ごと・地点ごとに季節平均を計算
    def safe_mean(x):
        """欠損値を考慮した平均計算"""
        valid_count = x.notna().sum()
        if valid_count >= min_months:
            return x.mean()
        else:
            return np.nan

    # 年ごとにグループ化して平均を計算
    seasonal_avg = seasonal_data.groupby('year')[stations].apply(
        lambda group: group.apply(safe_mean, axis=0)
    )

    # 年をインデックスとして整形
    seasonal_avg.index.name = 'year'

    # 統計情報の出力
    n_years = len(seasonal_avg)
    n_stations = len(stations)
    n_complete_cases = seasonal_avg.notna().all(axis=1).sum()
    n_missing_total = seasonal_avg.isna().sum().sum()

    print(f"  年数: {n_years}")
    print(f"  観測所数: {n_stations}")
    print(f"  完全ケース（全地点でデータあり）: {n_complete_cases} 年")

    if n_missing_total > 0:
        print(f"  警告: {n_missing_total} 個の欠損値があります")
        # 年ごとの欠損数を表示（欠損がある年のみ）
        missing_per_year = seasonal_avg.isna().sum(axis=1)
        if missing_per_year.sum() > 0:
            print("  年別欠損地点数（欠損がある年のみ）:")
            for year, count in missing_per_year[missing_per_year > 0].items():
                print(f"    {year}: {count} 地点")

    return seasonal_avg


def remove_incomplete_cases(seasonal_avg):
    """
    欠損値がある年を除去（完全ケースのみを残す）

    Parameters
    ----------
    seasonal_avg : pd.DataFrame
        季節平均気温（行: 年、列: 観測所名）

    Returns
    -------
    complete_cases : pd.DataFrame
        完全ケース（全地点でデータがある年のみ）
    """
    # 全地点でデータがある年のみを抽出
    complete_mask = seasonal_avg.notna().all(axis=1)
    complete_cases = seasonal_avg[complete_mask]

    n_removed = len(seasonal_avg) - len(complete_cases)

    print(f"\n[完全ケースの抽出]")
    print(f"  元の年数: {len(seasonal_avg)}")
    print(f"  完全ケース: {len(complete_cases)}")
    print(f"  除外された年数: {n_removed}")

    if n_removed > 0:
        removed_years = seasonal_avg[~complete_mask].index.tolist()
        print(f"  除外された年: {removed_years}")

    return complete_cases


def compute_anomalies(seasonal_avg, remove_trend=False):
    """
    年々偏差（アノマリー）を計算

    Parameters
    ----------
    seasonal_avg : pd.DataFrame
        季節平均気温（行: 年、列: 観測所名）
    remove_trend : bool
        線形トレンドを除去するかどうか（デフォルト: False）

    Returns
    -------
    anomalies : pd.DataFrame
        年々偏差（単位: °C）
    climatology : pd.Series
        各地点の時間平均（気候値）

    Notes
    -----
    - 時間平均（気候値）を各地点から引く: Z_{y,j} = X_{y,j} - X̄_j
    - remove_trend=True の場合、各地点で線形トレンドを除去してから平均を引く
    """
    print(f"\n[年々偏差（アノマリー）の計算]")

    if remove_trend:
        # 線形トレンドを除去
        from scipy.signal import detrend
        anomalies = seasonal_avg.copy()
        for col in anomalies.columns:
            anomalies[col] = detrend(seasonal_avg[col].values, type='linear')
        print("  線形トレンドを除去しました")

        # 気候値（時間平均）
        climatology = seasonal_avg.mean(axis=0)
    else:
        # 各地点の時間平均（気候値）を計算
        climatology = seasonal_avg.mean(axis=0)

        # 年々偏差 = 季節平均 - 気候値
        anomalies = seasonal_avg - climatology

        print("  時間平均（気候値）を除去しました")

    # 統計情報
    print(f"  年数: {len(anomalies)}")
    print(f"  観測所数: {len(anomalies.columns)}")
    print(f"  偏差の標準偏差（各地点の平均）: {anomalies.std().mean():.3f} °C")

    return anomalies, climatology
