"""
気象庁15地点の月平均気温データの読み込みモジュール

ファイルフォーマット:
- 第1行: 観測所の地点名
- 第2行: 観測所の緯度
- 第3行: 観測所の経度
- 第4行以降: 第1列に年月、第2列目以降に月平均気温（°C）

データは1897年12月から始まる
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_temperature_data(filepath):
    """
    気象庁15地点の月平均気温データを読み込む

    Parameters
    ----------
    filepath : str or Path
        データファイルのパス（input/MonthlyMeanTemp15points.txt）

    Returns
    -------
    df : pd.DataFrame
        月平均気温データ
        Index: DatetimeIndex (年月)
        Columns: 観測所名
    metadata : dict
        メタデータ（地点名、緯度、経度）
        Keys: 'stations', 'latitudes', 'longitudes'

    Notes
    -----
    - 欠損値は NaN として保持される
    - 年月は "YYYYMM" 形式の整数で与えられていると想定
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"データファイルが見つかりません: {filepath}")

    print(f"データ読み込み: {filepath}")

    # ファイルを開いて最初の3行（メタデータ）を読み込む
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 第1行: 観測所の地点名（先頭に "unit:degC" などのラベルが付いているため除外）
    stations = lines[0].strip().split()[1:]  # 先頭ラベルを除いて地点名のみ取り出す

    # 第2行: 緯度（先頭の "latitude" ラベルを除外）
    latitudes = [float(x) for x in lines[1].strip().split()[1:]]

    # 第3行: 経度（先頭の "longitude" ラベルを除外）
    longitudes = [float(x) for x in lines[2].strip().split()[1:]]

    # メタデータをまとめる
    metadata = {
        'stations': stations,
        'latitudes': latitudes,
        'longitudes': longitudes
    }

    print(f"観測所数: {len(stations)}")
    print(f"観測所: {', '.join(stations)}")

    # 第4行以降: 年月と気温データ
    # 第1列は年月（YYYYMM形式）、第2列以降は各地点の月平均気温
    df = pd.read_csv(
        filepath,
        sep=r'\s+',            # 可変長の空白区切り
        engine='python',       # 正規表現セパレータを使うため
        skiprows=3,            # 最初の3行（メタデータ）をスキップ
        header=None,
        names=['year_month'] + stations
    )

    # 年月を年と月に分割（形式: YYYY/MM）
    year_month_str = df['year_month'].astype(str)
    dt = pd.to_datetime(year_month_str, format='%Y/%m')
    df['year'] = dt.dt.year
    df['month'] = dt.dt.month

    # DatetimeIndexを作成（日は1日とする）
    df['date'] = pd.to_datetime(dt.dt.date)

    # Indexを日付に設定
    df.set_index('date', inplace=True)

    # 不要な列を削除
    df.drop(columns=['year_month', 'year', 'month'], inplace=True)

    print(f"データ期間: {df.index[0].strftime('%Y-%m')} ~ {df.index[-1].strftime('%Y-%m')}")
    print(f"データ数: {len(df)} ヶ月")

    # 欠損値のチェック
    n_missing = df.isna().sum().sum()
    if n_missing > 0:
        print(f"警告: {n_missing} 個の欠損値を検出しました")
        # 地点ごとの欠損数を表示
        missing_per_station = df.isna().sum()
        if missing_per_station.sum() > 0:
            print("地点別欠損数:")
            for station, count in missing_per_station[missing_per_station > 0].items():
                print(f"  {station}: {count}")

    return df, metadata


def get_station_info(metadata):
    """
    観測所情報をDataFrameとして整形

    Parameters
    ----------
    metadata : dict
        load_temperature_data() が返すメタデータ

    Returns
    -------
    station_info : pd.DataFrame
        観測所情報（地点名、緯度、経度）
        Columns: ['station', 'latitude', 'longitude']
    """
    station_info = pd.DataFrame({
        'station': metadata['stations'],
        'latitude': metadata['latitudes'],
        'longitude': metadata['longitudes']
    })

    return station_info
