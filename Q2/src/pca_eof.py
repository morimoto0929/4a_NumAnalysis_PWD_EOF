"""
PCA/EOF解析モジュール

共分散行列の固有値分解により、主成分分析（PCA）/経験的直交関数（EOF）解析を実行する。
有次元化したEOFパターンと規格化されたPC時系列を計算する。
"""

import numpy as np
import pandas as pd


def perform_pca_eof(anomalies, n_modes=3, adjust_sign=True):
    """
    PCA/EOF解析を実行（共分散PCA）

    Parameters
    ----------
    anomalies : pd.DataFrame
        年々偏差（行: 年、列: 観測所名）
        単位: °C
    n_modes : int
        計算するモード数（デフォルト: 3）
    adjust_sign : bool
        EOFとPCの符号を調整するか（デフォルト: True）
        Trueの場合、各モードで主要地点が正になるように調整

    Returns
    -------
    results : dict
        解析結果を含む辞書
        - 'eigenvalues': 固有値（全モード）
        - 'eigenvectors': 固有ベクトル（全モード、列が各EOF）
        - 'EOF_normalized': 正規化EOF（無次元、長さ1）(n_stations, n_modes)
        - 'EOF_dimensional': 有次元化EOF（°C単位）(n_stations, n_modes)
        - 'PC_raw': 生の主成分時系列（°C単位）(n_years, n_modes)
        - 'PC_standardized': 規格化PC（無次元、分散1）(n_years, n_modes)
        - 'explained_variance_ratio': 寄与率（%）(n_modes,)
        - 'cumulative_variance_ratio': 累積寄与率（%）(n_modes,)
        - 'years': 年のリスト
        - 'stations': 観測所名のリスト

    Notes
    -----
    共分散PCAの手順:
    1. 共分散行列の作成: V = Z^T Z / (Ny - 1)
    2. 固有値分解: V e_k = λ_k e_k
    3. 主成分時系列: PC_k(y) = Z_y · e_k
    4. 規格化PC: PC_k^(std)(y) = PC_k(y) / √λ_k
    5. 有次元EOF: EOF_k^(dim) = e_k √λ_k

    符号調整（adjust_sign=True）:
    - 各モードで、最大絶対値を持つ地点が正になるように調整
    """
    print(f"\n[PCA/EOF解析の実行]")
    print(f"  解析モード数: {n_modes}")

    # データの形状を取得
    n_years, n_stations = anomalies.shape
    years = anomalies.index.tolist()
    stations = anomalies.columns.tolist()

    print(f"  年数 (Ny): {n_years}")
    print(f"  観測所数 (n_stations): {n_stations}")

    # NumPy配列に変換
    Z = anomalies.values  # (n_years, n_stations)

    # 1. 共分散行列の作成（n_stations × n_stations）
    # V = Z^T Z / (Ny - 1)
    V = (Z.T @ Z) / (n_years - 1)
    print(f"  共分散行列のサイズ: {V.shape}")

    # 2. 固有値分解
    # np.linalg.eigh は昇順で返すため、降順に並べ替える
    eigenvalues, eigenvectors = np.linalg.eigh(V)

    # 降順にソート
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    print(f"  固有値（上位{n_modes}モード）:")
    for i in range(min(n_modes, len(eigenvalues))):
        print(f"    Mode {i+1}: λ = {eigenvalues[i]:.4f}")

    # 上位n_modesを取り出す
    eigenvalues_top = eigenvalues[:n_modes]
    eigenvectors_top = eigenvectors[:, :n_modes]  # (n_stations, n_modes)

    # 3. 主成分時系列（PC）の計算
    # PC_k(y) = Z_y · e_k
    PC_raw = Z @ eigenvectors_top  # (n_years, n_modes)

    # 理論的には Var(PC_k) ≈ λ_k となるはず（確認用）
    pc_variance = np.var(PC_raw, axis=0, ddof=1)
    print(f"  PC分散（理論値λ_kと比較）:")
    for i in range(n_modes):
        print(f"    Mode {i+1}: Var(PC) = {pc_variance[i]:.4f}, λ = {eigenvalues_top[i]:.4f}")

    # 4. 規格化されたPC（無次元、分散1）
    # PC_k^(std)(y) = PC_k(y) / √λ_k
    sigma_pc = np.sqrt(eigenvalues_top)  # √λ_k
    PC_standardized = PC_raw / sigma_pc  # (n_years, n_modes)

    # 確認: 規格化PCの分散が1になるか
    pc_std_variance = np.var(PC_standardized, axis=0, ddof=1)
    print(f"  規格化PC分散（1になるべき）:")
    for i in range(n_modes):
        print(f"    Mode {i+1}: Var(PC_std) = {pc_std_variance[i]:.4f}")

    # 5. 有次元化したEOF（°C単位）
    # EOF_k^(dim) = e_k √λ_k
    EOF_normalized = eigenvectors_top  # (n_stations, n_modes), 正規化EOF（無次元）
    EOF_dimensional = EOF_normalized * sigma_pc  # (n_stations, n_modes), 有次元EOF（°C）

    print(f"  有次元EOF（°C単位）:")
    for i in range(n_modes):
        print(f"    Mode {i+1}: 最大値 = {EOF_dimensional[:, i].max():.3f} °C, "
              f"最小値 = {EOF_dimensional[:, i].min():.3f} °C")

    # 6. 符号の調整（オプション）
    if adjust_sign:
        print(f"\n  符号調整を実行:")
        for i in range(n_modes):
            # 最大絶対値を持つ地点のインデックス
            max_abs_idx = np.argmax(np.abs(EOF_dimensional[:, i]))
            # その地点の値が負なら、EOFとPCの符号を反転
            if EOF_dimensional[max_abs_idx, i] < 0:
                EOF_normalized[:, i] *= -1
                EOF_dimensional[:, i] *= -1
                PC_raw[:, i] *= -1
                PC_standardized[:, i] *= -1
                print(f"    Mode {i+1}: 符号を反転（主要地点: {stations[max_abs_idx]}）")

    # 7. 寄与率（explained variance ratio）
    explained_variance = eigenvalues_top
    total_variance = eigenvalues.sum()
    explained_variance_ratio = (explained_variance / total_variance) * 100  # %
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    print(f"\n  寄与率:")
    for i in range(n_modes):
        print(f"    Mode {i+1}: {explained_variance_ratio[i]:.2f}% "
              f"(累積: {cumulative_variance_ratio[i]:.2f}%)")

    # 結果を辞書にまとめる
    results = {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'EOF_normalized': EOF_normalized,
        'EOF_dimensional': EOF_dimensional,
        'PC_raw': PC_raw,
        'PC_standardized': PC_standardized,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance_ratio': cumulative_variance_ratio,
        'years': years,
        'stations': stations,
        'n_modes': n_modes,
        'n_years': n_years,
        'n_stations': n_stations
    }

    print(f"\n  PCA/EOF解析が完了しました。")

    return results


def get_eof_dataframe(results, dimensional=True):
    """
    EOF（因子負荷量）をDataFrameとして整形

    Parameters
    ----------
    results : dict
        perform_pca_eof() の返り値
    dimensional : bool
        有次元化EOF（°C単位）を使うか（デフォルト: True）
        False の場合は正規化EOF（無次元）

    Returns
    -------
    eof_df : pd.DataFrame
        EOF値（行: 観測所、列: Mode1, Mode2, Mode3, ...）
    """
    if dimensional:
        eof_values = results['EOF_dimensional']
        unit = '(°C)'
    else:
        eof_values = results['EOF_normalized']
        unit = '(nondim.)'

    eof_df = pd.DataFrame(
        eof_values,
        index=results['stations'],
        columns=[f'Mode{i+1}' for i in range(results['n_modes'])]
    )

    eof_df.index.name = 'Station'

    return eof_df


def get_pc_dataframe(results, standardized=True):
    """
    PC時系列をDataFrameとして整形

    Parameters
    ----------
    results : dict
        perform_pca_eof() の返り値
    standardized : bool
        規格化PC（無次元）を使うか（デフォルト: True）
        False の場合は生のPC（°C単位）

    Returns
    -------
    pc_df : pd.DataFrame
        PC時系列（行: 年、列: Mode1, Mode2, Mode3, ...）
    """
    if standardized:
        pc_values = results['PC_standardized']
        unit = '(standardized)'
    else:
        pc_values = results['PC_raw']
        unit = '(°C)'

    pc_df = pd.DataFrame(
        pc_values,
        index=results['years'],
        columns=[f'Mode{i+1}' for i in range(results['n_modes'])]
    )

    pc_df.index.name = 'Year'

    return pc_df


def save_pca_results(results, output_dir, prefix='pca_eof'):
    """
    PCA/EOF解析の結果をCSVファイルとして保存

    Parameters
    ----------
    results : dict
        perform_pca_eof() の返り値
    output_dir : str or Path
        出力ディレクトリ
    prefix : str
        ファイル名のプレフィックス（デフォルト: 'pca_eof'）
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # EOF（有次元）を保存
    eof_df = get_eof_dataframe(results, dimensional=True)
    eof_path = output_dir / f'{prefix}_EOF_dimensional.csv'
    eof_df.to_csv(eof_path, float_format='%.4f')
    print(f"  有次元EOFを保存: {eof_path}")

    # PC（規格化）を保存
    pc_df = get_pc_dataframe(results, standardized=True)
    pc_path = output_dir / f'{prefix}_PC_standardized.csv'
    pc_df.to_csv(pc_path, float_format='%.4f')
    print(f"  規格化PCを保存: {pc_path}")

    # 寄与率を保存
    variance_df = pd.DataFrame({
        'Mode': [f'Mode{i+1}' for i in range(results['n_modes'])],
        'Eigenvalue': results['eigenvalues'][:results['n_modes']],
        'Explained_Variance_Ratio (%)': results['explained_variance_ratio'],
        'Cumulative_Variance_Ratio (%)': results['cumulative_variance_ratio']
    })
    variance_path = output_dir / f'{prefix}_variance.csv'
    variance_df.to_csv(variance_path, index=False, float_format='%.4f')
    print(f"  寄与率を保存: {variance_path}")


def compute_north_error_bars(results, se=None):
    """
    North's rule of thumb による寄与率の不確かさを計算

    Parameters
    ----------
    results : dict
        perform_pca_eof() の返り値
    se : float or None
        有効自由度（effective degrees of freedom）
        None の場合は se = n_years（標本数）を使用
        講義資料の簡便法：しばしば se = s を採用する

    Returns
    -------
    error_dict : dict
        エラーバー情報を含む辞書
        - 'eigenvalue_errors': 固有値の誤差 Δλ_i（全モード）
        - 'variance_ratio_errors': 寄与率の誤差 Δf_i（全モード）
        - 'eigenvalue_errors_pct': 固有値の誤差（%、上位n_modesのみ）
        - 'variance_ratio_errors_pct': 寄与率の誤差（%、上位n_modesのみ）
        - 'se': 使用した有効自由度

    Notes
    -----
    North's rule of thumb:
    - 固有値の誤差: Δλ_i = λ_i √(2/s_e)
    - 寄与率の誤差: Δf_i ≈ Δλ_i / Σλ_j （分母固定の近似）

    隣接モードのエラーバーが重ならなければ、両者は有意に分離している。
    重なる場合は縮退的で、線形結合が同程度にあり得る。

    References
    ----------
    - instruction_Q2.md の追加部分
    - 講義資料（PCA/EOF の有効自由度と North's rule）
    """
    # 有効自由度の設定
    if se is None:
        se = results['n_years']  # デフォルト: se = s （標本数）

    print(f"\n[North's rule of thumb による誤差推定]")
    print(f"  有効自由度: s_e = {se}")

    # 固有値（全モード）
    eigenvalues = results['eigenvalues']

    # 固有値の誤差: Δλ_i = λ_i √(2/s_e)
    eigenvalue_errors = eigenvalues * np.sqrt(2.0 / se)

    # 寄与率: f_i = λ_i / Σλ_j
    total_variance = eigenvalues.sum()
    variance_ratios = eigenvalues / total_variance

    # 寄与率の誤差: Δf_i ≈ Δλ_i / Σλ_j
    variance_ratio_errors = eigenvalue_errors / total_variance

    # パーセント表示用
    n_modes = results['n_modes']
    eigenvalue_errors_pct = eigenvalue_errors[:n_modes]
    variance_ratio_errors_pct = variance_ratio_errors[:n_modes] * 100  # %

    print(f"  固有値の誤差（上位{n_modes}モード）:")
    for i in range(n_modes):
        print(f"    Mode {i+1}: Δλ = {eigenvalue_errors[i]:.4f} "
              f"(λ = {eigenvalues[i]:.4f})")

    print(f"  寄与率の誤差（上位{n_modes}モード）:")
    for i in range(n_modes):
        print(f"    Mode {i+1}: Δf = {variance_ratio_errors_pct[i]:.2f}% "
              f"(f = {results['explained_variance_ratio'][i]:.2f}%)")

    # エラーバー情報を辞書にまとめる
    error_dict = {
        'eigenvalue_errors': eigenvalue_errors,
        'variance_ratio_errors': variance_ratio_errors,
        'eigenvalue_errors_pct': eigenvalue_errors_pct,
        'variance_ratio_errors_pct': variance_ratio_errors_pct,
        'se': se
    }

    return error_dict
