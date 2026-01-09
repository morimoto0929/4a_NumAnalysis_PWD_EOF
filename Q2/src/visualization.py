"""
可視化モジュール

PCA/EOF解析の結果を可視化する。
- 規格化PC時系列のグラフ
- EOFパターンの数値表
- EOFパターンの地図表示
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_pc_timeseries(results, output_dir, filename='PC_timeseries.png',
                       separate_plots=False, figsize=(12, 8), dpi=300):
    """
    規格化PC時系列をプロット

    Parameters
    ----------
    results : dict
        perform_pca_eof() の返り値
    output_dir : str or Path
        出力ディレクトリ
    filename : str
        出力ファイル名
    separate_plots : bool
        各モードを別々の図にするか（False: 1つの図に3曲線）
    figsize : tuple
        図のサイズ
    dpi : int
        解像度

    Returns
    -------
    output_path : Path
        保存されたファイルパス
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    years = results['years']
    PC_std = results['PC_standardized']
    n_modes = results['n_modes']
    explained_variance_ratio = results['explained_variance_ratio']

    if separate_plots:
        # 各モードを別々の図にプロット
        fig, axes = plt.subplots(n_modes, 1, figsize=figsize, sharex=True)
        if n_modes == 1:
            axes = [axes]

        for i in range(n_modes):
            ax = axes[i]
            ax.plot(years, PC_std[:, i], 'b-', linewidth=1.5)
            ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_ylabel(f'PC{i+1} (std)', fontsize=11)
            ax.set_title(f'Mode {i+1} (Explained Variance: {explained_variance_ratio[i]:.2f}%)',
                         fontsize=12, fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.5)

        axes[-1].set_xlabel('Year', fontsize=11)
        plt.tight_layout()

    else:
        # 1つの図に3曲線をプロット
        fig, ax = plt.subplots(figsize=figsize)

        colors = ['b', 'r', 'g']
        for i in range(n_modes):
            label = f'Mode {i+1} ({explained_variance_ratio[i]:.2f}%)'
            ax.plot(years, PC_std[:, i], color=colors[i % len(colors)],
                    linewidth=1.5, label=label, alpha=0.8)

        ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Standardized PC (nondim.)', fontsize=12)
        ax.set_title('Standardized Principal Component Time Series', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()

    output_path = output_dir / filename
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"  PC時系列を保存: {output_path}")
    plt.close()

    return output_path


def plot_eof_pattern(results, metadata, output_dir, filename='EOF_pattern_table.png',
                     figsize=(10, 6), dpi=300):
    """
    EOFパターンを数値表として可視化（棒グラフ付き）

    Parameters
    ----------
    results : dict
        perform_pca_eof() の返り値
    metadata : dict
        観測所のメタデータ（地点名、緯度、経度）
    output_dir : str or Path
        出力ディレクトリ
    filename : str
        出力ファイル名
    figsize : tuple
        図のサイズ
    dpi : int
        解像度

    Returns
    -------
    output_path : Path
        保存されたファイルパス
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stations = results['stations']
    EOF_dim = results['EOF_dimensional']
    n_modes = results['n_modes']
    explained_variance_ratio = results['explained_variance_ratio']

    # 各モードの棒グラフをプロット
    fig, axes = plt.subplots(1, n_modes, figsize=figsize, sharey=True)
    if n_modes == 1:
        axes = [axes]

    for i in range(n_modes):
        ax = axes[i]
        eof_values = EOF_dim[:, i]

        # 棒グラフ
        colors = ['red' if val >= 0 else 'blue' for val in eof_values]
        bars = ax.barh(stations, eof_values, color=colors, alpha=0.7, edgecolor='black')

        ax.axvline(0, color='k', linestyle='-', linewidth=1)
        ax.set_xlabel('EOF (°C)', fontsize=10)
        ax.set_title(f'Mode {i+1}\n({explained_variance_ratio[i]:.2f}%)',
                     fontsize=11, fontweight='bold')
        ax.grid(True, axis='x', linestyle=':', alpha=0.5)

    axes[0].set_ylabel('Station', fontsize=10)
    plt.tight_layout()

    output_path = output_dir / filename
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"  EOFパターン（棒グラフ）を保存: {output_path}")
    plt.close()

    return output_path


def plot_eof_map(results, metadata, output_dir, filename_prefix='EOF_map',
                 figsize=(15, 5), dpi=300, vmin=None, vmax=None):
    """
    EOFパターンを地図上に表示

    Parameters
    ----------
    results : dict
        perform_pca_eof() の返り値
    metadata : dict
        観測所のメタデータ（地点名、緯度、経度）
    output_dir : str or Path
        出力ディレクトリ
    filename_prefix : str
        出力ファイル名のプレフィックス
    figsize : tuple
        図のサイズ
    dpi : int
        解像度
    vmin, vmax : float
        カラーバーの範囲（Noneの場合は自動）

    Returns
    -------
    output_paths : list of Path
        保存されたファイルパスのリスト
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stations = results['stations']
    EOF_dim = results['EOF_dimensional']
    n_modes = results['n_modes']
    explained_variance_ratio = results['explained_variance_ratio']

    lats = metadata['latitudes']
    lons = metadata['longitudes']

    output_paths = []

    # 全モードを1つの図にプロット
    fig, axes = plt.subplots(1, n_modes, figsize=figsize,
                             subplot_kw={'projection': None})
    if n_modes == 1:
        axes = [axes]

    # カラーマップの範囲を全モードで統一
    if vmin is None or vmax is None:
        abs_max = np.abs(EOF_dim).max()
        vmin = -abs_max
        vmax = abs_max

    for i in range(n_modes):
        ax = axes[i]
        eof_values = EOF_dim[:, i]

        # 散布図（サイズと色でEOF値を表現）
        scatter = ax.scatter(lons, lats, c=eof_values, s=200,
                             cmap='RdBu_r', vmin=vmin, vmax=vmax,
                             edgecolor='black', linewidth=1.5, alpha=0.8)

        # 地点名をラベル表示
        for j, station in enumerate(stations):
            ax.text(lons[j], lats[j], station, fontsize=7,
                    ha='center', va='bottom')

        # カラーバー
        cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal',
                            pad=0.05, shrink=0.8)
        cbar.set_label('EOF (°C)', fontsize=9)

        ax.set_xlabel('Longitude (°E)', fontsize=10)
        ax.set_ylabel('Latitude (°N)', fontsize=10)
        ax.set_title(f'Mode {i+1} EOF Pattern\n({explained_variance_ratio[i]:.2f}%)',
                     fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    output_path = output_dir / f'{filename_prefix}_all_modes.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"  EOF地図（全モード）を保存: {output_path}")
    plt.close()

    output_paths.append(output_path)

    return output_paths


def print_eof_table(results, metadata):
    """
    EOFパターンを数値表として標準出力に表示

    Parameters
    ----------
    results : dict
        perform_pca_eof() の返り値
    metadata : dict
        観測所のメタデータ（地点名、緯度、経度）
    """
    from .pca_eof import get_eof_dataframe

    print("\n" + "=" * 60)
    print("有次元EOF（因子負荷量）: 単位 °C")
    print("=" * 60)

    eof_df = get_eof_dataframe(results, dimensional=True)
    print(eof_df.to_string(float_format=lambda x: f'{x:7.3f}'))

    print("\n" + "=" * 60)


def print_variance_table(results):
    """
    寄与率を表として標準出力に表示

    Parameters
    ----------
    results : dict
        perform_pca_eof() の返り値
    """
    print("\n" + "=" * 60)
    print("寄与率（Explained Variance Ratio）")
    print("=" * 60)

    for i in range(results['n_modes']):
        print(f"Mode {i+1}: {results['explained_variance_ratio'][i]:6.2f}% "
              f"(累積: {results['cumulative_variance_ratio'][i]:6.2f}%)")

    print("=" * 60)
