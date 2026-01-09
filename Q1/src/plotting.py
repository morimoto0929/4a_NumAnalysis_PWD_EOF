"""
プロット作成モジュール
"""

import matplotlib.pyplot as plt


def plot_spectrum(freq, power, intervals, nu, N, dt, window_size, output_path):
    """
    パワースペクトルを信頼区間とともにlog-logプロットで図示

    Parameters
    ----------
    freq : ndarray
        周波数（Hz）
    power : ndarray
        パワースペクトル
    intervals : dict
        信頼区間（compute_confidence_intervals の戻り値）
    nu : float
        等価自由度
    N : int
        データ長
    dt : float
        サンプリング間隔
    window_size : int
        平滑化窓サイズ
    output_path : str
        出力ファイルパス
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # スペクトル曲線
    ax.loglog(freq, power, 'b-', linewidth=1.5, label='Estimated Spectrum')

    # 95%信頼区間
    if 0.95 in intervals:
        lower_95, upper_95 = intervals[0.95]
        ax.loglog(freq, lower_95, 'g--', linewidth=1, alpha=0.7, label='95% CI')
        ax.loglog(freq, upper_95, 'g--', linewidth=1, alpha=0.7)
        ax.fill_between(freq, lower_95, upper_95, color='green', alpha=0.1)

    # 99%信頼区間
    if 0.99 in intervals:
        lower_99, upper_99 = intervals[0.99]
        ax.loglog(freq, lower_99, 'r:', linewidth=1, alpha=0.7, label='99% CI')
        ax.loglog(freq, upper_99, 'r:', linewidth=1, alpha=0.7)
        ax.fill_between(freq, lower_99, upper_99, color='red', alpha=0.05)

    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Power (arb. unit)', fontsize=12)
    ax.set_title('One-Sided Power Spectrum with Confidence Intervals', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)

    # 解析条件の注記
    fs = 1 / dt
    nyquist_freq = fs / 2
    textstr = '\n'.join([
        f'Data length: N = {N}',
        f'Sampling interval: Δt = {dt:.4f} (fs = {fs:.4f} Hz)',
        f'Nyquist frequency: {nyquist_freq:.6f} Hz',
        f'Taper: Cos20',
        f'Smoothing: Moving average (m = {window_size})',
        f'Equivalent DOF: ν = {nu:.2f}'
    ])
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"スペクトル図を保存しました: {output_path}")
    plt.close()
