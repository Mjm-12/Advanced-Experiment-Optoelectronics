
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import tkinter as tk
from tkinter import filedialog
import math
import os
import csv

# ===== User-configurable parameters (ユーザ設定パラメータ) =====

# 波形描画モードの選択:
# "original"：生の実測データをそのまま描画
# "averaged"  ：基本周波数に基づいて各周期を抽出・平均化した波形を描画
WAVEFORM_PLOT_MODE = "original"       # "original" または "averaged"
num_cycles_user = 20

# 波形グラフのx軸表示設定（manualまたはautoを選択）
WAVEFORM_AXIS_MODE = "auto"         # "auto" または "manual"
MANUAL_WAVEFORM_XLIM = (-10, 10)        # Manualモード時の波形グラフのx軸範囲 [ms]

# FFTグラフの表示設定
FFT_AXIS_MODE = "auto"                # FFTのx軸表示を"auto"（自動設定）または"manual"（手動設定）にする
MANUAL_FFT_XLIM = (100, 50000)         # Manualモード時のFFTのx軸表示範囲 [Hz]
MANUAL_FFT_YLIM = (-100, 5)            # Manualモード時のFFTのy軸表示範囲 [dB]

# FFT解析における高調波表示の最大次数（2次からこの値まで表示）
MAX_HARMONIC_ORDER = 10

# FFT結果をCSV出力するか否かのフラグ
Export_FFT_Result = False

# FFT解析時に使用する窓関数の選択
# 選択可能："hann", "hamming", "blackman", "bartlett", "rectangular"（または"none"）
FFT_WINDOW = "rectangular"

# FFT解析前に適用するフィルタ設定（Hz）
min_freq_temp = 10     # ハイパスフィルタ（下限）
max_freq_temp = 99000  # ローパスフィルタ（上限）
# =============================================================

def format_frequency(freq):
    """
    周波数値をフォーマットして文字列として返す関数
    - 1000 Hz未満の場合： xxx.xx Hz の形式で表示
    - 1000 Hz以上の場合： x.xxxx kHz の形式で表示
    """
    if freq < 1000:
        return f"{freq:.2f} Hz"
    else:
        return f"{freq/1000:.4f} kHz"

def fft_tick_formatter(x, pos):
    """
    FFTグラフのx軸目盛りラベルを整形するためのフォーマッタ関数
    - 例: 100, 500, 1k, 5k, 10k, … のように表示する
    """
    if x < 1000:
        return f"{int(x)}"
    else:
        if abs(x % 1000) < 1e-6:
            return f"{int(x//1000)}k"
        else:
            return f"{x/1000:.0f}k"

def generate_fft_ticks(lower, upper):
    """
    指定された下限(lower)～上限(upper)の範囲内で、
    1×10^n, 5×10^n の目盛り値をリストとして返す関数
    ※ 最小値と最大値も必ず含むようにする
    """
    ticks = []
    n_low = int(np.floor(np.log10(lower))) if lower > 0 else 0
    n_high = int(np.ceil(np.log10(upper)))
    for n in range(n_low, n_high+1):
        for base in [1, 5]:
            val = base * (10**n)
            if lower <= val <= upper:
                ticks.append(val)
    if lower not in ticks:
        ticks.append(lower)
    if upper not in ticks:
        ticks.append(upper)
    ticks = sorted(ticks)
    return ticks

def ordinal(n):
    """
    整数 n に対して、正しい英語の序数接尾辞 (ordinal suffix) を返す関数
    例: 1 -> "st", 2 -> "nd", 3 -> "rd", 4 -> "th", 11 -> "th", etc.
    """
    if 11 <= n % 100 <= 13:
        return "th"
    else:
        if n % 10 == 1:
            return "st"
        elif n % 10 == 2:
            return "nd"
        elif n % 10 == 3:
            return "rd"
        else:
            return "th"

# =============================================================================
# クラス WaveformPlotter
# － 入力データ（CSVから読み込んだ時間・電圧データ）を用いて
#    波形描画とFFT解析を行い、結果をグラフにプロットする
# =============================================================================
class WaveformPlotter:
    def __init__(self, original_data, rates, original_time_axes, file_names, file_path):
        """
        コンストラクタ
        :param original_data: 電圧データ（各要素は1次元のnp.array。生の実測値）
        :param rates: サンプリング周波数（Hz）のリスト
        :param original_time_axes: 時間軸データ（秒→msに換算済み）のリスト
        :param file_names: 対象ファイル名のリスト
        :param file_path: 選択されたCSVファイルのフルパス
        """
        self.original_data = original_data
        self.rates = rates
        self.original_time_axes = original_time_axes
        self.file_names = file_names
        self.file_path = file_path

    def plot(self, mode='original'):
        """
        グラフウィンドウを作成し、上段に波形、下段にFFT解析結果をプロットする
        :param mode: 波形描画モード。 'original' なら生データ、 'averaged' なら平均化波形を描画
        """
        fig, axs = plt.subplots(2, 1, figsize=(15, 8))
        if mode.lower() == 'averaged':
            self._plot_averaged_waveform(axs[0])
        else:
            self._plot_waveform(axs[0])
        self._plot_fft(axs[1])
        plt.tight_layout()
        plt.show()

    def _print_waveform_statistics(self, data):
        """
        生データから波形統計情報を計算し、コンソールに出力する
        (最大値、最小値、ピーク・トゥー・ピーク、DC成分)
        """
        V_Max = np.max(data)
        V_Min = np.min(data)
        V_pp = V_Max - V_Min
        V_DC = np.mean(data)
        print("--------------------------------------------------")
        print(f"File: {self.file_names[0]}")
        print("--------------------------------------------------")
        print("Waveform Statistics:")
        print(f"  V_Max = {V_Max:.4f} V")
        print(f"  V_Min = {V_Min:.4f} V")
        print(f"  V_p-p = {V_pp:.4f} V")
        print(f"  V_DC  = {V_DC:.4f} V")
        print("--------------------------------------------------")

    def _get_fundamental_frequency(self, data, rate):
        """
        FFTを用いて基本周波数 f0 を取得するヘルパー関数
        :param data: 電圧データ（1次元np.array）
        :param rate: サンプリング周波数（Hz）
        :return: 基本周波数 f0（Hz）。有効な成分がない場合は0を返す。
        """
        N = len(data)
        # FFT解析に使用する窓関数の選択
        if FFT_WINDOW.lower() in ["hann", "hanning"]:
            window = np.hanning(N)
        elif FFT_WINDOW.lower() in ["hamming"]:
            window = np.hamming(N)
        elif FFT_WINDOW.lower() in ["blackman"]:
            window = np.blackman(N)
        elif FFT_WINDOW.lower() in ["bartlett"]:
            window = np.bartlett(N)
        elif FFT_WINDOW.lower() in ["rectangular", "none"]:
            window = np.ones(N)
        else:
            print(f"Unknown FFT window '{FFT_WINDOW}'. Using rectangular window.")
            window = np.ones(N)

        # 窓関数を適用してFFT実行
        windowed_data = data * window
        fft_result = np.fft.fft(windowed_data)
        dt = 1 / rate
        # 正の周波数部分のみを取得
        freqs = np.fft.fftfreq(N, d=dt)[:N//2]
        fft_magnitude = np.abs(fft_result)[:N//2]

        # 指定したフィルタ範囲内の成分のみを抽出
        valid_indices = np.logical_and(freqs >= min_freq_temp, freqs <= max_freq_temp)
        if np.any(valid_indices):
            valid_idx = np.where(valid_indices)[0]
            # 最大の振幅を持つ周波数を基本周波数とする
            fundamental_index = valid_idx[np.argmax(fft_magnitude[valid_idx])]
            f0 = freqs[fundamental_index]
            return f0
        else:
            return 0

    def _plot_waveform(self, ax):
        """
        生データ（original waveform）の波形グラフを描画する
        """
        data = self.original_data[0]
        time_ms = self.original_time_axes[0]
        title = "Waveform Data"

        # 波形統計情報を出力（共通処理）
        self._print_waveform_statistics(data)
        
        # y軸の範囲設定用：最大絶対値から丸めた値を用いる
        V_Max = np.max(data)
        V_Min = np.min(data)
        limit = max(abs(V_Max), abs(V_Min))
        limit_rounded = math.ceil(limit * 10) / 10.0
        yticks = [-limit_rounded, -limit_rounded/2, 0, limit_rounded/2, limit_rounded]
        
        # グラフの基本タイトルと軸ラベルの設定
        ax.set_title(title)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage')
        
        # x軸の表示範囲設定
        if WAVEFORM_AXIS_MODE.lower() == "manual":
            # ユーザ指定の場合
            lower_bound, upper_bound = MANUAL_WAVEFORM_XLIM
            ticks = np.linspace(lower_bound, upper_bound, 11)
        else:
            # 自動設定の場合：FFTから基本周波数 f0 を取得し、それに応じた範囲を決定
            rate = self.rates[0]
            N = len(data)
            if FFT_WINDOW.lower() in ["hann", "hanning"]:
                window = np.hanning(N)
            elif FFT_WINDOW.lower() in ["hamming"]:
                window = np.hamming(N)
            elif FFT_WINDOW.lower() in ["blackman"]:
                window = np.blackman(N)
            elif FFT_WINDOW.lower() in ["bartlett"]:
                window = np.bartlett(N)
            elif FFT_WINDOW.lower() in ["rectangular", "none"]:
                window = np.ones(N)
            else:
                print(f"Unknown FFT window '{FFT_WINDOW}'. Using rectangular window.")
                window = np.ones(N)
            windowed_data = data * window
            fft_result = np.fft.fft(windowed_data)
            dt = 1 / rate
            freqs = np.fft.fftfreq(N, d=dt)[:N//2]
            fft_magnitude = np.abs(fft_result)[:N//2]

            valid_indices = np.logical_and(freqs >= min_freq_temp, freqs <= max_freq_temp)
            if np.any(valid_indices):
                valid_idx = np.where(valid_indices)[0]
                fundamental_index = valid_idx[np.argmax(fft_magnitude[valid_idx])]
                f0 = freqs[fundamental_index]
            else:
                f0 = 0

            if f0 > 0:
                num_digits = len(str(int(f0)))
                exponent = 4 - num_digits
                plot_range = 5 * (10 ** exponent)  # 単位：ms
                lower_bound = -plot_range
                upper_bound = plot_range
                ticks = np.linspace(lower_bound, upper_bound, 11)
                print(f"Waveform Auto-mode: f0 = {f0:.2f} Hz, x-axis range set to [{lower_bound:.2f}, {upper_bound:.2f}] ms")
            else:
                lower_bound, upper_bound = -5, 5
                ticks = np.linspace(lower_bound, upper_bound, 11)
                print("Waveform Auto-mode: Fundamental frequency not detected, using default x-axis [-5, 5] ms")
        
        ax.set_xlim(lower_bound, upper_bound)
        ax.set_xticks(ticks)
        x_minor = (upper_bound - lower_bound) / 100.0
        ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
        
        # y軸の目盛り設定（ピーク・トゥー・ピーク値から動的に設定）
        V_pp = V_Max - V_Min
        if V_pp > 0:
            order = 10 ** (math.floor(math.log10(V_pp)))
            one_sig = round(V_pp / order) * order
            y_minor = one_sig / 50.0
        else:
            y_minor = 0.01
        ax.yaxis.set_minor_locator(MultipleLocator(y_minor))
        ax.set_yticks(yticks)
        
        ax.grid(True, which='major', linestyle='-')
        ax.grid(True, which='minor', linestyle=':')
        
        # 生データの波形をプロット
        ax.plot(time_ms, data)

    def _plot_averaged_waveform(self, ax):
        """
        基本周波数に基づいて各周期を抽出し、
        利用可能な全周期分を平均化して、1周期の平均波形を算出し、
        その平均波形を10周期分連結して描画する
        ※ 波形統計情報も共通処理で出力する
        """
        data = self.original_data[0]
        time_ms = self.original_time_axes[0]
        rate = self.rates[0]
        
        # 生データの統計情報を出力
        self._print_waveform_statistics(data)
        
        V_DC = np.mean(data)  # 平均値をDC成分とする
        
        # FFTから基本周波数 f0 を取得
        f0 = self._get_fundamental_frequency(data, rate)
        if f0 <= 0:
            print("基本周波数が検出できなかったため、平均化波形の描画を中止します。")
            ax.text(0.5, 0.5, "Fundamental frequency not detected", 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            return

        T_sec = 1 / f0         # 1周期の時間（秒）
        T_ms = T_sec * 1000    # 1周期の時間（ミリ秒）
        N_cycle = int(round(rate / f0))  # 1周期あたりのサンプル数

        print(f"Basic frequency for averaging: {f0:.2f} Hz  →  Period = {T_ms:.2f} ms")
        print(f"Samples per cycle: {N_cycle}")
        
        # DC値付近の上昇零交差を基準に、周期の開始位置（位相合わせ）を検出する
        start_idx = 0
        for i in range(len(data) - 1):
            if data[i] < V_DC and data[i+1] >= V_DC:
                start_idx = i
                break
        print(f"Cycle segmentation starting at sample index: {start_idx}")
        
        # 利用可能な完全な周期数を計算
        num_full_cycles = (len(data) - start_idx) // N_cycle
        num_cycles_to_average = min(num_full_cycles, num_cycles_user)
        if num_cycles_to_average < 1:
            print("十分な周期数が得られなかったため、平均化波形を描画できません。")
            ax.text(0.5, 0.5, "Not enough cycles for averaging", 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            return
        
        # 各周期のデータを抽出してリストに格納
        cycles = []
        for i in range(num_cycles_to_average):
            start = start_idx + i * N_cycle
            end = start + N_cycle
            if end <= len(data):
                cycles.append(data[start:end])
        cycles = np.array(cycles)
        
        # 各サンプル位置ごとに平均値を算出し、1周期の平均波形とする
        averaged_cycle = np.mean(cycles, axis=0)
        
        # 平均波形（1周期）を10周期分連結して描画用データを作成
        rep_cycles = 10
        y_plot = np.tile(averaged_cycle, rep_cycles)
        N_total = rep_cycles * N_cycle
        # x軸は0から10周期分の時間（ミリ秒）とする
        t_plot = np.linspace(0, rep_cycles * T_ms, N_total, endpoint=False)
        
        ax.set_title(f"Averaged Waveform ({num_cycles_to_average} cycles)")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Voltage")
        ax.grid(True, which='major', linestyle='-')
        ax.grid(True, which='minor', linestyle=':')
        
        # 平均化した波形をプロット（線の太さをやや太めに設定）
        ax.plot(t_plot, y_plot, color='C1', linewidth=2)
        ax.set_xlim(0, rep_cycles * T_ms)
        x_minor = (rep_cycles * T_ms) / 100.0
        ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
        
        print("--------------------------------------------------")

    def _plot_fft(self, ax):
        """
        FFT解析を実行し、FFT結果をdB表示でプロットする
        """
        ax.set_title("FFT Result")
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_xscale('log')  # x軸は対数スケールで表示
        ax.grid(True, which='both', linestyle='-')
        
        noise_floor_db_list = []  # ノイズフロアのdB値（後でTHD計算に利用）
        fundamental_bounds = []   # 基本周波数に基づくFFT表示範囲の候補
        
        data = self.original_data[0]
        rate = self.rates[0]
        N = len(data)
        resolution = rate / N  # FFT分解能
        print(f"Sampling Frequency: {format_frequency(rate)}")
        print(f"FFT Resolution: {format_frequency(resolution)}")
        
        # FFT解析に使用する窓関数の選択
        if FFT_WINDOW.lower() in ["hann", "hanning"]:
            window = np.hanning(N)
        elif FFT_WINDOW.lower() in ["hamming"]:
            window = np.hamming(N)
        elif FFT_WINDOW.lower() in ["blackman"]:
            window = np.blackman(N)
        elif FFT_WINDOW.lower() in ["bartlett"]:
            window = np.bartlett(N)
        elif FFT_WINDOW.lower() in ["rectangular", "none"]:
            window = np.ones(N)
        else:
            print(f"Unknown FFT window '{FFT_WINDOW}'. Using rectangular window.")
            window = np.ones(N)
        windowed_data = data * window
        
        # FFTの実行
        fft_result = np.fft.fft(windowed_data)
        dt = 1 / rate
        freqs = np.fft.fftfreq(N, d=dt)[:N//2]
        fft_magnitude = np.abs(fft_result)[:N//2]
        
        # 指定したフィルタ範囲内の成分のみを対象とする
        valid_indices = np.logical_and(freqs >= min_freq_temp, freqs <= max_freq_temp)
        if np.any(valid_indices):
            valid_idx = np.where(valid_indices)[0]
            fundamental_index = valid_idx[np.argmax(fft_magnitude[valid_idx])]
            A0 = fft_magnitude[fundamental_index]  # 基本成分の振幅
            f0 = freqs[fundamental_index]            # 基本周波数
        else:
            A0 = 1
            f0 = 0
        
        if np.any(valid_indices):
            max_in_range = np.max(fft_magnitude[valid_indices])
        else:
            max_in_range = 1  
        fft_magnitude_db = 20 * np.log10(fft_magnitude / max_in_range)
        ax.plot(freqs, fft_magnitude_db, color='C0')
        
        if f0 <= 0 or A0 == 0:
            print("Fundamental frequency not detected.")
        else:
            print(f"Fundamental frequency: {format_frequency(f0)}")
            for h in range(2, MAX_HARMONIC_ORDER+1):
                target_freq = h * f0
                idx = np.argmin(np.abs(freqs - target_freq))
                A_h = fft_magnitude[idx]
                ratio = (A_h / A0) * 100
                h_str = f"{h}{ordinal(h)}".rjust(6)
                print(f"  {h_str} harmonic: {ratio:7.4f}% (target: {format_frequency(target_freq)}, actual: {format_frequency(freqs[idx])})")
            
            exclude_indices = np.array([], dtype=int)
            for h in range(1, MAX_HARMONIC_ORDER+1):
                target_freq = h * f0
                idx = np.argmin(np.abs(freqs - target_freq))
                window_idx = np.arange(max(idx-2, 0), min(idx+3, len(freqs)))
                exclude_indices = np.concatenate((exclude_indices, window_idx))
            exclude_indices = np.unique(exclude_indices)
            noise_indices = np.setdiff1d(np.where(valid_indices)[0], exclude_indices)
            if noise_indices.size > 0:
                noise_floor_amp = np.median(fft_magnitude[noise_indices])
            else:
                noise_floor_amp = 0
            print(f"Noise Floor: {(noise_floor_amp/A0)*100:7.4f}%  ({20 * np.log10(noise_floor_amp/A0):7.2f} dB)")
            noise_floor_db_list.append(20 * np.log10(noise_floor_amp/A0))
            
            sum_squares_user = 0
            for h in range(2, MAX_HARMONIC_ORDER+1):
                target_freq = h * f0
                idx = np.argmin(np.abs(freqs - target_freq))
                A_h = fft_magnitude[idx]
                effective_A_h = max(A_h - noise_floor_amp, 0)
                sum_squares_user += effective_A_h**2
            THD_user = (np.sqrt(sum_squares_user) / A0) * 100
            print(f"THD (2nd-{MAX_HARMONIC_ORDER}th harmonics, noise subtracted): {THD_user:7.4f}%")
            
            H_max = int(np.floor((rate/2) / f0))
            sum_squares_all = 0
            for h in range(2, H_max+1):
                target_freq = h * f0
                idx = np.argmin(np.abs(freqs - target_freq))
                A_h = fft_magnitude[idx]
                effective_A_h = max(A_h - noise_floor_amp, 0)
                sum_squares_all += effective_A_h**2
            THD_all = (np.sqrt(sum_squares_all) / A0) * 100
            print(f"THD (all harmonics up to Nyquist, noise subtracted): {THD_all:7.4f}%")
            
            if f0 > 0:
                d = int(np.floor(np.log10(f0))) + 1
                factor = 10 ** (d - 1)
                rounded_f0 = round(f0 / factor) * factor
                lower_bound_candidate = rounded_f0 / 2
                upper_bound_candidate = rounded_f0 * 40
                upper_bound_candidate = min(upper_bound_candidate, rate/2)  # Nyquist制限
                fundamental_bounds.append((lower_bound_candidate, upper_bound_candidate))
            else:
                fundamental_bounds.append((min_freq_temp, max_freq_temp))
        print("--------------------------------------------------")
        
        if FFT_AXIS_MODE.lower() == "manual":
            user_lower, user_upper = MANUAL_FFT_XLIM
            if user_lower < resolution:
                print(f"User-specified lower frequency {user_lower} Hz is below FFT resolution {resolution:.2f} Hz; using FFT resolution instead.")
                lower_bound_fft = resolution
            else:
                lower_bound_fft = user_lower
            if user_upper > rate/2:
                print(f"User-specified upper frequency {user_upper} Hz is above Nyquist frequency {rate/2:.2f} Hz; using Nyquist frequency instead.")
                upper_bound_fft = rate/2
            else:
                upper_bound_fft = user_upper
            y_lower, y_upper = MANUAL_FFT_YLIM
        else:
            lower_bound_fft, upper_bound_fft = fundamental_bounds[0]
            if noise_floor_db_list:
                global_noise_db = min(noise_floor_db_list)
                y_lower = 10 * np.floor((global_noise_db - 10) / 10)
            else:
                y_lower = -120
            y_upper = 5
        
        xticks = generate_fft_ticks(lower_bound_fft, upper_bound_fft)
        ax.set_xlim(lower_bound_fft, upper_bound_fft)
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(FuncFormatter(fft_tick_formatter))
        ax.set_ylim(y_lower, y_upper)
        
        # FFT結果のエクスポート
        if Export_FFT_Result:
            export_mask = (freqs >= resolution) & (freqs <= rate/2)
            export_freqs = freqs[export_mask]
            export_magnitude_db = fft_magnitude_db[export_mask]
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            export_file_name = base_name + "_FFT.csv"
            export_file_path = os.path.join(os.path.dirname(self.file_path), export_file_name)
            with open(export_file_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([f"FFT({FFT_WINDOW})", f"{os.path.basename(self.file_path)}"])
                writer.writerow(["Frequency[Hz]", "Magnitude[dB]"])
                for f_val, m_val in zip(export_freqs, export_magnitude_db):
                    writer.writerow([f"{f_val:.4f}", f"{m_val:.4f}"])
            print(f"FFT result exported to: {export_file_path}")

# =============================================================================
# main関数：ファイル選択ダイアログからCSVファイルを読み込み、波形・FFTグラフを描画
# =============================================================================
def main():
    # Tkinterを利用してファイル選択ダイアログを表示
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV files", "*.csv")]
    )
    if not file_path:
        print("No file selected.")
        return

    # CSVファイルを読み込み (skip_header=2として、ヘッダーをスキップ)
    data = np.genfromtxt(file_path, delimiter=',', skip_header=2)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    # CSV仕様: 1列目が時間（秒）、2列目が電圧（実測値）
    time = data[:, 0]
    voltage = data[:, 1]
    
    # サンプリング周波数の計算
    dt = np.mean(np.diff(time))
    fs = 1.0 / dt
    
    voltage_raw = voltage
    original_time_ms = time * 1000  # 時間軸を秒からミリ秒に換算

    # WaveformPlotterクラスのインスタンス生成
    plotter = WaveformPlotter(
        original_data=[voltage_raw],
        rates=[fs],
        original_time_axes=[original_time_ms],
        file_names=[os.path.basename(file_path)],
        file_path=file_path
    )
    # ユーザ設定のWAVEFORM_PLOT_MODEに応じたモードでグラフを描画
    plotter.plot(mode=WAVEFORM_PLOT_MODE)

if __name__ == "__main__":
    main()