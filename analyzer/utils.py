import io
import base64
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=2):
    """فیلتر میان‌گذر برای حذف نویز Baseline Wander و نویز فرکانس بالا"""
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.0001)
    high = min(highcut / nyq, 0.99)
    if low >= high:
        low = high * 0.1
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def process_ecg(file_obj, fs=None):
    """
    پردازش و رسم سیگنال ECG
    :param file_obj: فایل ارسالی کاربر (CSV)
    :param fs: نرخ نمونه‌برداری (اگر کاربر ارسال کند اولویت دارد)
    """
    try:
        # ۱. اعتبارسنجی و تبدیل fs ورودی کاربر
        user_fs = None
        if fs is not None and str(fs).strip() != '':
            try:
                val = float(fs)
                if val > 0:
                    user_fs = int(round(val))
            except (ValueError, TypeError):
                user_fs = None

        # ۲. خواندن فایل CSV
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)

        try:
            df = pd.read_csv(file_obj)
            if df.empty or not np.issubdtype(df.dtypes.iloc[0], np.number):
                if hasattr(file_obj, 'seek'):
                    file_obj.seek(0)
                df = pd.read_csv(file_obj, header=None)
        except Exception:
            if hasattr(file_obj, 'seek'):
                file_obj.seek(0)
            df = pd.read_csv(file_obj, header=None)

        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            return None, "هیچ ستون عددی در فایل پیدا نشد."

        # ۳. تشخیص هوشمند ستون سیگنال و زمان
        signal_col = None
        time_col = None

        for col in num_cols:
            col_lower = str(col).lower()
            if any(k in col_lower for k in ['time', 'sec', 'زمان', 't']):
                time_col = col
            elif any(k in col_lower for k in ['ecg', 'lead', 'mv', 'volt', 'سیگنال', 'val']):
                signal_col = col

        if signal_col is None:
            if len(num_cols) > 1:
                first_col = df[num_cols[0]].values
                if len(first_col) > 10 and np.all(np.diff(first_col[:50]) > 0):
                    time_col = num_cols[0]
                    signal_col = num_cols[1]
                else:
                    signal_col = num_cols[0]
            else:
                signal_col = num_cols[0]

        signal = df[signal_col].to_numpy(dtype=float)

        # ۴. تعیین نهایی نرخ نمونه‌برداری
        inferred_fs = None
        if time_col is not None:
            time_vals = df[time_col].values
            if len(time_vals) > 1:
                dt = np.diff(time_vals[:200])
                valid_dt = dt[dt > 0]
                if len(valid_dt) > 0 and np.median(valid_dt) > 0:
                    inferred_fs = int(round(1.0 / np.median(valid_dt)))

        if user_fs is not None:
            actual_fs = user_fs
        elif inferred_fs is not None:
            actual_fs = inferred_fs
        else:
            actual_fs = 360  # مقدار پیش‌فرض

        # ۵. برش حداکثر ۱۰ ثانیه سیگنال
        max_samples = min(len(signal), actual_fs * 10)
        raw_segment = signal[:max_samples]

        # ۶. فیلتر کردن سیگنال
        try:
            clean_segment = bandpass_filter(raw_segment, fs=actual_fs, lowcut=0.5, highcut=40.0)
        except Exception:
            clean_segment = raw_segment - np.mean(raw_segment)

        time = np.arange(len(clean_segment)) / actual_fs

        # ۷. تشخیص دقیق‌تر پیک‌های R متناسب با فرکانس (به همراه Prominence برای جلوگیری از خطای لبه و انتهای سیگنال)
        min_distance = max(int(0.35 * actual_fs), 1)  # حداقل فاصله بین دو ضربان
        q75, q25 = np.percentile(clean_segment, [75, 25])
        iqr = q75 - q25
        height_threshold = np.median(clean_segment) + 1.1 * iqr

        # محاسبه آستانه برجستگی پویا برای رد کردن پیک‌های ضعیف انتهای سیگنال
        prominence_threshold = max(0.1 * np.ptp(clean_segment), 0.25 * np.std(clean_segment))

        peaks, _ = find_peaks(
            clean_segment,
            distance=min_distance,
            height=height_threshold,
            prominence=prominence_threshold
        )

        if len(peaks) < 2:
            # جستجوی جایگزین با prominence در صورت پیدا نشدن با ارتفاع
            peaks, _ = find_peaks(
                clean_segment,
                distance=min_distance,
                prominence=prominence_threshold
            )

        if len(peaks) < 2:
            return None, f"تعداد پیک‌های کافی با نرخ {actual_fs} Hz پیدا نشد. لطفاً نرخ نمونه‌برداری را بررسی کنید."

        # ۸. محاسبه فواصل ضربان و آریتمی
        rr_intervals = np.diff(peaks) / actual_fs
        avg_hr = 60.0 / np.mean(rr_intervals)

        arrhythmias = []
        if avg_hr > 100:
            arrhythmias.append("تاکی‌کاردی (Tachycardia - ضربان بالاتر از ۱۰۰)")
        elif avg_hr < 60:
            arrhythmias.append("برادی‌کاردی (Bradycardia - ضربان کمتر از ۶۰)")
        else:
            arrhythmias.append("ریتم نرمال سینوسی (Normal Sinus Rhythm)")

        # بررسی بی‌نظمی ریتم بر اساس ضریب تغییرات RR (CV)
        rr_cv = np.std(rr_intervals) / np.mean(rr_intervals)
        if rr_cv > 0.09:
            arrhythmias.append("احتمال آریتمی / بی‌نظمی در فواصل ضربان (Irregular RR)")

        rhythm_status_str = " | ".join(arrhythmias)

        # ۹. رسم نمودار استاندارد نوار قلب
        fig, ax = plt.subplots(figsize=(13, 4.5), dpi=130)
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#fffbfc')

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which='major', color='#f5a9b8', linestyle='-', linewidth=0.8, alpha=0.8)
        ax.grid(which='minor', color='#fce4ec', linestyle=':', linewidth=0.5, alpha=0.9)

        ax.plot(time, clean_segment, label="Filtered ECG Lead", color="#004d40", linewidth=1.3, zorder=3)

        ax.scatter(time[peaks], clean_segment[peaks], color="#d32f2f", s=38,
                   edgecolors='black', linewidth=0.6, label=f"R-peaks ({len(peaks)} beats)", zorder=4)

        ax.set_title(f"ECG Analysis  |  Sampling Rate: {actual_fs} Hz  |  Heart Rate: {avg_hr:.1f} BPM",
                     fontsize=12, fontweight='bold', pad=12, color='#212121')
        ax.set_xlabel("Time (seconds)", fontsize=10, labelpad=6, fontweight='500')
        ax.set_ylabel("Amplitude (mV)", fontsize=10, labelpad=6, fontweight='500')
        ax.set_xlim(0, max(time))

        for spine in ax.spines.values():
            spine.set_color('#757575')
            spine.set_linewidth(0.8)

        ax.legend(loc="upper right", frameon=True, facecolor='white', framealpha=0.9,
                  edgecolor='#cccccc', fontsize=9)

        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        plt.close(fig)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # ۱۰. برگرداندن تمام کلیدهای احتمالی
        results = {
            'avg_hr': round(avg_hr, 1),
            'num_beats': len(peaks),
            'detected_fs': actual_fs,
            'arrhythmias': arrhythmias,
            'plot_img': image_base64,
            'heart_rate': round(avg_hr, 1),
            'bpm': round(avg_hr, 1),
            'num_peaks': len(peaks),
            'peaks_count': len(peaks),
            'fs_used': actual_fs,
            'rhythm_status': rhythm_status_str,
            'status': rhythm_status_str,
            'plot_base64': image_base64,
            'plot': image_base64,
        }

        return results, None

    except Exception as e:
        plt.close('all')
        return None, f"خطا در پردازش سیگنال: {str(e)}"
