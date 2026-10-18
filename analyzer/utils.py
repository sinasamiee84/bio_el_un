import io
import base64
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=2):
    """فیلتر میان‌گذر برای حذف نویز Baseline و نویز فرکانس بالا"""
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def process_ecg(file_obj, fs=360):
    # ۱. خواندن فایل CSV
    try:
        df = pd.read_csv(file_obj)
    except Exception as e:
        return None, f"خطا در خواندن فایل: {str(e)}"

    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return None, "هیچ ستون عددی در فایل پیدا نشد."

    # ۲. تشخیص هوشمند ستون سیگنال و زمان
    signal_col = None
    time_col = None

    # جستجوی نام ستون‌ها
    for col in num_cols:
        col_lower = str(col).lower()
        if any(k in col_lower for k in ['time', 'sec', 'زمان']):
            time_col = col
        elif any(k in col_lower for k in ['ecg', 'lead', 'mv', 'volt', 'سیگنال']):
            signal_col = col

    # اگر از نام ستون‌ها پیدا نشد
    if signal_col is None:
        if len(num_cols) > 1:
            first_col = df[num_cols[0]].values
            # بررسی اینکه آیا ستون اول زمان (صعودی یکنواخت) است یا نه
            if np.all(np.diff(first_col[:50]) > 0):
                time_col = num_cols[0]
                signal_col = num_cols[1]
            else:
                signal_col = num_cols[0]
        else:
            signal_col = num_cols[0]

    signal = df[signal_col].to_numpy(dtype=float)

    # تخمین نرخ نمونه‌برداری از روی ستون زمان در صورت وجود
    if time_col is not None:
        dt = np.diff(df[time_col].values[:100])
        valid_dt = dt[dt > 0]
        if len(valid_dt) > 0:
            mean_dt = np.mean(valid_dt)
            if mean_dt > 0:
                fs = int(round(1.0 / mean_dt))

    # ۳. برش حداکثر ۱۰ ثانیه سیگنال
    max_samples = min(len(signal), fs * 10)
    raw_segment = signal[:max_samples]

    # ۴. پیش‌پردازش و حذف نویز با فیلتر میان‌گذر
    try:
        clean_segment = bandpass_filter(raw_segment, fs=fs, lowcut=0.5, highcut=40.0)
    except Exception:
        clean_segment = raw_segment - np.mean(raw_segment)

    time = np.arange(len(clean_segment)) / fs

    # ۵. تشخیص پیک‌های R (مقاوم در برابر نویز با IQR)
    min_distance = int(0.3 * fs)  # حداقل ۳۰۰ میلی‌ثانیه بین دو R
    q75, q25 = np.percentile(clean_segment, [75, 25])
    iqr = q75 - q25
    height_threshold = np.median(clean_segment) + 1.2 * iqr

    peaks, _ = find_peaks(clean_segment, distance=min_distance, height=height_threshold)

    if len(peaks) < 2:
        return None, "تعداد پیک‌های کافی برای محاسبه ضربان شناسایی نشد (لطفاً کیفیت فایل یا نرخ نمونه‌برداری را بررسی کنید)."

    # ۶. تحلیل فواصل ضربان و آریتمی
    rr_intervals = np.diff(peaks) / fs
    avg_hr = 60.0 / np.mean(rr_intervals)

    arrhythmias = []
    if avg_hr > 100:
        arrhythmias.append("تاکی‌کاردی (Tachycardia - ضربان بالاتر از حد نرمال)")
    elif avg_hr < 60:
        arrhythmias.append("برادی‌کاردی (Bradycardia - ضربان پایین‌تر از حد نرمال)")
    else:
        arrhythmias.append("ریتم نرمال سینوسی (Normal Sinus Rhythm)")

    # ارزیابی بی‌نظمی با SDNN
    sdnn = np.std(rr_intervals)
    if sdnn > 0.12:
        arrhythmias.append("احتمال آریتمی سینوسی یا بی‌نظمی ضربان (Irregular RR Intervals)")

    # ۷. رسم نمودار و تبدیل به Base64
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(time, clean_segment, label="Filtered ECG",
            color="#1f77b4", linewidth=1.2)

    ax.plot(time[peaks], clean_segment[peaks], "ro",
            label=f"R-peaks ({len(peaks)} beats)", markersize=5)

    ax.set_title(
        f"ECG Analysis (Sampling Rate: {fs} Hz, HR: {avg_hr:.1f} BPM)",
        fontsize=11
    )
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Amplitude (mV)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    # راهنما بیرونِ نمودار، سمت راست
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    # جا باز کردن برای راهنمای بیرون نمودار
    fig.tight_layout(rect=[0, 0, 0.80, 1])


    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120)
    plt.close(fig)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return {
        'avg_hr': round(avg_hr, 1),
        'num_beats': len(peaks),
        'detected_fs': fs,
        'arrhythmias': arrhythmias,
        'plot_img': image_base64
    }, None
