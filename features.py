"""
استخراج الخصائص من الصوت لمشروع "نخلة"
====================================
ما استخدمنا librosa (مو متوفرة بكل البيئات) - بنينا استخراج خصائص خفيف
بالكامل على scipy/numpy، يعطي نتائج مشابهة من حيث المبدأ لـ Mel-Spectrogram:

1. Spectrogram عبر scipy.signal.spectrogram (STFT)
2. تجميع الطاقة بحزم تردد Mel-scale تقريبية (Mel-filterbank يدوي مبسّط)
3. خصائص إحصائية إضافية (RMS، Zero-Crossing Rate، Spectral Centroid)
   تفيد خصوصًا نموذج RandomForest/MLP كبديل عملي عن CNN وقت ما تكون
   بيئة التنفيذ بدون TensorFlow/PyTorch (زي حالتنا هنا).
"""
import numpy as np
from scipy.signal import spectrogram

SAMPLE_RATE = 4000
N_MELS = 20


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def mel_filterbank(n_fft_bins, sr, n_mels=N_MELS, fmin=20, fmax=800):
    """بنك مرشحات Mel مبسّط لتجميع طاقة الترددات - بديل خفيف لمكتبة librosa."""
    mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft_bins - 1) * hz_points / (sr / 2)).astype(int)
    bin_points = np.clip(bin_points, 0, n_fft_bins - 1)

    fbank = np.zeros((n_mels, n_fft_bins))
    for m in range(1, n_mels + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for k in range(left, center):
            fbank[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            fbank[m - 1, k] = (right - k) / max(right - center, 1)
    return fbank


def mel_spectrogram(audio, sr=SAMPLE_RATE, n_mels=N_MELS):
    f, t, Sxx = spectrogram(audio, fs=sr, nperseg=256, noverlap=128, mode="magnitude")
    fbank = mel_filterbank(len(f), sr, n_mels=n_mels, fmax=sr / 2 - 1)
    mel_spec = fbank @ Sxx  # (n_mels, n_frames)
    mel_spec_db = 10 * np.log10(mel_spec + 1e-10)
    return mel_spec_db


def extract_features(audio, sr=SAMPLE_RATE):
    """
    يحوّل موجة صوتية خام إلى متجه خصائص ثابت الطول، يجمع بين:
    - إحصائيات Mel-spectrogram (متوسط + انحراف معياري لكل حزمة Mel)
    - خصائص زمنية عامة (RMS، Zero-Crossing Rate، Spectral Centroid)
    """
    mel_db = mel_spectrogram(audio, sr=sr)
    mel_mean = mel_db.mean(axis=1)
    mel_std = mel_db.std(axis=1)

    rms = np.sqrt(np.mean(audio ** 2))
    zero_crossings = np.mean(np.abs(np.diff(np.sign(audio)))) / 2

    f, t, Sxx = spectrogram(audio, fs=sr, nperseg=256, noverlap=128, mode="magnitude")
    power = Sxx.mean(axis=1) + 1e-10
    spectral_centroid = np.sum(f * power) / np.sum(power)

    # نسبة الطاقة داخل نطاق نشاط اليرقات (100-800Hz) - نفس النطاق
    # المستخدم بأبحاث الكشف الصوتي المنشورة عن سوسة النخيل
    band_mask = (f >= 100) & (f <= 800)
    band_energy_ratio = power[band_mask].sum() / power.sum()

    feats = np.concatenate([
        mel_mean, mel_std,
        [rms, zero_crossings, spectral_centroid, band_energy_ratio],
    ])
    return feats.astype(np.float32)


FEATURE_NAMES = (
    [f"mel_mean_{i}" for i in range(N_MELS)]
    + [f"mel_std_{i}" for i in range(N_MELS)]
    + ["rms", "zero_crossing_rate", "spectral_centroid", "band_energy_ratio_100_800hz"]
)
