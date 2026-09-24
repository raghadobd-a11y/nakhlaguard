"""
مولّد بيانات صوتية محاكاة لمشروع "نخلة"
====================================
ما فيه datasets صوتية عامة متاحة لسوسة النخيل الحمراء (الأبحاث المنشورة
تعتمد على بيانات خاصة بمعدات مخبرية مكلفة زي الألياف الضوئية DAS).
هذا الملف يولّد بيانات محاكاة واقعية قدر الإمكان لإثبات صحة المنهجية
(Proof of Concept) - لازم تستبدلينها ببيانات حقيقية مسجلة من الميدان
وقت ما تتوفر (شوفي README.md).

فيزياء المحاكاة:
- الخلفية الطبيعية لجذع النخلة: ضجيج بني/وردي منخفض (رياح، حركة أوراق)
- صوت مضغ اليرقات: نبضات دورية قصيرة بترددات 100-800Hz (نفس نطاق
  الترشيح المستخدم بأبحاث KAUST المنشورة عن نفس المشكلة)
- شدة الإصابة تُحاكى بزيادة معدل تكرار النبضات وشدتها:
    سليم        -> بدون نبضات
    مبكرة       -> نبضات قليلة ومتباعدة وخافتة
    متوسطة      -> نبضات أكثر وأوضح
    متقدمة      -> نبضات كثيفة ومستمرة تقريبًا
"""
import numpy as np
import os
import json

SAMPLE_RATE = 4000  # Hz - كافي لأن نطاق الاهتمام حتى 800Hz فقط (Nyquist=2000Hz)
DURATION_SEC = 5
CLASSES = ["سليم", "مبكرة", "متوسطة", "متقدمة"]
CLASS_IDS = {c: i for i, c in enumerate(CLASSES)}

# معدل النبضات بالثانية لكل فئة (يمثل نشاط اليرقات)
PULSE_RATE = {"سليم": 0.0, "مبكرة": 0.6, "متوسطة": 2.0, "متقدمة": 5.0}
PULSE_AMP = {"سليم": 0.0, "مبكرة": 0.25, "متوسطة": 0.45, "متقدمة": 0.7}


def pink_noise(n_samples, rng):
    """ضجيج وردي يحاكي الخلفية الطبيعية لجذع النخلة (رياح/احتكاك أوراق)."""
    white = rng.normal(0, 1, n_samples)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1 / SAMPLE_RATE)
    freqs[0] = freqs[1]  # تجنب القسمة على صفر
    fft = fft / np.sqrt(freqs)
    pink = np.fft.irfft(fft, n=n_samples)
    return pink / (np.max(np.abs(pink)) + 1e-9)


def larva_pulse(t_rel, freq=350.0):
    """نبضة واحدة تحاكي صوت قرض/مضغ اليرقة داخل الجذع."""
    envelope = np.exp(-t_rel * 40) * (t_rel >= 0)
    return envelope * np.sin(2 * np.pi * freq * t_rel)


def generate_sample(cls, rng, snr_jitter=True):
    n = int(SAMPLE_RATE * DURATION_SEC)
    t = np.arange(n) / SAMPLE_RATE
    background = 0.15 * pink_noise(n, rng)

    rate = PULSE_RATE[cls]
    amp = PULSE_AMP[cls]
    signal = np.zeros(n)

    if rate > 0:
        # عدد النبضات خلال المدة، بتوزيع شبه عشوائي (Poisson) يحاكي عدم
        # انتظام نشاط اليرقة الحقيقي
        n_pulses = rng.poisson(rate * DURATION_SEC)
        pulse_times = rng.uniform(0, DURATION_SEC - 0.05, size=max(n_pulses, 0))
        freq_center = rng.uniform(200, 500)
        for pt in pulse_times:
            idx0 = int(pt * SAMPLE_RATE)
            length = int(0.05 * SAMPLE_RATE)
            idx1 = min(idx0 + length, n)
            t_rel = t[idx0:idx1] - pt
            this_amp = amp * rng.uniform(0.7, 1.3)
            signal[idx0:idx1] += this_amp * larva_pulse(t_rel, freq=freq_center)

    jitter = rng.uniform(0.85, 1.15) if snr_jitter else 1.0
    audio = background * jitter + signal
    audio = audio / (np.max(np.abs(audio)) + 1e-9) * 0.9
    return audio.astype(np.float32)


def build_dataset(n_per_class=60, seed=42, out_dir="/home/claude/nakhla/data/raw"):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for cls in CLASSES:
        for i in range(n_per_class):
            audio = generate_sample(cls, rng)
            fname = f"{cls}_{i:03d}.npy"
            np.save(os.path.join(out_dir, fname), audio)
            manifest.append({"file": fname, "label": cls, "label_id": CLASS_IDS[cls]})
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"تم توليد {len(manifest)} عينة صوتية محاكاة بـ {out_dir}")
    return manifest


if __name__ == "__main__":
    build_dataset()
