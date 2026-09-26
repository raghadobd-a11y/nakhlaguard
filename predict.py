"""
الاستدلال (Inference) - مشروع "نخلة"
====================================
يحمّل النموذج المدرّب ويصنّف تسجيل صوتي جديد إلى إحدى فئات شدة الإصابة،
مع نسبة ثقة لكل فئة (مفيد لواجهة المستخدم ولـ Fusion Score لاحقًا).
"""
import os
import numpy as np
import joblib
from scipy.io import wavfile

from features import extract_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "nakhla_model.joblib")
_bundle = None


def _load_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "ما فيه نموذج مدرّب. شغّلي train_model.py أول مرة."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def load_audio_any(path, target_sr=4000):
    """يقرأ ملف .wav أو .npy ويعيد موجة صوتية بمعدل العينات المطلوب."""
    if path.endswith(".npy"):
        return np.load(path), target_sr
    sr, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    if sr != target_sr:
        # إعادة أخذ عينات بسيطة (Linear Resample) بدون الحاجة لمكتبة خارجية
        n_target = int(len(audio) * target_sr / sr)
        audio = np.interp(
            np.linspace(0, len(audio), n_target, endpoint=False),
            np.arange(len(audio)),
            audio,
        )
    return audio.astype(np.float32), target_sr


def predict_severity_from_array(audio, sr=4000):
    bundle = _load_bundle()
    clf, classes = bundle["model"], bundle["classes"]
    feats = extract_features(audio, sr=sr).reshape(1, -1)
    proba = clf.predict_proba(feats)[0]
    pred_idx = int(np.argmax(proba))
    return {
        "predicted_class": classes[pred_idx],
        "confidence": float(proba[pred_idx]),
        "probabilities": {classes[i]: float(p) for i, p in enumerate(proba)},
    }


def predict_severity_from_file(path):
    audio, sr = load_audio_any(path)
    return predict_severity_from_array(audio, sr=sr)


if __name__ == "__main__":
    import sys
    import glob

    # اختبار سريع على عينات محاكاة موجودة
    raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_raw")
    sample_files = sorted(glob.glob(os.path.join(raw_dir, "*.npy")))[:4]
    for f in sample_files:
        result = predict_severity_from_file(f)
        print(os.path.basename(f), "->", result["predicted_class"],
              f"(ثقة {result['confidence']:.0%})")
