"""
جسر الاتصال بين Arduino والسيرفر - مشروع "نخلة"
================================================
يشتغل على الكمبيوتر المتصل بـ Arduino عبر USB:
1. يقرأ عينات الصوت المرسلة من nakhla_sensor_uno.ino عبر Serial
2. يبنيها كملف WAV
3. يرسله لسيرفر FastAPI (/predict-full) ويطبع النتيجة

الاستخدام:
    pip install pyserial requests
    python serial_bridge.py COM3        # على ويندوز
    python serial_bridge.py /dev/ttyUSB0  # على لينكس/ماك

اضغطي زر الفحص المتصل بـ Arduino (أو اكتبي R بالمنفذ التسلسلي)
لبدء تسجيل جديد.
"""
import sys
import time
import numpy as np
import requests
import serial
from scipy.io import wavfile

SAMPLE_RATE = 4000  # يجب أن يطابق SAMPLE_RATE بملف الأردوينو وfeatures.py
SERVER_URL = "http://localhost:8000/predict-full"
PALM_ID = "نخلة-تجريبية"


def read_one_recording(ser):
    """يقرأ تسجيل واحد كامل بين إشارتي START و END من Arduino."""
    samples = []
    print("بانتظار بدء تسجيل جديد (اضغطي الزر بـ Arduino)...")
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if line == "START":
            print("جاري استقبال العينات...")
            samples = []
        elif line == "END":
            print(f"اكتمل التسجيل: {len(samples)} عينة")
            return np.array(samples, dtype=np.float32)
        elif line.isdigit() or (line.startswith("-") and line[1:].isdigit()):
            samples.append(int(line))


def samples_to_wav(samples, path):
    # تحويل من مدى ADC (0-1023) إلى موجة موقّعة طبيعية
    centered = samples - np.mean(samples)
    normalized = centered / (np.max(np.abs(centered)) + 1e-9)
    audio_int16 = (normalized * 32767 * 0.9).astype(np.int16)
    wavfile.write(path, SAMPLE_RATE, audio_int16)


def send_to_server(wav_path):
    with open(wav_path, "rb") as f:
        files = {"audio": ("recording.wav", f, "audio/wav")}
        data = {"palm_id": PALM_ID}
        resp = requests.post(SERVER_URL, files=files, data=data, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    if len(sys.argv) < 2:
        print("الاستخدام: python serial_bridge.py <PORT>  مثال: COM3 أو /dev/ttyUSB0")
        sys.exit(1)

    port = sys.argv[1]
    ser = serial.Serial(port, 115200, timeout=2)
    time.sleep(2)  # وقت إعادة تشغيل Arduino بعد فتح المنفذ

    while True:
        samples = read_one_recording(ser)
        if len(samples) < SAMPLE_RATE:  # تسجيل ناقص/فاشل
            print("تسجيل قصير جدًا، تجاهلته. جربي مرة ثانية.")
            continue

        wav_path = "recording_tmp.wav"
        samples_to_wav(samples, wav_path)
        print("جاري الإرسال للسيرفر...")
        try:
            result = send_to_server(wav_path)
            cls = result["fusion_result"]["final_class"]
            print(f"\n>>> النتيجة: {cls}")
            print(">>> التوصية:", result["recommendation"], "\n")
        except Exception as e:
            print("خطأ بالاتصال بالسيرفر (تأكدي إن app.py شغّال):", e)


if __name__ == "__main__":
    main()
