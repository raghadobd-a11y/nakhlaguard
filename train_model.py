"""
تدريب نموذج تصنيف شدة إصابة النخلة - مشروع "نخلة"
================================================
بدل CNN عميق (يحتاج بيانات ضخمة + TensorFlow/PyTorch)، استخدمنا
RandomForest على خصائص Mel-spectrogram + خصائص صوتية إحصائية
(انظر features.py) - هذا أسلوب "Transfer Learning البديل" مناسب
تمامًا لبيانات محدودة (عشرات-مئات العينات) وهو نفس النهج المستخدم
كثيرًا ببحوث تصنيف الأصوات البيولوجية عند ندرة البيانات.

النموذج يصنّف 4 فئات (شدة الإصابة) بدل فئتين فقط - هذي التوسعة
اللي ترفع من قيمة المشروع العملية: تعطي المزارع قرار متدرج
(مراقبة / علاج وقائي / تدخل عاجل) بدل تنبيه ثنائي بسيط.
"""
import os
import sys
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))

from features import extract_features, FEATURE_NAMES  # noqa: E402
from generate_synthetic_data import build_dataset, CLASSES, CLASS_IDS  # noqa: E402

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "nakhla_model.joblib")


def load_or_build_dataset():
    manifest_path = os.path.join(RAW_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        print("ما فيه بيانات محاكاة بعد - جاري توليدها...")
        build_dataset(n_per_class=60, out_dir=RAW_DIR)
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def main():
    manifest = load_or_build_dataset()

    X, y = [], []
    for item in manifest:
        audio = np.load(os.path.join(RAW_DIR, item["file"]))
        X.append(extract_features(audio))
        y.append(item["label_id"])
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("=" * 60)
    print("تقرير الأداء على بيانات الاختبار (محاكاة - وليست ميدانية):")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=CLASSES, digits=3))
    print("مصفوفة الالتباس:")
    print(confusion_matrix(y_test, y_pred))

    # أهم الخصائص - يفيد بشرح النموذج بالعرض التقديمي (Explainability)
    importances = sorted(
        zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1]
    )[:8]
    print("\nأهم 8 خصائص أثّرت على قرار النموذج:")
    for name, imp in importances:
        print(f"  {name}: {imp:.3f}")

    joblib.dump({"model": clf, "classes": CLASSES, "class_ids": CLASS_IDS}, MODEL_PATH)
    print(f"\nتم حفظ النموذج: {MODEL_PATH}")


if __name__ == "__main__":
    main()
