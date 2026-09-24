"""
ترجمة تصنيفات الشدة العربية (الداخلية بقاعدة البيانات والنموذج المدرّب)
إلى إنجليزي - يُستخدم فقط عند تجهيز البيانات لإرسالها لـ Gemini
(توصية، تقرير، أو أدوات الوكيل الذكي).

السبب: النموذج المدرّب والـ Database يستخدمون تصنيفات عربية
(سليم/مبكرة/متوسطة/متقدمة) لأنها كذا من أول المشروع، وإعادة تدريب
النموذج بأسماء إنجليزية يكسر النموذج المدرّب الموجود عندك حاليًا.
بدل كذا، نترجم القيم فقط لحظة إرسالها لـ Gemini - يضمن ما يتسرب أي
نص عربي داخل مخرجات الذكاء الاصطناعي بالديمو، مع بقاء النموذج
والواجهات الحالية شغّالة بدون أي تعديل إضافي.
"""
SEVERITY_EN = {
    "سليم": "healthy",
    "مبكرة": "early",
    "متوسطة": "moderate",
    "متقدمة": "advanced",
}

# أسماء الحقول اللي تحمل قيمة تصنيف شدة (عبر كل الملفات: fusion.py، app.py)
SEVERITY_KEYS = {"class", "predicted_class", "latest_status", "final_class", "severity_class"}


def anglicize(value):
    """
    يترجم أي نص عربي معروف لشدة الإصابة داخل value (تعمل بشكل متكرر
    Recursive على dict/list متداخلة) إلى الإنجليزي المقابل، وتترك أي
    قيمة أخرى (مواقع، أرقام، تواريخ) كما هي بدون أي تغيير.
    """
    if isinstance(value, dict):
        return {
            k: (SEVERITY_EN.get(v, v) if k in SEVERITY_KEYS and isinstance(v, str) else anglicize(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [anglicize(v) for v in value]
    if isinstance(value, str):
        return SEVERITY_EN.get(value, value)
    return value
