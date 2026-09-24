"""
عميل مشترك للاتصال بـ Gemini API - يستخدمه fusion.py (التوصية/التقرير
الكامل) و agent.py (الوكيل الذكي) بدل ما كل ملف يكرر نفس منطق الاتصال.

Gemini أحيانًا يرجع:
- 503 (الخادم مشغول مؤقتًا - شائع مع النماذج الجديدة اللي عليها ضغط استخدام)
- 429 (تجاوزتِ حد الطلبات المسموح)

هذي أخطاء عابرة (Transient) تُحل غالبًا بإعادة المحاولة بعد ثانية أو
ثانيتين. لكن لو نموذج معين ضل مزدحم حتى بعد كل المحاولات (شائع مع
النماذج الجديدة جدًا زي gemini-3.8-flash وقت ضغط الاستخدام العالي)،
ننتقل تلقائيًا لنموذج احتياطي أخف وأقدم بدل ما نفشل الطلب كامل.

الاستثناء الوحيد: خطأ 401 (مفتاح غير صالح) يفشل بنفس الشكل مع أي
نموذج، فنرميه فورًا بدون إضاعة وقت بتجربة نماذج إضافية.
"""
import asyncio
import httpx

# نموذج أساسي جديد وقوي، ثم بديلان أقدم وأخف لو الأول مزدحم
FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-3.1-flash-lite"]

TRANSIENT_STATUS_CODES = {429, 503}


async def _attempt_model(client, model_name, payload, headers, max_retries, base_delay):
    """
    يحاول نموذج واحد بعدد محاولات محدد (إعادة محاولة عند ازدحام مؤقت).
    يرجّع (data, None) عند النجاح، أو (None, الخطأ_الأخير) لو فشلت كل
    محاولات هذا النموذج تحديدًا - عندها الطالب ينتقل للنموذج التالي.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json(), None
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 401:
                raise  # مشكلة بالمفتاح نفسه - ما تتحل بتغيير النموذج
            last_error = e
            if status in TRANSIENT_STATUS_CODES and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"[NakhlaGuard] {model_name} -> {status}, retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
                continue
            break  # خطأ غير عابر، أو استنفدنا محاولات هذا النموذج تحديدًا
    return None, last_error


async def call_gemini(payload, api_key, models=None, max_retries=2, base_delay=1.5):
    """
    يجرّب كل نموذج بقائمة models بالترتيب (افتراضيًا FALLBACK_MODELS).
    لو نموذج فشل تمامًا بعد كل محاولاته (503/429 مستمر)، ينتقل تلقائيًا
    للنموذج التالي بالقائمة - يرفع فرصة نجاح الطلب بشكل كبير وقت ضغط
    الاستخدام العالي على نموذج معين (شي متوقع مع نموذج جديد جدًا).
    """
    models = models or FALLBACK_MODELS
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    last_error = None

    async with httpx.AsyncClient(timeout=30) as client:
        for model_name in models:
            data, error = await _attempt_model(client, model_name, payload, headers, max_retries, base_delay)
            if data is not None:
                return data
            print(f"[NakhlaGuard] {model_name} exhausted - trying next fallback model")
            last_error = error

    raise last_error
