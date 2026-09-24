"""
دمج النتائج (Fusion) + التوصية الذكية - مشروع "نخلة"
===================================================
هذا الملف يمثّل نقطة التميّز الأساسية بالمشروع (الأصالة الحقيقية):
الأبحاث المنشورة تتوقف عند "تصنيف الإشارة الصوتية" كنتيجة أكاديمية.
هنا نضيف طبقتين ما وجدناهما بالأبحاث المنشورة:

1. دمج الصوت + الصورة (Multi-modal Fusion) لرفع الثقة بالقرار
2. تحويل التصنيف التقني لتوصية عملية + تقدير خسارة اقتصادية عبر Gemini،
   بما يحوّل "نتيجة نموذج" إلى "قرار يفهمه المزارع مباشرة"
"""
import os
import json
from severity_labels import anglicize

SEVERITY_WEIGHT = {"سليم": 0.0, "مبكرة": 0.33, "متوسطة": 0.66, "متقدمة": 1.0}
SEVERITY_ORDER = ["سليم", "مبكرة", "متوسطة", "متقدمة"]


def fuse_audio_image(audio_result, image_infested_probability=None, image_weight=0.35):
    """
    يدمج نتيجة النموذج الصوتي مع نتيجة تحليل الصورة (لو متوفرة).

    audio_result: مخرجات predict_severity_from_array()
    image_infested_probability: احتمال وجود إصابة من تحليل الصورة (0-1)،
        عادة تجيء من استدعاء Gemini Vision على صورة الجذع (أعراض ظاهرية:
        ثقوب، إفرازات صمغية، نشارة خشب). لو ما فيه صورة، يتجاهل الدمج
        ويعتمد على الصوت فقط.
    """
    audio_score = sum(
        SEVERITY_WEIGHT[cls] * p for cls, p in audio_result["probabilities"].items()
    )

    if image_infested_probability is None:
        final_score = audio_score
        used_fusion = False
    else:
        final_score = (1 - image_weight) * audio_score + image_weight * image_infested_probability
        used_fusion = True

    # تحويل النتيجة المدمجة (0-1) رجوع لأقرب فئة شدة
    idx = min(int(round(final_score * (len(SEVERITY_ORDER) - 1))), len(SEVERITY_ORDER) - 1)
    final_class = SEVERITY_ORDER[idx]

    return {
        "audio_only_score": round(audio_score, 3),
        "fusion_used": used_fusion,
        "final_score": round(final_score, 3),
        "final_class": final_class,
    }


NEEDS_ATTENTION = {"متوسطة", "متقدمة"}

# ملاحظة: النصوص هنا بالإنجليزي بالكامل (بما فيها تعليمات اللغة نفسها)
# عشان مخرجات Gemini تطلع إنجليزي ثابت دايمًا بالديمو، بغض النظر عن
# أي شي - يطابق باقي الموقع اللي صار إنجليزي بالكامل بناءً على متطلبات
# لغة التقديم بمسابقة سيف.
SHORT_SYSTEM_PROMPT = """You are an agricultural assistant specialized in Red Palm \
Weevil infestations in Saudi Arabia.
The palm is in a relatively good condition (healthy or very early infestation). \
Give a short note only (3-4 lines): reassure the farmer, and mention one simple \
preventive monitoring tip.
Respond in English only, plain and clear, with no preamble."""

FULL_REPORT_SYSTEM_PROMPT = """You are an agricultural expert specialized in Red \
Palm Weevil infestations in Saudi Arabia.
The palm's condition requires real attention. Write a complete, organized report \
for the farmer using exactly the following section headers (use ## before each one):

## Diagnosis
A short paragraph explaining what the current classification means in plain terms.

## Treatment Plan
Clear, numbered, practical steps in the correct chronological order (first to last).

## Suggested Timeline
A simple schedule: what to do today, what to do this week, and when to re-check.

## Economic Estimate
A simple, realistic (relative, not an exact figure) comparison between the cost of \
treating now versus the expected loss if treatment is delayed - in a way that \
shows why early intervention is cheaper.

## Final Note
One motivating, realistic closing sentence.

Respond in English only, plain and clear, with no preamble before the first heading."""


def build_gemini_prompt(fusion_result, palm_id=None, history=None, location=None):
    """
    يبني الرسالة المُرسلة لـ Gemini API. يختار مستوى التفصيل تلقائيًا:
    ملاحظة قصيرة للحالات السليمة/المبكرة، وتقرير كامل + خطة علاجية
    مفصّلة للحالات اللي تستدعي عناية فعلية (متوسطة/متقدمة).

    كل القيم تُترجم للإنجليزي قبل الإرسال (anglicize) عشان مخرجات
    Gemini تطلع إنجليزي ثابت دايمًا، حتى لو النموذج الداخلي يصنّف
    بالعربي.
    """
    context = {
        "severity_class": anglicize(fusion_result["final_class"]),
        "risk_score": fusion_result["final_score"],
        "used_audio_and_image_fusion": fusion_result["fusion_used"],
    }
    if palm_id:
        context["palm_id"] = palm_id
    if location:
        context["palm_location"] = location
    if history:
        context["check_history"] = anglicize(history)  # لتتبع تطور الحالة عبر الزمن

    needs_full_report = fusion_result["final_class"] in NEEDS_ATTENTION
    system_prompt = FULL_REPORT_SYSTEM_PROMPT if needs_full_report else SHORT_SYSTEM_PROMPT

    user_message = "Analyze the following case:\n" + json.dumps(context, ensure_ascii=False, indent=2)
    return system_prompt, user_message


# الاستدعاء الفعلي لـ Gemini (نفس أسلوب بصيرة) - عبر عميل مشترك
# (gemini_client.py) يعيد المحاولة تلقائيًا عند ازدحام الخادم (503/429)
async def get_gemini_recommendation(fusion_result, palm_id=None, history=None, location=None):
    from gemini_client import call_gemini

    fallback_text = (
        f"The classified condition ({anglicize(fusion_result['final_class'])}) "
        "requires a field inspection to confirm the result before deciding on treatment."
    )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ GEMINI_API_KEY is not set - this is a placeholder recommendation: " + fallback_text

    system_prompt, user_message = build_gemini_prompt(fusion_result, palm_id, history, location)
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
    }
    try:
        data = await call_gemini(payload, api_key)
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        # نطبع الخطأ الحقيقي بالترمنال عشان يسهل تشخيصه، ونرجّع نص بديل
        # بدل ما نفشل الطلب كامل (500 Internal Server Error)
        print(f"[NakhlaGuard] Gemini call failed: {type(e).__name__}: {e}")
        return "⚠️ Could not reach Gemini right now - this is a placeholder recommendation: " + fallback_text
