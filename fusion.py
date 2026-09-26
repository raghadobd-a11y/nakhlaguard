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

# النظام يدعم لغتين الآن - المستخدم يختار (زر تبديل باللغة بالواجهة)
# ويُرسل معه "language" ("en" أو "ar") مع كل طلب، فيولّد Gemini جوابه
# بنفس اللغة المطلوبة بدل ما تكون إنجليزي ثابت دائمًا.
SHORT_SYSTEM_PROMPT_EN = """You are an agricultural assistant specialized in Red Palm \
Weevil infestations in Saudi Arabia.
The palm is in a relatively good condition (healthy or very early infestation). \
Give a short note only (3-4 lines): reassure the farmer, and mention one simple \
preventive monitoring tip.
Respond in English only, plain and clear, with no preamble."""

FULL_REPORT_SYSTEM_PROMPT_EN = """You are an agricultural expert specialized in Red \
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

SHORT_SYSTEM_PROMPT_AR = """أنت مساعد زراعي متخصص بآفة سوسة النخيل الحمراء بالسعودية.
النخلة بحالة جيدة نسبيًا (سليمة أو إصابة مبكرة جدًا). أعطِ ملاحظة قصيرة
فقط (3-4 أسطر): طمّن المزارع، واذكر نصيحة مراقبة وقائية بسيطة واحدة.
أجب بالعربية الفصحى المبسطة فقط، بوضوح، بدون أي مقدمة."""

FULL_REPORT_SYSTEM_PROMPT_AR = """أنت خبير زراعي متخصص بآفة سوسة النخيل الحمراء بالسعودية.
حالة النخلة تستدعي عناية فعلية. اكتب تقريرًا كاملاً ومنظّمًا للمزارع
بنفس العناوين التالية بالضبط (استخدمي ## قبل كل عنوان):

## التشخيص
فقرة قصيرة تشرح معنى التصنيف الحالي بلغة بسيطة يفهمها المزارع.

## خطة العلاج
خطوات عملية مرقّمة وواضحة، بالترتيب الزمني الصحيح (من الأولى للأخيرة).

## الجدول الزمني المقترح
جدول بسيط: ما الذي يُنفَّذ اليوم، وما الذي يُنفَّذ خلال الأسبوع، ومتى يُعاد الفحص.

## التقدير الاقتصادي
مقارنة بسيطة وواقعية (نسبية لا رقمًا دقيقًا) بين تكلفة العلاج الآن والخسارة
المتوقعة لو تأخر العلاج - بأسلوب يوضح لماذا التدخل المبكر أوفر.

## نصيحة أخيرة
جملة واحدة تحفيزية وواقعية.

أجب بالعربية الفصحى المبسطة فقط، بدون أي مقدمة قبل العنوان الأول."""


def build_gemini_prompt(fusion_result, palm_id=None, history=None, location=None, language="en"):
    """
    يبني الرسالة المُرسلة لـ Gemini API. يختار مستوى التفصيل تلقائيًا
    (ملاحظة قصيرة أو تقرير كامل)، ولغة الجواب حسب ما يطلبه المستخدم
    (language: "en" أو "ar").

    بالإنجليزي: نترجم القيم الداخلية (anglicize) عشان الجواب ما يتسرب
    فيه أي كلمة عربية. بالعربي: نُبقي القيم كما هي (عربية أصلاً) لأن
    هذا أطبع لنموذج يجاوب بالعربي.
    """
    severity_value = (
        anglicize(fusion_result["final_class"]) if language == "en" else fusion_result["final_class"]
    )
    context = {
        "severity_class" if language == "en" else "شدة_الإصابة": severity_value,
        "risk_score" if language == "en" else "نسبة_الخطورة": fusion_result["final_score"],
    }
    if palm_id:
        context["palm_id" if language == "en" else "رقم_النخلة"] = palm_id
    if location:
        context["palm_location" if language == "en" else "موقع_النخلة"] = location
    if history:
        hist_value = anglicize(history) if language == "en" else history
        context["check_history" if language == "en" else "السجل_التاريخي"] = hist_value

    needs_full_report = fusion_result["final_class"] in NEEDS_ATTENTION
    if language == "ar":
        system_prompt = FULL_REPORT_SYSTEM_PROMPT_AR if needs_full_report else SHORT_SYSTEM_PROMPT_AR
        intro = "حلّل الحالة التالية:\n"
    else:
        system_prompt = FULL_REPORT_SYSTEM_PROMPT_EN if needs_full_report else SHORT_SYSTEM_PROMPT_EN
        intro = "Analyze the following case:\n"

    user_message = intro + json.dumps(context, ensure_ascii=False, indent=2)
    return system_prompt, user_message


# الاستدعاء الفعلي لـ Gemini (نفس أسلوب بصيرة) - عبر عميل مشترك
# (gemini_client.py) يعيد المحاولة تلقائيًا عند ازدحام الخادم (503/429)
async def get_gemini_recommendation(fusion_result, palm_id=None, history=None, location=None, language="en"):
    from gemini_client import call_gemini

    if language == "ar":
        fallback_text = (
            f"الحالة المصنّفة ({fusion_result['final_class']}) تستدعي "
            "فحصًا ميدانيًا لتأكيد النتيجة قبل اتخاذ قرار العلاج."
        )
        no_key_msg = "⚠️ لم يتم ضبط GEMINI_API_KEY - هذا نص توصية تجريبي بديل: "
        fail_msg = "⚠️ تعذّر الاتصال بـ Gemini حاليًا - هذا نص توصية تجريبي بديل: "
    else:
        fallback_text = (
            f"The classified condition ({anglicize(fusion_result['final_class'])}) "
            "requires a field inspection to confirm the result before deciding on treatment."
        )
        no_key_msg = "⚠️ GEMINI_API_KEY is not set - this is a placeholder recommendation: "
        fail_msg = "⚠️ Could not reach Gemini right now - this is a placeholder recommendation: "

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return no_key_msg + fallback_text

    system_prompt, user_message = build_gemini_prompt(fusion_result, palm_id, history, location, language)
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
        return fail_msg + fallback_text
