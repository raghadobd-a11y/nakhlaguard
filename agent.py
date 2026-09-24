"""
وكيل نخلة الذكي (Agentic AI) - مشروع NakhlaGuard
=================================================
الفرق بين هذا وبين استدعاء Gemini العادي بـ fusion.py:
fusion.py يرسل سؤال واحد ويستقبل جواب واحد (نمط "استدعاء" بسيط).

هذا الملف وكيل حقيقي (Agent): يعطى الوصول لأدوات حقيقية تقرأ من
قاعدة بيانات المزرعة، ويقرر بنفسه (بدون مسار مبرمج مسبقًا):
- هل يحتاج بيانات فعلية للإجابة، ولو يحتاج، أي أداة يستخدم؟
- هل جواب أداة وحدة كافي، ولا يحتاج يستدعي أداة ثانية؟
- متى يتوقف ويعطي الجواب النهائي للمزارع؟

هذا نمط "Tool Use / Function Calling Loop" المعياري لبناء وكلاء
ذكاء اصطناعي، مبني على generateContent API نفسه عبر عميل مشترك
(gemini_client.py) يعيد المحاولة تلقائيًا عند ازدحام خادم Gemini
المؤقت (503/429) - بدون الحاجة لأي مكتبة خارجية إضافية.
"""
import os
from gemini_client import call_gemini
from severity_labels import anglicize

AGENT_SYSTEM_PROMPT = """You are "Farm Assistant" - an AI agent inside the \
NakhlaGuard system that helps a palm farmer track the health of their palms.

You have real tools that connect you to the actual farm data:
- list_palms: every registered palm, its location, and its latest check status
- get_palm_details: full details for one specific palm (location, notes, and \
its complete check history over time)
- palms_needing_attention: only the palms whose current status is moderate or \
advanced (need the farmer's attention now)

Core rule: any question about real farm data (how many palms, where is palm X, \
what's the status of X, which palms need attention) - you must call the right \
tool before answering. Never guess or invent numbers or locations.

If the question is general knowledge about Red Palm Weevil (not about data \
registered in this system), answer directly from your own agricultural \
knowledge without calling any tool.

Always respond in English only, regardless of the language the farmer wrote \
in - be concise, clear, and practically useful."""

TOOLS = [{
    "function_declarations": [
        {
            "name": "list_palms",
            "description": (
                "Get every registered palm tree with its location and its "
                "most recent health status (healthy / early / moderate / advanced)."
            ),
            "parameters": {"type": "OBJECT", "properties": {}},
        },
        {
            "name": "get_palm_details",
            "description": (
                "Get full details for one specific palm tree: its location, "
                "notes, and its complete check history over time (every "
                "past reading with date and severity)."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "palm_id": {
                        "type": "STRING",
                        "description": "The palm's ID exactly as registered, e.g. Palm-A12",
                    },
                },
                "required": ["palm_id"],
            },
        },
        {
            "name": "palms_needing_attention",
            "description": (
                "Get only the palms whose latest check came back moderate "
                "or advanced infestation - i.e. palms that currently need "
                "the farmer's attention right now."
            ),
            "parameters": {"type": "OBJECT", "properties": {}},
        },
    ]
}]


async def run_agent(user_message, tool_executor, history=None, max_steps=5):
    """
    حلقة الوكيل (Agent Loop):
    1. يرسل رسالة المزارع لـ Gemini مع تعريف الأدوات المتاحة
    2. لو النموذج قرر يستدعي أداة (functionCall) -> ننفذها فعليًا
       بقاعدة البيانات الحقيقية، ونرجّع نتيجتها له (functionResponse)
    3. يكرر هذا لين النموذج يوصل لجواب نهائي (نص) أو نوصل للحد الأقصى
       من الخطوات (max_steps) كحماية من التكرار اللانهائي

    tool_executor: dict {اسم_الأداة: دالة بايثون فعلية تنفذها} - لازم
    نفس الأسماء المعرّفة بـ TOOLS بالضبط.
    history: قائمة رسائل سابقة بصيغة contents (لمحادثة متعددة الأدوار).

    يرجّع: {
        "answer": الجواب النهائي كنص,
        "trace": [{"tool": اسم_الأداة, "args": المدخلات}, ...] بترتيب
                 الاستدعاء - مفيد لعرض "كيف فكّر الوكيل" بالعرض التقديمي
    }
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {
            "answer": "⚠️ GEMINI_API_KEY is not set - the AI assistant needs it to work.",
            "trace": [],
        }

    contents = list(history) if history else []
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    trace = []

    for _ in range(max_steps):
        payload = {
            "system_instruction": {"parts": [{"text": AGENT_SYSTEM_PROMPT}]},
            "contents": contents,
            "tools": TOOLS,
        }
        try:
            data = await call_gemini(payload, api_key)
        except Exception as e:
            print(f"[NakhlaGuard Agent] call failed: {type(e).__name__}: {e}")
            return {
                "answer": f"⚠️ Could not reach the AI assistant right now ({type(e).__name__}).",
                "trace": trace,
            }

        candidate = data["candidates"][0]["content"]
        parts = candidate.get("parts", [])
        function_call_part = next((p for p in parts if "functionCall" in p), None)

        if function_call_part is None:
            # النموذج ما طلب أي أداة - هذا جوابه النهائي
            final_text = "".join(p.get("text", "") for p in parts)
            return {"answer": final_text, "trace": trace}

        # النموذج قرر بنفسه يستدعي أداة - ننفذها فعليًا على بياناتنا
        fn_name = function_call_part["functionCall"]["name"]
        fn_args = function_call_part["functionCall"].get("args", {}) or {}
        trace.append({"tool": fn_name, "args": fn_args})

        contents.append({"role": "model", "parts": [function_call_part]})

        executor = tool_executor.get(fn_name)
        if executor is None:
            tool_result = {"error": f"Unknown tool: {fn_name}"}
        else:
            try:
                tool_result = executor(**fn_args)
            except Exception as e:
                tool_result = {"error": str(e)}

        # نترجم أي تصنيف شدة عربي (سليم/مبكرة/متوسطة/متقدمة) للإنجليزي
        # قبل ما يوصل النموذج - يضمن ما يتسرب نص عربي بجواب الوكيل
        tool_result = anglicize(tool_result)

        contents.append({
            "role": "user",
            "parts": [{
                "functionResponse": {
                    "name": fn_name,
                    "response": {"result": tool_result},
                }
            }],
        })

    return {
        "answer": "⚠️ The agent used several tools without reaching a final answer - try rephrasing your question.",
        "trace": trace,
    }
