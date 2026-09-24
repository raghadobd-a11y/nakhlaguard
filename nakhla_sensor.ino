/*
  ==========================================================
   نخلة - جهاز الاستشعار الميداني منخفض التكلفة
  ==========================================================
  هذا أهم نقطة تميّز عملية بالمشروع: الأبحاث المنشورة (KAUST وغيرها)
  تستخدم أجهزة استشعار صوتي موزّع بالألياف الضوئية (DAS) بتكلفة آلاف
  الدولارات وتحتاج خبرة تركيب متخصصة. هذا التصميم يستبدلها بمستشعر
  اهتزاز بيزو (Piezo Contact Mic) + ESP32 بتكلفة إجمالية أقل من
  100 ريال، يقدر أي مزارع يثبته بنفسه على جذع النخلة.

  المكونات:
  - ESP32 (فيه WiFi مدمج لإرسال البيانات مباشرة للسيرفر)
  - مستشعر بيزو (Piezo Disc) ملاصق لجذع النخلة بشريط لاصق + جل صوتي
  - مقاومة 1M أوم بالتوازي مع البيزو (لتفريغ الشحنة الساكنة)
  - دائرة تكبير بسيطة اختيارية (op-amp LM358) لو الإشارة ضعيفة

  آلية العمل:
  1. يقرأ البيزو الاهتزازات الميكانيكية من داخل الجذع عبر ADC
  2. يجمع عينات لمدة 5 ثوانٍ بمعدل 4000 عينة/ثانية (نفس معدل تدريب النموذج)
  3. يرسلها كـ WAV بسيط عبر HTTP POST لسيرفر FastAPI (/predict-full)
  4. يكرر كل فترة زمنية محددة (مثلاً كل ساعة) أو عند الطلب (زر يدوي)

  ملاحظة: هذا كود أولي (Prototype) لإثبات الفكرة الهندسية - يحتاج ضبط
  دقيق لمعدل العينات الفعلي حسب دقة ADC بـ ESP32 وتجربة ميدانية حقيقية.
*/

#include <WiFi.h>
#include <HTTPClient.h>

// ---------- إعدادات الشبكة والسيرفر ----------
const char* WIFI_SSID     = "ضعي_اسم_الشبكة";
const char* WIFI_PASSWORD = "ضعي_كلمة_المرور";
const char* SERVER_URL    = "http://192.168.1.100:8000/predict-audio"; // رابط سيرفر نخلة
const char* PALM_ID       = "نخلة-A12";

// ---------- إعدادات الاستشعار ----------
const int   PIEZO_PIN        = 34;      // GPIO34 (ADC1) على ESP32
const int   SAMPLE_RATE      = 4000;    // Hz - يطابق تدريب النموذج (features.py)
const int   DURATION_SEC     = 5;
const int   NUM_SAMPLES      = SAMPLE_RATE * DURATION_SEC;
const unsigned long READ_INTERVAL_MS = 1000000UL / SAMPLE_RATE; // بالميكروثانية

int16_t audioBuffer[NUM_SAMPLES];

void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("جاري الاتصال بالشبكة");
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }
  Serial.println("\nتم الاتصال! IP: " + WiFi.localIP().toString());
}

// يبني رأس ملف WAV بسيط (44 بايت) قبل بيانات الصوت الخام
void writeWavHeader(uint8_t* header, int dataSize) {
  int sampleRate = SAMPLE_RATE;
  int byteRate = sampleRate * 2; // 16-bit mono
  int chunkSize = 36 + dataSize;

  memcpy(header, "RIFF", 4);
  memcpy(header + 4, &chunkSize, 4);
  memcpy(header + 8, "WAVE", 4);
  memcpy(header + 12, "fmt ", 4);
  int subchunk1Size = 16;
  memcpy(header + 16, &subchunk1Size, 4);
  int16_t audioFormat = 1, numChannels = 1;
  memcpy(header + 20, &audioFormat, 2);
  memcpy(header + 22, &numChannels, 2);
  memcpy(header + 24, &sampleRate, 4);
  memcpy(header + 28, &byteRate, 4);
  int16_t blockAlign = 2, bitsPerSample = 16;
  memcpy(header + 32, &blockAlign, 2);
  memcpy(header + 34, &bitsPerSample, 2);
  memcpy(header + 36, "data", 4);
  memcpy(header + 40, &dataSize, 4);
}

void recordAudio() {
  Serial.println("جاري تسجيل صوت الجذع...");
  unsigned long nextSampleTime = micros();
  for (int i = 0; i < NUM_SAMPLES; i++) {
    while (micros() < nextSampleTime) { /* انتظار دقيق بين العينات */ }
    int raw = analogRead(PIEZO_PIN);          // 0-4095 (ADC 12-bit)
    audioBuffer[i] = (int16_t)((raw - 2048) * 16); // تحويل لمدى 16-bit موقّع
    nextSampleTime += READ_INTERVAL_MS;
  }
  Serial.println("تم التسجيل، جاري الإرسال...");
}

void sendToServer() {
  if (WiFi.status() != WL_CONNECTED) { connectWiFi(); }

  int dataSize = NUM_SAMPLES * 2;
  uint8_t header[44];
  writeWavHeader(header, dataSize);

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "audio/wav");

  // ملاحظة: لبساطة المثال نرسل الجسم كاملاً دفعة واحدة.
  // بمشروع فعلي يُفضّل multipart/form-data ليطابق /predict-audio
  // بـ FastAPI (UploadFile) - انظري توثيق FastAPI للـ multipart بالـ ESP32.
  uint8_t* payload = (uint8_t*)malloc(44 + dataSize);
  memcpy(payload, header, 44);
  memcpy(payload + 44, audioBuffer, dataSize);

  int httpCode = http.POST(payload, 44 + dataSize);
  Serial.printf("نتيجة الإرسال: %d\n", httpCode);
  if (httpCode == 200) {
    Serial.println(http.getString());
  }
  free(payload);
  http.end();
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  connectWiFi();
}

void loop() {
  recordAudio();
  sendToServer();
  delay(60UL * 60UL * 1000UL); // كرري كل ساعة - عدّليها حسب الحاجة
}
