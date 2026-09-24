/*
  ==========================================================
   نخلة - جهاز الاستشعار الميداني (نسخة Arduino Uno + USB)
  ==========================================================
  عدّلنا التصميم ليتوافق مع كت Arduino Uno العادي (بدون WiFi
  مدمج)، باستخدام مستشعر الصوت الموجود بالكت (Sound Sensor
  Module) بدل قرص البيزو المتخصص، وإرسال البيانات عبر كيبل
  USB مباشرة للكمبيوتر - أضمن للعرض التنافسي (بدون اعتماد
  على شبكة WiFi بمكان العرض).

  التوصيل:
  - مستشعر الصوت: الطرف VCC -> 5V | GND -> GND
    الطرف AO (الخرج التناظري) -> A0 على Arduino
  - ثبّتي المستشعر ملاصقًا لسطح الجذع (أو نموذج تجريبي) بشريط
    لاصق، وغطّيه بقطعة قماش/إسفنج لعزل ضوضاء الهواء المحيط
    قدر الإمكان وتحسين التقاط الاهتزاز الحقيقي من الجذع.

  آلية العمل:
  1. يقرأ المستشعر عينات تناظرية بمعدل يطابق تدريب النموذج (4000Hz تقريبًا)
  2. يرسلها كأرقام نصية مفصولة بفواصل عبر المنفذ التسلسلي (Serial)
  3. سكربت بايثون بسيط على الكمبيوتر (serial_bridge.py) يستقبلها،
     يبنيها كملف WAV، ويرسلها لسيرفر FastAPI تلقائيًا

  ملاحظة: معدل 4000 عينة/ثانية بالضبط صعب تحقيقه بدقة عالية على
  Arduino Uno (المعالج أبطأ من ESP32) - الكود يحاول الاقتراب منه
  قدر الإمكان؛ لو لاحظتِ تشوّه بالإشارة، قللي SAMPLE_RATE هنا
  وبنفس القيمة بملف features.py على الكمبيوتر (يجب أن يتطابقا دائمًا).
*/

const int   SOUND_PIN     = A0;
const int   SAMPLE_RATE   = 4000;   // Hz - يجب أن يطابق SAMPLE_RATE بـ model/features.py
const int   DURATION_SEC  = 5;
const long  NUM_SAMPLES   = (long)SAMPLE_RATE * DURATION_SEC;
const unsigned long SAMPLE_INTERVAL_US = 1000000UL / SAMPLE_RATE;

const int   TRIGGER_PIN   = 2;      // زر فحص يدوي (اختياري) - وصّلي زر بين D2 و GND
bool        lastButtonState = HIGH;

void setup() {
  Serial.begin(115200);
  pinMode(TRIGGER_PIN, INPUT_PULLUP);
  Serial.println("جاهز - نخلة | اضغطي الزر أو أرسلي 'R' من الكمبيوتر لبدء الفحص");
}

void recordAndSend() {
  Serial.println("START");  // إشارة بداية التسجيل لسكربت البايثون
  unsigned long nextSampleTime = micros();
  for (long i = 0; i < NUM_SAMPLES; i++) {
    while (micros() < nextSampleTime) { /* انتظار دقيق */ }
    int value = analogRead(SOUND_PIN);   // 0-1023
    Serial.println(value);
    nextSampleTime += SAMPLE_INTERVAL_US;
  }
  Serial.println("END");    // إشارة نهاية التسجيل
}

void loop() {
  // فحص الزر اليدوي
  bool buttonState = digitalRead(TRIGGER_PIN);
  if (buttonState == LOW && lastButtonState == HIGH) {
    recordAndSend();
    delay(500); // منع الضغط المتكرر السريع
  }
  lastButtonState = buttonState;

  // فحص أمر من الكمبيوتر عبر Serial (اكتبي R بالـ Serial Monitor)
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'R' || cmd == 'r') {
      recordAndSend();
    }
  }
}
