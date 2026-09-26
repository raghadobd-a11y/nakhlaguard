# 🌴 NakhlaGuard — Early Red Palm Weevil Detection

Full working prototype: data generation → feature extraction → training →
inference → fusion → AI recommendation → agentic AI assistant → web UI →
low-cost hardware path (Arduino + sound sensor).

**Note on structure**: every file sits in one flat folder (no subfolders) so
the whole project can be dragged straight into a GitHub repo without any
folder-upload issues.

## Quick start

```bash
pip install -r requirements.txt --break-system-packages

# 1) Train the model (auto-generates simulated training data the first time)
python train_model.py

# 2) Set your Gemini key (needed for recommendations + the AI assistant)
export GEMINI_API_KEY="your-key-here"      # Windows PowerShell: $env:GEMINI_API_KEY="your-key"

# 3) Run the server
python app.py
# Open: http://localhost:8000/
```

Without a Gemini key, the server still runs and returns a placeholder
recommendation instead of failing.

To test inference directly without the server:
```bash
python predict.py
```

## Project files

```
app.py                     # Main FastAPI server (all routes)
predict.py                 # Inference on a new audio file
features.py                # Mel-spectrogram-style feature extraction
train_model.py             # Trains the RandomForest severity classifier
generate_synthetic_data.py # Simulated training audio generator
fusion.py                  # Audio+image fusion + Gemini recommendation/report
agent.py                   # Agentic AI: tool-calling loop over farm data
gemini_client.py           # Shared Gemini client with retry + model fallback
severity_labels.py         # Arabic->English label translation for AI output
serial_bridge.py           # PC-side bridge for the Arduino Uno version
nakhla_sensor.ino           # ESP32 + WiFi version of the sensor firmware
nakhla_sensor_uno.ino       # Arduino Uno + USB serial version
index.html / test.html / dashboard.html / palms.html / agent.html
requirements.txt
```

## Deploying on Render

- Build Command: `pip install -r requirements.txt && python train_model.py`
- Start Command: `python app.py`
- Environment variable: `GEMINI_API_KEY` = your key

## Why this design? (originality points)

| Element | Published research (KAUST, Prince Sultan University...) | NakhlaGuard |
|---|---|---|
| Sensor | Distributed fiber-optic sensing — expensive, specialist install | Low-cost sound sensor + microcontroller, self-installed |
| Output | Binary classification (infested/healthy) as an academic result | 4 severity stages → a practical, graded decision |
| Modality | Audio only | Audio + image fusion for higher confidence |
| After detection | Stops at classification | Treatment plan + economic estimate + live alert |
| Interaction | One-shot reading | Agentic AI assistant that queries the farm data itself |

**Important for the presentation**: mention explicitly that you reviewed the
published KAUST and Prince Sultan University research in this area, and that
NakhlaGuard's contribution is turning that research (accurate but expensive)
into a cheap, field-usable tool the farmer can operate themselves.

## Known limitations (be transparent about these with the judges)

1. **Training data is simulated** — no public audio dataset exists for Red
   Palm Weevil detection, so the generator produces signals that mimic the
   underlying physics (periodic pulses in the 100-800Hz range over
   background noise) to prove the full technical pipeline works. The real
   next step is recording samples from an actual farm.
2. **RandomForest instead of a deep CNN** — since TensorFlow/PyTorch wasn't
   available in the development environment, a classical model on
   hand-extracted features was used instead; this is a well-known, effective
   approach with limited data, and can be swapped for a CNN later behind the
   same `predict_severity_from_array` interface.
3. **Arduino code is untested on real hardware** — it represents the correct
   engineering design, but needs field calibration (sensor sensitivity,
   mounting, noise isolation) once built.

## Future steps

- Replace simulated data with real recordings from a partner farm
- Try a real CNN (MobileNetV2 on the spectrogram as an image) given a GPU
- Send real multipart data from the Arduino (matching FastAPI's UploadFile)
- Expand the farm dashboard into a full map view of every palm's status
