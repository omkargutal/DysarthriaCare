# 🎙️ Dysarthria ML Model — Integration Guide

This guide explains how to use the trained ML artifacts (`scaler.pkl` and `dysarthria_model.pkl`) inside **any application** — including the provided Streamlit web app, Android apps, and desktop applications.

---

## 1. The Artifacts

After running `research_model/main.ipynb`, two files are produced:

| File | Purpose |
|---|---|
| `dysarthria_model.pkl` | Trained Random Forest classifier (detects dysarthric speech) |
| `scaler.pkl` | StandardScaler used to normalize the 40 MFCC features before prediction |

These are located in `app_source/artifacts/`.

---

## 2. Core Prediction Logic (Python)

This is the base logic used by any Python-based integration:

```python
import joblib
import librosa
import numpy as np

# Load the AI artifacts
model = joblib.load('artifacts/dysarthria_model.pkl')
scaler = joblib.load('artifacts/scaler.pkl')

def predict_from_audio(audio_path):
    """Returns True if speech is dysarthric, False if clear."""
    y, sr = librosa.load(audio_path, sr=16000, duration=2.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mean_mfcc = np.mean(mfcc.T, axis=0).reshape(1, -1)
    scaled = scaler.transform(mean_mfcc)
    prediction = model.predict(scaled)[0]  # 1 = Dysarthric, 0 = Clear
    confidence = model.predict_proba(scaled)[0].max()
    return prediction == 1, confidence
```

---

## 3. Implementing the 3 Communication Modes (Python / Streamlit)

### 🟦 Mode 1 — Face-to-Face

```python
import speech_recognition as sr
from gtts import gTTS

def face_to_face_mode(audio_path, lang='en'):
    is_dysarthric, conf = predict_from_audio(audio_path)
    if is_dysarthric:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language=lang)
        tts = gTTS(text, lang=lang)
        tts.save("clear_output.mp3")
        # Play for the nearby listener
        import os; os.system("afplay clear_output.mp3")  # macOS
```

### 🟩 Mode 2 — WhatsApp & Share

```python
def whatsapp_mode(audio_path, lang='en'):
    is_dysarthric, _ = predict_from_audio(audio_path)
    if is_dysarthric:
        text = transcribe(audio_path, lang)
        tts = gTTS(text, lang=lang)
        tts.save("clear_message.mp3")
        wa_url = f"https://api.whatsapp.com/send?text={text}"
        return "clear_message.mp3", wa_url
```

### 🟪 Mode 3 — Phone Calls

```python
def phone_mode(audio_path, lang='en'):
    is_dysarthric, _ = predict_from_audio(audio_path)
    if is_dysarthric:
        text = transcribe(audio_path, lang)
        tts = gTTS(text, lang=lang)
        tts.save("clear_call.mp3")
        # Play this near the phone mic for the caller to hear
        return "clear_call.mp3"
```

### ⚠️ Streamlit Cloud Deployment Notes

If you are deploying this Streamlit app to **Streamlit Community Cloud**, keep these environment differences in mind:

1. **System Dependencies (`ffmpeg`)**: The `pydub` library requires `ffprobe` and `ffmpeg` to parse audio. You **must** create a `packages.txt` file in the root of your GitHub repository with the text `ffmpeg` to install these dependencies on the server.
2. **Dynamic Artifact Paths**: Streamlit Cloud runs your app from the repository root. If your script is inside `app_source/`, relative paths like `'artifacts/scaler.pkl'` will fail. Always construct absolute paths dynamically using `__file__`:
   ```python
   import os, joblib
   base_dir = os.path.dirname(os.path.abspath(__file__))
   scaler = joblib.load(os.path.join(base_dir, 'artifacts', 'scaler.pkl'))
   ```

---

## 4. 🤖 Android App Integration

If you have an existing Android app and want to embed this AI pipeline, you have two main approaches:

---

### Option A — Python Microservice (Recommended)

Run the Python model as a **local or cloud REST API** and call it from your Android app via HTTP.

#### Step 1: Create a FastAPI Server

```python
# server.py
from fastapi import FastAPI, UploadFile, File
import joblib, librosa, numpy as np, io
from gtts import gTTS
from fastapi.responses import FileResponse
import speech_recognition as sr
import tempfile, os

app = FastAPI()
model = joblib.load("artifacts/dysarthria_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

@app.post("/enhance")
async def enhance_speech(file: UploadFile = File(...), lang: str = "en"):
    # Save uploaded audio temporarily
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Classify
    y, sr_rate = librosa.load(tmp_path, sr=16000, duration=2.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=40)
    features = scaler.transform(np.mean(mfcc.T, axis=0).reshape(1, -1))
    is_dysarthric = model.predict(features)[0] == 1
    confidence = float(model.predict_proba(features)[0].max())

    # Transcribe
    recognizer = sr.Recognizer()
    with sr.AudioFile(tmp_path) as source:
        audio_data = recognizer.record(source)
    text = recognizer.recognize_google(audio_data, language=f"{lang}-IN")

    # Synthesize clear speech
    out_path = tmp_path.replace(".wav", "_clear.mp3")
    gTTS(text, lang=lang).save(out_path)
    os.unlink(tmp_path)

    return FileResponse(out_path, media_type="audio/mpeg", headers={
        "X-Transcription": text,
        "X-Is-Dysarthric": str(is_dysarthric),
        "X-Confidence": str(round(confidence * 100, 1))
    })
```

**Run the server:**
```bash
pip install fastapi uvicorn
uvicorn server:app --host 0.0.0.0 --port 8000
```

#### Step 2: Call the API from Android (Kotlin)

```kotlin
// In your Android ViewModel or Repository
suspend fun enhanceSpeech(audioFile: File, language: String): EnhancementResult {
    val client = OkHttpClient()

    val requestBody = MultipartBody.Builder()
        .setType(MultipartBody.FORM)
        .addFormDataPart(
            "file", audioFile.name,
            audioFile.asRequestBody("audio/wav".toMediaType())
        )
        .addFormDataPart("lang", language)
        .build()

    val request = Request.Builder()
        .url("http://YOUR_SERVER_IP:8000/enhance")
        .post(requestBody)
        .build()

    val response = client.newCall(request).execute()

    val transcription = response.header("X-Transcription") ?: ""
    val isDysarthric = response.header("X-Is-Dysarthric") == "True"
    val confidence = response.header("X-Confidence")?.toFloat() ?: 0f
    val audioBytes = response.body?.bytes() ?: byteArrayOf()

    return EnhancementResult(transcription, isDysarthric, confidence, audioBytes)
}
```

**Dependencies (build.gradle):**
```gradle
implementation("com.squareup.okhttp3:okhttp:4.12.0")
implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
```

**Android Manifest — add internet permission:**
```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

---

### Option B — On-Device with ONNX (Fully Offline)

Convert the scikit-learn model to ONNX format and run it **directly on the Android device** — no internet or server needed.

#### Step 1: Export the model to ONNX (Python, run once)

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib

model = joblib.load("artifacts/dysarthria_model.pkl")
initial_type = [("mfcc_input", FloatTensorType([None, 40]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("dysarthria_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("✅ Exported to dysarthria_model.onnx")
```

```bash
pip install skl2onnx onnx
```

#### Step 2: Place the model in Android assets

Copy `dysarthria_model.onnx` into your Android project:
```
MyAndroidApp/
└── app/
    └── src/
        └── main/
            └── assets/
                └── dysarthria_model.onnx
```

#### Step 3: Run inference on Android (Kotlin + ONNX Runtime)

```kotlin
import ai.onnxruntime.*

class DysarthriaClassifier(context: Context) {
    private val session: OrtSession
    private val env = OrtEnvironment.getEnvironment()

    init {
        val modelBytes = context.assets.open("dysarthria_model.onnx").readBytes()
        session = env.createSession(modelBytes, OrtSession.SessionOptions())
    }

    fun predict(mfccFeatures: FloatArray): Pair<Boolean, Float> {
        // mfccFeatures must be float[40] — the 40 mean MFCC values
        val tensor = OnnxTensor.createTensor(env, arrayOf(mfccFeatures))
        val result = session.run(mapOf("mfcc_input" to tensor))

        val label = (result[0].value as LongArray)[0]
        val probs = (result[1].value as Array<FloatArray>)[0]
        val confidence = probs.max()

        return Pair(label == 1L, confidence)
    }
}
```

**Dependency (build.gradle):**
```gradle
implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.0")
```

> **Note:** For STT and TTS on Android offline, use:
> - STT: [Vosk Android](https://github.com/alphacep/vosk-android-demo) (offline, supports Indian languages)
> - TTS: Android's built-in `TextToSpeech` API with appropriate language packs installed

---

## 5. 🖥️ Desktop App Integration (Python)

For a desktop app built with **PyQt6**, **Tkinter**, or **Kivy**:

```python
# desktop_integration.py
import joblib, librosa, numpy as np
from gtts import gTTS
import speech_recognition as sr
import playsound, tempfile, os

model = joblib.load("artifacts/dysarthria_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

def run_pipeline(audio_path: str, lang: str = "en") -> dict:
    """Full classify → transcribe → synthesize pipeline for any desktop app."""

    # 1. Classify
    y, sr_rate = librosa.load(audio_path, sr=16000, duration=2.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=40)
    features = scaler.transform(np.mean(mfcc.T, axis=0).reshape(1, -1))
    is_dysarthric = model.predict(features)[0] == 1
    confidence = float(model.predict_proba(features)[0].max())

    # 2. Transcribe
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language=f"{lang}-IN")
    except sr.UnknownValueError:
        return {"error": "Could not transcribe speech."}

    # 3. Synthesize clear speech
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    gTTS(text, lang=lang).save(tmp.name)

    # 4. Play (or return path to the caller)
    playsound.playsound(tmp.name)

    return {
        "is_dysarthric": is_dysarthric,
        "confidence": round(confidence * 100, 1),
        "transcription": text,
        "clear_audio_path": tmp.name
    }
```

**Additional dependency for desktop audio playback:**
```bash
pip install playsound
```

---

## 6. Scaler Values Reference

The `scaler.pkl` stores the **mean** and **variance** of the 40 MFCC features computed from the TORGO training set. If you are re-implementing normalization manually (e.g., in a non-Python environment), you can export these values:

```python
import joblib, json
scaler = joblib.load("artifacts/scaler.pkl")
params = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist()
}
with open("scaler_params.json", "w") as f:
    json.dump(params, f)
print("Saved scaler_params.json — use these to normalize in any language/platform.")
```

Then in any language (Java, Kotlin, Swift, JS), apply:
```
normalized[i] = (raw[i] - mean[i]) / scale[i]
```

---

## 7. Integration Checklist

- [ ] Copy `dysarthria_model.pkl` and `scaler.pkl` into your project
- [ ] Load the model + scaler before processing any audio
- [ ] Extract 40 MFCC features (mean across frames) from 2–3 second audio clips at 16kHz
- [ ] Normalize with the scaler before calling `model.predict()`
- [ ] Connect STT (Google or Vosk) to get the transcript
- [ ] Connect TTS (gTTS or Android TTS) to generate clear audio
- [ ] Choose the correct language code for both STT and TTS
- [ ] Test with at least one dysarthric and one clear audio sample
