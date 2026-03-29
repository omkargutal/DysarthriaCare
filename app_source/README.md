# 🎙️ BhasiniBridge — Streamlit App

> The Streamlit-powered front-end for the **BhasiniBridge Dysarthria AI Speech Enhancement** system.
> This app lets individuals with dysarthria record their voice and receive clear, strong speech output in their preferred Indian language — across 3 distinct communication modes.

---

## 📁 Directory Structure

```
app_source/
│
├── app.py               ← Main Streamlit application
├── requirements.txt     ← All Python dependencies
└── artifacts/
    ├── dysarthria_model.pkl   ← Trained Random Forest classifier
    └── scaler.pkl             ← StandardScaler for MFCC normalization
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+ (tested on 3.14)
- `pip`
- Active internet connection (for Google STT & TTS APIs)
- `ffmpeg` installed on your system

**Install ffmpeg (macOS):**
```bash
brew install ffmpeg
```

### 1. Create & Activate Virtual Environment
```bash
# From the project root
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Python Dependencies
```bash
cd app_source
pip install -r requirements.txt
```

---

## 🚀 Running the App

```bash
# From within app_source/
python -m streamlit run app.py
```

The app will open at: **`http://localhost:8501`**

---

## 🔄 The Enhancement Pipeline

Every audio clip recorded in the app is processed through this pipeline:

```
🎤 Microphone Input
    │
    ▼
[1] CLASSIFY  →  MFCC extraction + Random Forest model
                 → Result: "Dysarthric" or "Clear"
    │
    ▼
[2] TRANSCRIBE →  Google Speech-to-Text (language-aware)
                  → Result: Plain text transcript
    │
    ▼
[3] SYNTHESIZE →  Google Text-to-Speech (gTTS)
                  → Result: Clear, strong, well-paced .mp3
    │
    ▼
[4] DELIVER →  Auto-play / Download / WhatsApp link / Phone replay
```

---

## 📱 Application Modes

### 🟦 Face-to-Face Mode
- User speaks into microphone
- AI pipeline processes speech immediately
- Clear audio **auto-plays aloud** for the nearby listener
- Ideal for: doctors, waiters, shop assistants

### 🟩 WhatsApp & Share Mode
- User records a voice message
- AI generates a clear `.mp3`
- One-tap **Download** button + **WhatsApp share link**
- Ideal for: messaging, voice notes, remote communication

### 🟪 Phone Call Mode
- User presses "Tap to Speak" during a live call
- AI generates clear speech immediately
- User **plays clear audio near the phone mic**
- The person on the other end hears the clear voice
- Ideal for: phone calls, video calls

---

## 🌍 Language Support

The UI and all AI processing (STT + TTS) are fully localized for:

| Language | Display Name |
|----------|-------------|
| English  | English |
| हिंदी | Hindi |
| मराठी | Marathi |
| தமிழ் | Tamil |
| తెలుగు | Telugu |
| ಕನ್ನಡ | Kannada |

The entire UI — buttons, instructions, status messages, and results — switches to the selected language automatically via the built-in `TRANSLATIONS` dictionary and `_t()` helper.

---

## 📦 Dependencies (`requirements.txt`)

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `librosa` | Audio loading & MFCC feature extraction |
| `scikit-learn` | Random Forest classifier & StandardScaler |
| `joblib` | Loading `.pkl` artifacts |
| `numpy`, `pandas` | Numerical processing |
| `matplotlib`, `seaborn` | (Research/debug visualization) |
| `SpeechRecognition` | Google STT wrapper |
| `gTTS` | Google Text-to-Speech |
| `pydub` | Audio manipulation |
| `ffmpeg` | Audio format backend (required by pydub) |

---

## 🗝️ Key Session State Variables

| Variable | Type | Description |
|---|---|---|
| `app_state` | `str` | Current screen: `language_selection`, `home`, `face_to_face`, `whatsapp_mode`, `phone_mode` |
| `language` | `str` | Selected language key (e.g., `"हिंदी (Hindi)"`) |

---

## 🐛 Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: librosa` | Run `pip install -r requirements.txt` |
| STT returns nothing | Check internet connection; speak clearly and louder |
| TTS audio not generated | Verify internet connection; gTTS requires Google servers |
| `pydub` audio errors | Install `ffmpeg`: `brew install ffmpeg` (macOS) |
| App won't start | Ensure you're running from inside `app_source/` with the venv activated |
| Model file not found | Verify `artifacts/dysarthria_model.pkl` and `artifacts/scaler.pkl` exist |

---

## 📌 Important Notes

- **Internet Required**: Google STT and gTTS are cloud APIs — no offline use.
- **Microphone Permission**: The browser must be granted microphone access for `st.audio_input()` to work.
- **Audio Duration**: Speak for at least **2–3 seconds** for reliable MFCC feature extraction.
- The pipeline processes speech **even if classified as clear** — it still generates an enhanced TTS version for consistent sharing quality.
