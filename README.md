# 🎙️ SpeechEase — Dysarthria AI Speech Enhancement

**AI Speech Enhancement · Dysarthria Detection · Random Forest (99% Accuracy) · Multi-Language Support · Streamlit Deployment**

[![Try the Live App](https://img.shields.io/badge/🚀_Try_the_Live_App-Click_Here-28a745?style=for-the-badge&logoColor=white)](https://speechease-dysarthriacare.streamlit.app)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/iamhungundji/dysarthria-detection)

---

## 🧠 What is Dysarthria?

**Dysarthria** is a neurological speech disorder caused by conditions like cerebral palsy, Parkinson's disease, ALS, or stroke. It results in:
- 🗣️ Slurred or mumbled speech
- 🐢 Abnormal speaking speed (too slow or too fast)
- 📢 Weak or breathy voice
- 😰 Difficulty pronouncing words clearly

SpeechEase bridges the communication gap for these individuals using AI.

---

## 🖼️ Application Preview

### 🖥️ User Interface Structure
The application features a clean, intuitive interface for selecting languages and communication modes.

<div align="center">
  <img src="Assets/Home Preview/Choose Language.png" width="45%" alt="Choose Language">
  <img src="Assets/Home Preview/Select Mode.png" width="45%" alt="Select Mode">
</div>

### 🔊 Sample Voice Results
Listen to how SpeechEase enhances dysarthric speech into clear, natural audio across different languages:

| Language | Sample Enhanced Audio |
|----------|----------------------|
| **English** | <audio controls src="Assets/Sample Results/bhasini_clear_message[English].mp3"></audio> |
| **Hindi** | <audio controls src="Assets/Sample Results/bhasini_clear_message[Hindi].mp3"></audio> |
| **Marathi** | <audio controls src="Assets/Sample Results/bhasini_clear_message[Marathi].mp3"></audio> |
| **Kannada** | <audio controls src="Assets/Sample Results/bhasini_clear_message[kannada].mp3"></audio> |
| **Tamil** | <audio controls src="Assets/Sample Results/bhasini_clear_message[tamil].mp3"></audio> |

---

## 📊 Data Source

SpeechEase is powered by the **TORGO Database**, a specialized collection of acoustic and articulatory speech from speakers with Dysarthria, developed by the University of Toronto.

- **Sample Count:** 2,000+ high-quality audio recordings.
- **Classes:** Dysarthric vs. Control (Non-Dysarthric).
- **Goal:** Supporting research in speech impairment analysis and real-time detection.

---
## ✨ Key Features

| Feature | Description |
|---|---|
| 🔬 **AI Classification** | Random Forest model trained on the TORGO dataset to detect dysarthric speech via MFCC analysis |
| 📝 **Speech-to-Text (STT)** | Google Speech Recognition transcribes slurred speech to text in the native language |
| 🔊 **Text-to-Speech (TTS)** | Google TTS re-synthesizes the transcript into clear, strong, well-paced audio |
| 🌐 **6-Language UI** | Full UI localization: English, हिंदी, मराठी, தமிழ், తెలుగు, ಕನ್ನಡ |
| 📱 **3 Communication Modes** | Face-to-Face, WhatsApp & Share, and Phone Call modes |

---

## 🗂️ Project Structure

```
Dysarthria/
│
├── README.md                        ← You are here
├── Dysarthria_App_Integration.md    ← ML & Android/Desktop integration guide
│
├── research_model/
│   └── main.ipynb                   ← Full ML pipeline (EDA → Training → Evaluation)
│
├── app_source/
│   ├── app.py                       ← Streamlit app (main entry point)
│   ├── requirements.txt             ← Python dependencies
│   ├── README.md                    ← App-specific setup & run guide
│   └── artifacts/
│       ├── dysarthria_model.pkl     ← Trained Random Forest classifier
│       └── scaler.pkl               ← StandardScaler for MFCC features
│
└── torgo_data/                      ← TORGO dataset (audio files for training)
```

---

## 🤖 The AI Pipeline

Every audio recording passes through a **4-step enhancement pipeline**:

```
🎤 User Speaks
      │
      ▼
┌─────────────────────┐
│  Step 1: CLASSIFY   │  MFCC features → Random Forest → Dysarthric / Clear
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Step 2: TRANSCRIBE │  Google STT → Text in chosen language
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Step 3: SYNTHESIZE │  Google TTS → Clear, strong, well-paced MP3
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Step 4: DELIVER    │  Auto-play / Download / WhatsApp / Phone replay
└─────────────────────┘
```

---

## 📱 Communication Modes

### 🟦 Mode 1 — Face-to-Face
User speaks → AI classifies & transcribes → **Plays clear audio aloud** for the listener (doctor, waiter, family member, etc.)

### 🟩 Mode 2 — WhatsApp & Share
User records message → AI generates clear `.mp3` → **1-tap download + WhatsApp share link**

### 🟪 Mode 3 — Phone Calls
During a live call → user taps "Speak Clear" → AI generates clear audio → **Play it near the phone mic** for the other person to hear

---

## 🌍 Supported Languages

| Language | STT Code | TTS Code |
|----------|----------|----------|
| English  | `en-US`  | `en`     |
| हिंदी (Hindi) | `hi-IN` | `hi`  |
| मराठी (Marathi) | `mr-IN` | `mr` |
| தமிழ் (Tamil) | `ta-IN` | `ta`  |
| తెలుగు (Telugu) | `te-IN` | `te` |
| ಕನ್ನಡ (Kannada) | `kn-IN` | `kn` |

---

## 🧪 ML Model Details

| Parameter | Value |
|---|---|
| **Algorithm** | Random Forest Classifier (Ensemble) |
| **Accuracy** | **~99.0%** (Validation Score) |
| **Dataset** | TORGO (2000 Speech Samples) |
| **Features** | 40 MFCC Coefficients (Mean across frames) |
| **Preprocessing** | StandardScaler Normalization |
| **Execution** | Real-time Inference (<100ms) |

Full training notebook: [`research_model/main.ipynb`](./research_model/main.ipynb)

---

## 🚀 Quick Start

**1. Clone / navigate to the project:**
```bash
cd /Users/omkar/Desktop/Dysarthria
```

**2. Set up the virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
```

**3. Install dependencies:**
```bash
pip install -r app_source/requirements.txt
```

**4. Run the Streamlit app:**
```bash
cd app_source
python -m streamlit run app.py
```

**5. Open in browser:** `http://localhost:8501`

> ⚠️ An active **internet connection** is required for Google STT and TTS APIs.

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `streamlit` | Web application framework |
| `librosa` | Audio loading & MFCC feature extraction |
| `scikit-learn` | Random Forest model & StandardScaler |
| `joblib` | Loading `.pkl` model artifacts |
| `SpeechRecognition` | Google STT integration |
| `gTTS` | Google Text-to-Speech |
| `pydub` + `ffmpeg` | Audio format conversion & processing |
| `numpy` / `pandas` | Data processing |

---


## 👤 Author

Developed with ❤️ by **Omkar Gutal**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/omkar-gutal-a25935249/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/omkargutal)

---

## ⭐ Support

If you find this project helpful, please give it a **Star**! It helps the project grow and reach more developers.

<div align="center">
  <img src="https://img.shields.io/github/stars/omkargutal/Medium-Article-Summariser---MedSum-AI?style=social" alt="Stars">
</div>