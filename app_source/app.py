import streamlit as st
import numpy as np
import librosa
import joblib
import io
import os
import time
import tempfile
import base64
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment

# Basic configuration
st.set_page_config(page_title="BhasiniBridge - Dysarthria AI", page_icon="🧠", layout="centered", initial_sidebar_state="collapsed")

# Language code mapping for gTTS and Speech Recognition
LANGUAGE_CONFIG = {
    "English": {"tts": "en", "stt": "en-US", "label": "English"},
    "हिंदी (Hindi)": {"tts": "hi", "stt": "hi-IN", "label": "Hindi"},
    "मराठी (Marathi)": {"tts": "mr", "stt": "mr-IN", "label": "Marathi"},
    "தமிழ் (Tamil)": {"tts": "ta", "stt": "ta-IN", "label": "Tamil"},
    "తెలుగు (Telugu)": {"tts": "te", "stt": "te-IN", "label": "Telugu"},
    "ಕನ್ನಡ (Kannada)": {"tts": "kn", "stt": "kn-IN", "label": "Kannada"},
}

# UI Translation Dictionary
TRANSLATIONS = {
    "English": {
        "title": "BhasiniBridge",
        "choose_lang": "Choose Language",
        "select_pref": "Select your preferred language to begin",
        "users": "Users",
        "languages": "Languages",
        "selected": "Selected",
        "latency": "0ms Latency",
        "choose_mode": "Choose Mode",
        "f2f_title": "Face-to-Face",
        "f2f_desc": "Speak → AI clarifies → Plays clear voice aloud for the listener.",
        "f2f_btn": "Start Face-to-Face Mode",
        "wa_title": "WhatsApp & Share",
        "wa_desc": "Record → Clarify → Download or share clear audio via WhatsApp.",
        "wa_btn": "Start WhatsApp Mode",
        "phone_title": "Phone Calls",
        "phone_desc": "Record during a call → AI generates clear speech to replay for the listener.",
        "phone_btn": "Start Phone Call Mode",
        "back_lang": "← Change Language",
        "back_dash": "← Back to Dashboard",
        "tap_to_speak": "🎙️ Tap to Speak",
        "speak_into_mic": "Speak into the mic → AI converts to clear speech in",
        "processing_pipeline": "🔄 Processing Pipeline",
        "step1_msg": "🔬 Step 1: Analyzing speech patterns with AI model...",
        "step1_done": "✅ Classification: {status} — Confidence: {conf}%",
        "dysarthric": "Dysarthric Speech Detected",
        "clear_speech": "Clear Speech Detected",
        "step2_msg": "📝 Step 2: Transcribing speech in {lang}...",
        "step2_done": "✅ Transcribed: \"{text}\"",
        "step2_fail": "⚠️ Could not transcribe speech. Please speak louder or try again.",
        "step3_msg": "🔊 Step 3: Generating clear, strong speech in {lang}...",
        "step3_done": "✅ Clear speech generated! Clearer voice • Better speed • Stronger audio",
        "step3_clear": "🔊 Step 3: Speech is clear! Generating enhanced version...",
        "step3_fail": "❌ TTS generation failed. Check your internet connection.",
        "clear_ready": "Clear Speech Ready",
        "enhanced_voice": "Enhanced • Clearer • Stronger Voice",
        "transcription_label": "TRANSCRIPTION ({lang})",
        "play_listener": "▶️ ENHANCED CLEAR AUDIO — Play this for the listener:",
        "clear_detected_title": "Clear Speech Detected!",
        "no_enhancement": "No enhancement needed — speech is already clear.",
        "how_it_works": "How it works",
        "how_step1": "Record: The dysarthric user speaks into the microphone.",
        "how_step2": "Classify: AI model detects dysarthric speech patterns (MFCC analysis).",
        "how_step3": "Transcribe: Speech-to-Text extracts what the user is trying to say.",
        "how_step4": "Speak Clear: AI generates clear, strong voice — play it aloud for the listener.",
        "wa_preview": "▶️ PREVIEW CLEAR AUDIO:",
        "download_clear": "⬇️ Download Clear Audio",
        "share_wa": "📲 Share on WhatsApp",
        "wa_tip": "💡 Tip: Download the audio file, then attach it in WhatsApp for the best experience.",
        "phone_mode_banner": "During a call, record → AI generates clear speech → Replay",
        "active_call": "Active Call Mode",
        "phone_instruction": "Press \"Speak Clear\" when you want to say something",
        "phone_tip_box": "The AI will capture your voice and replay a clear version",
        "phone_bridge": "🔄 AI Speech Bridge Processing",
        "play_near_mic": "🔊 PLAY CLEAR SPEECH — Hold your phone to the speaker:",
        "phone_tip": "💡 Tip: Put phone on speaker, then play this audio near the mic.",
        "ai_powered": "AI Powered",
        "integration": "Integration",
        "voice_only": "Voice Only"
    },
    "हिंदी (Hindi)": {
        "title": "भाषिणी ब्रिज (BhasiniBridge)",
        "choose_lang": "भाषा चुनें",
        "select_pref": "शुरू करने के लिए अपनी पसंदीदा भाषा चुनें",
        "users": "उपयोगकर्ता",
        "languages": "भाषाएं",
        "selected": "चुनी गई",
        "latency": "0ms की देरी",
        "choose_mode": "मोड चुनें",
        "f2f_title": "आमने-सामने (Face-to-Face)",
        "f2f_desc": "बोलें → AI स्पष्ट करेगा → सुनने वाले के लिए स्पष्ट आवाज़ बजाएगा।",
        "f2f_btn": "आमने-सामने मोड शुरू करें",
        "wa_title": "WhatsApp और शेयर",
        "wa_desc": "रिकॉर्ड करें → स्पष्ट करें → WhatsApp के माध्यम से स्पष्ट ऑडियो डाउनलोड या शेयर करें।",
        "wa_btn": "WhatsApp मोड शुरू करें",
        "phone_title": "फोन कॉल्स",
        "phone_desc": "कॉल के दौरान रिकॉर्ड करें → AI सुनने वाले के लिए स्पष्ट आवाज़ तैयार करेगा।",
        "phone_btn": "फोन कॉल मोड शुरू करें",
        "back_lang": "← भाषा बदलें",
        "back_dash": "← डैशबोर्ड पर वापस जाएं",
        "tap_to_speak": "🎙️ बोलने के लिए टैप करें",
        "speak_into_mic": "माइक में बोलें → AI इसे स्पष्ट आवाज़ में बदलेगा:",
        "processing_pipeline": "🔄 प्रोसेसिंग पाइपलाइन",
        "step1_msg": "🔬 चरण 1: AI मॉडल के साथ भाषण पैटर्न का विश्लेषण...",
        "step1_done": "✅ वर्गीकरण: {status} — विश्वास: {conf}%",
        "dysarthric": "अस्पष्ट भाषण (Dysarthria) पाया गया",
        "clear_speech": "स्पष्ट भाषण पाया गया",
        "step2_msg": "📝 चरण 2: {lang} में अनुवाद किया जा रहा है...",
        "step2_done": "✅ अनुवादित: \"{text}\"",
        "step2_fail": "⚠️ भाषण का अनुवाद नहीं हो सका। कृपया जोर से बोलें या फिर से प्रयास करें।",
        "step3_msg": "🔊 चरण 3: {lang} में स्पष्ट और मजबूत आवाज़ तैयार की जा रही है...",
        "step3_done": "✅ स्पष्ट आवाज़ तैयार! स्पष्ट आवाज़ • बेहतर गति • मजबूत ऑडियो",
        "step3_clear": "🔊 चरण 3: भाषण स्पष्ट है! बेहतर संस्करण तैयार किया जा रहा है...",
        "step3_fail": "❌ आवाज़ तैयार नहीं हो सकी। अपना इंटरनेट कनेक्शन जांचें।",
        "clear_ready": "स्पष्ट आवाज़ तैयार",
        "enhanced_voice": "बेहتر • स्पष्ट • मजबूत आवाज़",
        "transcription_label": "अनुवाद ({lang})",
        "play_listener": "▶️ बेहतर स्पष्ट ऑडियो — इसे सुनने वाले के लिए बजाएं:",
        "clear_detected_title": "स्पष्ट आवाज़ पाई गई!",
        "no_enhancement": "किसी सुधार की आवश्यकता नहीं है — आवाज़ पहले से ही स्पष्ट है।",
        "how_it_works": "यह कैसे काम करता है",
        "how_step1": "रिकॉर्ड: उपयोगकर्ता माइक्रोफ़ोन में बोलता है।",
        "how_step2": "वर्गीकरण: AI मॉडल भाषण पैटर्न का पता लगाता है।",
        "how_step3": "अनुवाद: स्पीच-టు-టెక్స్ట్ పద్ధతిలో మాటలను అర్థం చేసుకుంటుంది.",
        "how_step4": "स्पष्ट बोलें: AI स्पष्ट आवाज़ तैयार करता है — इसे सुनने वाले के लिए बजाएं।",
        "wa_preview": "▶️ स्पष्ट ऑडियो का पूर्वावलोकन:",
        "download_clear": "⬇️ स्पष्ट ऑडियो डाउनलोड करें",
        "share_wa": "📲 WhatsApp पर शेयर करें",
        "wa_tip": "💡 सलाह: ऑडियो फ़ाइल डाउनलोड करें, फिर बेहतर अनुभव के लिए इसे WhatsApp में अटैच करें।",
        "phone_mode_banner": "कॉल के दौरान, रिकॉर्ड करें → AI स्पष्ट आवाज़ बनाएगा → दोबारा बजाएं",
        "active_call": "सक्रिय कॉल मोड",
        "phone_instruction": "जब आप कुछ कहना चाहें तो \"स्पष्ट बोलें\" दबाएं",
        "phone_tip_box": "AI आपकी आवाज़ पकड़ लेगा और स्पष्ट संस्करण बजाएगा",
        "phone_bridge": "🔄 AI स्पीच ब्रिज प्रोसेसिंग",
        "play_near_mic": "🔊 स्पष्ट भाषण बजाएं — अपना फोन स्पीकर के पास रखें:",
        "phone_tip": "💡 सलाह: फोन को स्पीकर पर रखें, फिर इस ऑडियो को माइक के पास बजाएं।",
        "ai_powered": "AI संचालित",
        "integration": "एकीकरण",
        "voice_only": "केवल आवाज़"
    },
    "मराठी (Marathi)": {
        "title": "भाषिणी ब्रिज (BhasiniBridge)",
        "choose_lang": "भाषा निवडा",
        "select_pref": "सुरू करण्यासाठी तुमची आवडती भाषा निवडा",
        "users": "वापरकर्ते",
        "languages": "भाषा",
        "selected": "निवडलेली",
        "latency": "0ms विलंब",
        "choose_mode": "मोड निवडा",
        "f2f_title": "आमने-सामने (Face-to-Face)",
        "f2f_desc": "बोला → AI स्पष्ट करेल → ऐकणाऱ्यासाठी स्पष्ट आवाज वाजवेल.",
        "f2f_btn": "आमने-सामने मोड सुरू करा",
        "wa_title": "WhatsApp आणि शेअर",
        "wa_desc": "रेकॉर्ड करा → स्पष्ट करा → WhatsApp द्वारे स्पष्ट ऑडिओ डाउनलोड किंवा शेअर करा.",
        "wa_btn": "WhatsApp मोड सुरू करा",
        "phone_title": "फोन कॉल्स",
        "phone_desc": "कॉल दरम्यान रेकॉर्ड करा → AI ऐकणाऱ्यासाठी स्पष्ट आवाज तयार करेल.",
        "phone_btn": "फोन कॉल मोड सुरू करा",
        "back_lang": "← भाषा बदला",
        "back_dash": "← डॅशबोर्डवर परत जा",
        "tap_to_speak": "🎙️ बोलण्यासाठी टॅप करा",
        "speak_into_mic": "माईकमध्ये बोला → AI स्पष्ट आवाजात रूपांतर करेल:",
        "processing_pipeline": "🔄 प्रोसेसिंग पाइपलाइन",
        "dysarthric": "अस्पष्ट भाषण आढळले",
        "clear_speech": "स्पष्ट भाषण आढळले",
        "step1_msg": "🔬 पायरी 1: AI मॉडेलसह भाषणाचे विश्लेषण...",
        "step1_done": "✅ वर्गीकरण: {status} — विश्वास: {conf}%",
        "step2_msg": "📝 पायरी 2: {lang} मध्ये रूपांतरित करत आहे...",
        "step2_done": "✅ रूपांतरित: \"{text}\"",
        "step2_fail": "⚠️ भाषणाचे रूपांतर होऊ शकले नाही. कृपया जोरात बोला किंवा पुन्हा प्रयत्न करा.",
        "step3_msg": "🔊 पायरी 3: {lang} मध्ये स्पष्ट आणि मजबूत आवाज तयार करत आहे...",
        "step3_done": "✅ स्पष्ट आवाज तयार! स्पष्ट आवाज • चांगली गती • मजबूत ऑडिओ",
        "step3_clear": "🔊 पायरी 3: भाषण स्पष्ट आहे! सुधारित आवृत्ती तयार करत आहे...",
        "clear_ready": "स्पष्ट आवाज तयार",
        "enhanced_voice": "सुधारित • स्पष्ट • मजबूत आवाज",
        "transcription_label": "रूपांतरण ({lang})",
        "play_listener": "▶️ सुधारित स्पष्ट ऑडिओ — ऐकणाऱ्यासाठी वाजवा:",
        "clear_detected_title": "स्पष्ट आवाज आढळला!",
        "no_enhancement": "सुधारणेची गरज नाही — आवाज आधीच स्पष्ट आहे.",
        "how_it_works": "हे कसे कार्य करते",
        "how_step1": "रेकॉर्ड: वापरकर्ता मायक्रोफोनमध्ये बोलतो.",
        "how_step2": "वर्गीकरण: AI मॉडेल भाषणातील दोष ओळखते.",
        "how_step3": "रूपांतरण: स्पीच-टू-टेक्स्ट समजते की वापरकर्त्याला काय म्हणायचे आहे.",
        "how_step4": "स्पष्ट बोला: AI स्पष्ट आवाज तयार करते — तो ऐकणाऱ्यासाठी वाजवा.",
        "wa_preview": "▶️ स्पष्ट ऑडिओचे पूर्वावलोकन:",
        "download_clear": "⬇️ स्पष्ट ऑडिओ डाउनलोड करा",
        "share_wa": "📲 WhatsApp वर शेअर करा",
        "wa_tip": "💡 टीप: ऑडिओ फाइल डाउनलोड करा, मग चांगल्या अनुभवासाठी WhatsApp मध्ये जोडा.",
        "phone_mode_banner": "कॉल दरम्यान, रेकॉर्ड करा → AI स्पष्ट आवाज तयार करेल → पुन्हा वाजवा",
        "active_call": "सक्रिय कॉल मोड",
        "phone_instruction": "जेव्हा तुम्हाला काही बोलायचे असेल तेव्हा \"स्पष्ट बोला\" दाबा",
        "phone_tip_box": "AI तुमचा आवाज पकडेल आणि स्पष्ट आवृत्ती वाजवेल",
        "phone_bridge": "🔄 AI स्पीच ब्रिज प्रोसेसिंग",
        "play_near_mic": "🔊 स्पष्ट आवाज वाजवा — तुमचा फोन स्पीकरजवळ धरा:",
        "phone_tip": "💡 टीप: फोन स्पीकरवर ठेवा, मग हा ऑडिओ माईकजवळ वाजवा.",
        "ai_powered": "AI समर्थित",
        "integration": "एकीकरण",
        "voice_only": "फक्त आवाज"
    },
    "தமிழ் (Tamil)": {
        "title": "பாஷினி பிரிட்ஜ் (BhasiniBridge)",
        "choose_lang": "மொழியைத் தேர்ந்தெடுக்கவும்",
        "select_pref": "தொடங்க உங்களுக்கு விருப்பமான மொழியைத் தேர்ந்தெடுக்கவும்",
        "users": "பயனர்கள்",
        "languages": "மொழிகள்",
        "selected": "தேர்ந்தெடுக்கப்பட்டது",
        "latency": "0ms தாமதம்",
        "choose_mode": "முறையைத் தேர்ந்தெடுக்கவும்",
        "f2f_title": "நேருக்கு நேர் (Face-to-Face)",
        "f2f_desc": "பேசுங்கள் → AI தெளிவுபடுத்தும் → கேட்பவருக்குத் தெளிவான குரலை ஒலிக்கும்.",
        "f2f_btn": "நேருக்கு நேர் முறையைத் தொடங்கு",
        "wa_title": "WhatsApp & பகிர்வு",
        "wa_desc": "பதிவு செய் → தெளிவுபடுத்து → WhatsApp மூலம் தெளிவான ஆடியோவைப் பதிவிறக்கு அல்லது பகிர்.",
        "wa_btn": "WhatsApp முறையைத் தொடங்கு",
        "phone_title": "தொலைபேசி அழைப்புகள்",
        "phone_desc": "அழைப்பின் போது பதிவு செய் → AI கேட்பவருக்குத் தெளிவான குரலை உருவாக்கும்.",
        "phone_btn": "அழைப்பு முறையைத் தொடங்கு",
        "back_lang": "← மொழியை மாற்றவும்",
        "back_dash": "← முகப்புப்பக்கத்திற்குச் செல்",
        "tap_to_speak": "🎙️ பேசத் தொடங்கு",
        "speak_into_mic": "மைக்ரோஃபோனில் பேசுங்கள் → AI தெளிவான குரலாக மாற்றும்:",
        "processing_pipeline": "🔄 செயலாக்க வரிசை",
        "dysarthric": "தடைபட்ட பேச்சு கண்டறியப்பட்டது",
        "clear_speech": "தெளிவான பேச்சு கண்டறியப்பட்டது",
        "step1_msg": "🔬 படி 1: AI மாதிரி மூலம் பேச்சை ஆய்வு செய்தல்...",
        "step1_done": "✅ வகைப்பாடு: {status} — நம்பிக்கை: {conf}%",
        "step2_msg": "📝 படி 2: {lang} இல் மாற்றப்படுகிறது...",
        "step2_done": "✅ மாற்றப்பட்டது: \"{text}\"",
        "step2_fail": "⚠️ பேச்சை மாற்ற முடியவில்லை. தயவுசெய்து சத்தமாகப் பேசவும் அல்லது மீண்டும் முயற்சிக்கவும்.",
        "step3_msg": "🔊 படி 3: {lang} இல் தெளிவான குரல் உருவாக்கப்படுகிறது...",
        "step3_done": "✅ தெளிவான குரல் தயார்! தெளிவான குரல் • சிறந்த வேகம் • வலுவான ஆடியோ",
        "clear_ready": "தெளிவான குரல் தயார்",
        "enhanced_voice": "மேம்படுத்தப்பட்ட • தெளிவான • வலுவான குரல்",
        "transcription_label": "எழுத்தாக்கம் ({lang})",
        "play_listener": "▶️ மேம்படுத்தப்பட்ட ஆடியோ — கேட்பவருக்கு ஒலிக்கவும்:",
        "clear_detected_title": "தெளிவான பேச்சு!",
        "no_enhancement": "மாற்றம் தேவையில்லை — பேச்சு ஏற்கனவே தெளிவாக உள்ளது.",
        "how_it_works": "இது எவ்வாறு இயங்குகிறது",
        "how_step1": "பதிவு: பயனர் மைக்ரோஃபோனில் பேசுகிறார்.",
        "how_step2": "வகைப்பாடு: AI பேச்சின் தன்மையைக் கண்டறியும்.",
        "how_step3": "எழுத்தாக்கம்: பயனர் என்ன சொல்ல வருகிறார் என்பதை AI புரிந்து கொள்ளும்.",
        "how_step4": "தெளிவாகப் பேசு: AI தெளிவான குரலை உருவாக்கும் — அதை ஒலிக்கவும்.",
        "wa_preview": "▶️ தெளிவான ஆடியோ முன்னோட்டம்:",
        "download_clear": "⬇️ தெளிவான ஆடியோ பதிவிறக்கம்",
        "share_wa": "📲 WhatsApp இல் பகிர்",
        "active_call": "அழைப்பு முறை",
        "phone_instruction": "நீங்கள் எதாவது சொல்ல விரும்பினால் \"தெளிவாகப் பேசு\" என்பதை அழுத்தவும்",
        "phone_bridge": "🔄 AI குரல் பாலம் செயலாக்கம்",
        "play_near_mic": "🔊 தெளிவான பேச்சை ஒலிக்கவும் — போனை ஸ்பೀக்கர் அருகில் வைக்கவும்:",
        "ai_powered": "AI மூலம்",
        "integration": "ஒருங்கிணைப்பு",
        "voice_only": "குರல் மட்டும்"
    },
    "తెలుగు (Telugu)": {
        "title": "భాషిని బ్రిడ్జ్ (BhasiniBridge)",
        "choose_lang": "భాషను ఎంచుకోండి",
        "select_pref": "ప్రారంభించడానికి మీకు నచ్చిన భాషను ఎంచుకోండి",
        "users": "వినియోగదారులు",
        "languages": "భాషలు",
        "selected": "ఎంచుకోబడింది",
        "latency": "0ms ఆలస్యం",
        "choose_mode": "మోడ్ ఎంచుకోండి",
        "f2f_title": "ముఖాముఖి (Face-to-Face)",
        "f2f_desc": "మాట్లాడండి → AI స్పష్టం చేస్తుంది → వినేవారికి స్పష్టమైన స్వరాన్ని వినిపిస్తుంది.",
        "f2f_btn": "ముఖాముఖి మోడ్ ప్రారంభించండి",
        "wa_title": "WhatsApp & షేర్",
        "wa_desc": "రికార్డ్ చేయండి → స్పష్టం చేయండి → WhatsApp ద్వారా స్పష్టమైన ఆడియోను షేర్ చేయండి.",
        "wa_btn": "WhatsApp మోడ్ ప్రారంభించండి",
        "phone_title": "ఫోన్ కాల్స్",
        "phone_desc": "కాల్ సమయంలో రికార్డ్ చేయండి → AI స్పష్టమైన స్వరాన్ని రూపొందిస్తుంది.",
        "phone_btn": "ఫోన్ కాల్ మోడ్ ప్రారంభించండి",
        "back_lang": "← భాష మార్చండి",
        "back_dash": "← హోమ్ పేజీకి వెళ్ళండి",
        "tap_to_speak": "🎙️ మాట్లాడటానికి ట్యాప్ చేయండి",
        "speak_into_mic": "మైక్‌లో మాట్లాడండి → AI స్పష్టమైన స్వరంగా మారుస్తుంది:",
        "processing_pipeline": "🔄 ప్రాసెసింగ్ పైప్‌లైన్",
        "dysarthric": "అస్పష్టమైన మాటలు గుర్తించబడ్డాయి",
        "clear_speech": "స్పష్టమైన మాటలు గుర్తించబడ్డాయి",
        "step1_msg": "🔬 స్టెప్ 1: AI మోడల్‌తో మాటల విశ్లేషణ...",
        "step1_done": "✅ వర్గీకరణ: {status} — విశ్వాసం: {conf}%",
        "step2_msg": "📝 స్టెప్ 2: {lang}లోకి మారుస్తోంది...",
        "step2_done": "✅ మార్చబడింది: \"{text}\"",
        "step3_msg": "🔊 స్టెప్ 3: స్పష్టమైన స్వరాన్ని రూపొందిస్తోంది...",
        "clear_ready": "స్పష్టమైన స్వరం సిద్ధం",
        "enhanced_voice": "మెరుగుపరచబడిన • స్పష్టమైన స్వరం",
        "transcription_label": "అనువాదం ({lang})",
        "play_listener": "▶️ స్పష్టమైన ఆడియో — వినేవారి కోసం ప్లే చేయండి:",
        "how_it_works": "ఇది ఎలా పనిచేస్తుంది",
        "download_clear": "⬇️ ఆడియోను డౌన్‌లోడ్ చేయండి",
        "share_wa": "📲 WhatsApp లో షేర్ చేయండి",
        "active_call": "యాక్టివ్ కాల్ మోడ్",
        "ai_powered": "AI పవర్డ్",
        "integration": "ఇంటిగ్రేషన్",
        "voice_only": "వాయిస్ మాత్రమే"
    },
    "ಕನ್ನಡ (Kannada)": {
        "title": "ಭಾಷಿಣಿ ಬ್ರಿಡ್ಜ್ (BhasiniBridge)",
        "choose_lang": "ಭಾಷೆಯನ್ನು ಆರಿಸಿ",
        "select_pref": "ಪ್ರಾರಂಭಿಸಲು ನಿಮ್ಮ ಇಷ್ಟದ ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ",
        "users": "ಬಳಕೆದಾರರು",
        "languages": "ಭಾಷೆಗಳು",
        "selected": "ಆಯ್ಕೆ ಮಾಡಲಾಗಿದೆ",
        "latency": "0ms ವಿಳಂಬ",
        "choose_mode": "ಮೋಡ್ ಆಯ್ಕೆಮಾಡಿ",
        "f2f_title": "ಮುಖಾಮುಖಿ (Face-to-Face)",
        "f2f_desc": "ಮಾತನಾಡಿ → AI ಸ್ಪಷ್ಟಪಡಿಸುತ್ತದೆ → ಕೇಳುಗರಿಗೆ ಸ್ಪಷ್ಟ ಧ್ವನಿಯನ್ನು ಪ್ಲೇ ಮಾಡುತ್ತದೆ.",
        "f2f_btn": "ಮುಖಾಮುಖಿ ಮೋಡ್ ಪ್ರಾರಂಭಿಸಿ",
        "wa_title": "WhatsApp & ಶೇರ್",
        "wa_desc": "ರೆಕಾರ್ಡ್ ಮಾಡಿ → ಸ್ಪಷ್ಟಪಡಿಸಿ → WhatsApp ಮೂಲಕ ಆಡಿಯೊವನ್ನು ಶೇರ್ ಮಾಡಿ.",
        "wa_btn": "WhatsApp ಮೋಡ್ ಪ್ರಾರಂಭಿಸಿ",
        "phone_title": "ಫೋನ್ ಕರೆಗಳು",
        "phone_desc": "ಕರೆಯ ಸಮಯದಲ್ಲಿ ರೆಕಾರ್ಡ್ ಮಾಡಿ → AI ಸ್ಪಷ್ಟ ಧ್ವನಿಯನ್ನು ಸಿದ್ಧಪಡಿಸುತ್ತದೆ.",
        "phone_btn": "ಫೋನ್ ಕರೆ ಮೋಡ್ ಪ್ರಾರಂಭಿಸಿ",
        "back_lang": "← ಭಾಷೆ ಬದಲಾಯಿಸಿ",
        "back_dash": "← ಮುಖಪುಟಕ್ಕೆ ಹಿಂತಿರುಗಿ",
        "tap_to_speak": "🎙️ ಮಾತನಾಡಲು ಟ್ಯಾಪ್ ಮಾಡಿ",
        "speak_into_mic": "ಮೈಕ್‌ನಲ್ಲಿ ಮಾತನಾಡಿ → AI ಸ್ಪಷ್ಟ ಧ್ವನಿಗೆ ಬದಲಾಯಿಸುತ್ತದೆ:",
        "processing_pipeline": "🔄 ಪ್ರಕ್ರಿಯೆ ನಡೆಯುತ್ತಿದೆ",
        "dysarthric": "ಅಸ್ಪಷ್ಟ ಮಾತು ಪತ್ತೆಯಾಗಿದೆ",
        "clear_speech": "ಸ್ಪಷ್ಟ ಮಾತು ಪತ್ತೆಯಾಗಿದೆ",
        "step1_msg": "🔬 ಹಂತ 1: AI ಮಾದರಿಯಿಂದ ವಿಶ್ಲೇಷಣೆ...",
        "step1_done": "✅ ವರ್ಗೀಕರಣ: {status} — ವಿಶ್ವಾಸ: {conf}%",
        "step2_msg": "📝 ಹಂತ 2: {lang}ಗೆ ಪರಿವರ್ತಿಸಲಾಗುತ್ತಿದೆ...",
        "step2_done": "✅ ಪರಿವರ್ತಿಸಲಾಗಿದೆ: \"{text}\"",
        "step3_msg": "🔊 ಹಂತ 3: ಸ್ಪಷ್ಟ ಧ್ವನಿಯನ್ನು ಸಿದ್ಧಪಡಿಸಲಾಗುತ್ತಿದೆ...",
        "clear_ready": "ಸ್ಪಷ್ಟ ಧ್ವನಿ ಸಿದ್ಧವಾಗಿದೆ",
        "enhanced_voice": "ಸುಧಾರಿತ • ಸ್ಪಷ್ಟ ಧ್ವನಿ",
        "transcription_label": "ಅನುವಾದ ({lang})",
        "play_listener": "▶️ ಸ್ಪಷ್ಟ ಆಡಿಯೊ — ಕೇಳುಗರಿಗಾಗಿ ಪ್ಲೇ ಮಾಡಿ:",
        "how_it_works": "ಇದು ಹೇಗೆ ಕೆಲಸ ಮಾಡುತ್ತದೆ",
        "download_clear": "⬇️ ಆಡಿಯೊ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ",
        "share_wa": "📲 WhatsApp ಮೂಲಕ ಶೇರ್ ಮಾಡಿ",
        "active_call": "ಸಕ್ರಿಯ ಕರೆ ಮೋಡ್",
        "ai_powered": "AI ಶಕ್ತಿ",
        "integration": "ಸಂಯೋಜನೆ",
        "voice_only": "ಧ್ವನಿ ಮಾತ್ರ"
    }
}

def _t(key, **kwargs):
    """Helper function to get translated text."""
    lang = st.session_state.get('language', 'English')
    text = TRANSLATIONS.get(lang, TRANSLATIONS['English']).get(key, key)
    if kwargs:
        return text.format(**kwargs)
    return text


# Custom CSS for UI
st.markdown("""
<style>
/* Modern typography and spacing */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}

/* Base theme */
.stApp {
    background-color: #020617;
}

/* Reusable Glassmorphism Card Style */
.glass-card {
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 24px;
    padding: 24px;
    margin-bottom: 24px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    border: 1px solid rgba(59, 130, 246, 0.3);
}

/* Header style styling */
.main-header {
    background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.2rem;
    margin-bottom: 0.5rem;
    text-align: center;
    letter-spacing: -0.02em;
}

/* Stats Section */
.stats-container {
    display: flex;
    justify-content: space-between;
    background: rgba(30, 41, 59, 0.5);
    border-radius: 16px;
    padding: 16px 24px;
    margin-bottom: 32px;
}
.stat-item {
    text-align: center;
}
.stat-value {
    font-weight: 700;
    font-size: 1.25rem;
    color: #3b82f6;
}
.stat-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Mode Cards */
.mode-card {
    position: relative;
    background: linear-gradient(145deg, rgba(30,41,59,0.7), rgba(15,23,42,0.9));
    border-radius: 24px;
    padding: 24px;
    margin-bottom: 16px;
    border: 1px solid rgba(255,255,255,0.05);
    cursor: pointer;
    overflow: hidden;
}
.mode-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, #10b981, #3b82f6);
    opacity: 0;
    transition: opacity 0.3s ease;
}
.mode-card:hover::before {
    opacity: 1;
}

.mode-index {
    color: rgba(148, 163, 184, 0.3);
    font-size: 2.5rem;
    font-weight: 800;
    position: absolute;
    top: 16px;
    right: 24px;
}
.mode-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 8px;
}
.mode-desc {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-bottom: 20px;
    max-width: 80%;
}

/* Animations */
@keyframes pulse-ring {
  0% { transform: scale(0.8); box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7); }
  70% { transform: scale(1); box-shadow: 0 0 0 20px rgba(59, 130, 246, 0); }
  100% { transform: scale(0.8); box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
}

@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

.processing-shimmer {
    background: linear-gradient(90deg, rgba(59,130,246,0.1), rgba(59,130,246,0.3), rgba(59,130,246,0.1));
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
}

/* Result Card */
.result-success {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(2, 6, 23, 1));
    border: 1px solid rgba(16, 185, 129, 0.3);
}

/* Enhancement result card */
.enhancement-card {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(16, 185, 129, 0.08));
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 20px;
    padding: 24px;
    margin-top: 16px;
}

/* Pipeline step */
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 16px;
    margin: 6px 0;
    border-radius: 10px;
    background: rgba(30, 41, 59, 0.4);
    font-size: 0.9rem;
}
.pipeline-step.active {
    background: rgba(59, 130, 246, 0.15);
    border-left: 3px solid #3b82f6;
}
.pipeline-step.done {
    background: rgba(16, 185, 129, 0.1);
    border-left: 3px solid #10b981;
}

/* Custom st.button coloring to look like primary */
div.stButton > button:first-child {
    background: linear-gradient(90deg, #3b82f6, #2563eb);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
}
div.stButton > button:first-child:hover {
    box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.39);
    transform: translateY(-1px);
}

/* Green button variant for share */
.share-btn div.stButton > button:first-child {
    background: linear-gradient(90deg, #10b981, #059669);
}
.share-btn div.stButton > button:first-child:hover {
    box-shadow: 0 4px 14px 0 rgba(16, 185, 129, 0.39);
}

/* Purple button variant for phone */
.phone-btn div.stButton > button:first-child {
    background: linear-gradient(90deg, #a855f7, #7c3aed);
}
.phone-btn div.stButton > button:first-child:hover {
    box-shadow: 0 4px 14px 0 rgba(168, 85, 247, 0.39);
}

/* Audio player styling */
.stAudio > audio {
    border-radius: 12px;
    width: 100%;
}

/* Download link styling */
a.download-link {
    display: inline-block;
    background: linear-gradient(90deg, #10b981, #059669);
    color: white !important;
    padding: 10px 20px;
    border-radius: 12px;
    text-decoration: none;
    font-weight: 600;
    font-size: 0.9rem;
    transition: all 0.3s ease;
    text-align: center;
    width: 100%;
}
a.download-link:hover {
    box-shadow: 0 4px 14px 0 rgba(16, 185, 129, 0.39);
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# Application State Management
# ──────────────────────────────────────────────────────────────────
if 'app_state' not in st.session_state:
    st.session_state.app_state = 'language_selection'
if 'language' not in st.session_state:
    st.session_state.language = 'English'

# ──────────────────────────────────────────────────────────────────
# Core ML & Audio Processing Functions
# ──────────────────────────────────────────────────────────────────

@st.cache_resource
def load_ml_artifacts():
    """Load the trained dysarthria classification model and scaler."""
    try:
        scaler = joblib.load('artifacts/scaler.pkl')
        model = joblib.load('artifacts/dysarthria_model.pkl')
        return scaler, model
    except Exception as e:
        st.error(f"Error loading ML artifacts: {e}")
        return None, None

def extract_features_from_audio(audio_bytes, n_mfcc=40):
    """Extract MFCC features from audio bytes for classification."""
    try:
        y, sr_rate = librosa.load(io.BytesIO(audio_bytes), sr=16000, duration=2.5)
        mfcc = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def classify_speech(audio_bytes, scaler, model):
    """Classify audio as dysarthric or clear speech. Returns (is_dysarthric, confidence)."""
    features = extract_features_from_audio(audio_bytes)
    if features is None:
        return None, None
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    is_dysarthric = prediction == 1
    confidence = probability[1] if is_dysarthric else probability[0]
    return is_dysarthric, confidence

def transcribe_audio(audio_bytes, language_key):
    """Transcribe audio using Google Speech Recognition in the selected language."""
    lang_config = LANGUAGE_CONFIG.get(language_key, LANGUAGE_CONFIG["English"])
    stt_lang = lang_config["stt"]
    
    recognizer = sr.Recognizer()
    
    # Convert audio bytes to WAV format that SpeechRecognition can process
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        
        with sr.AudioFile(wav_buffer) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language=stt_lang)
            return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        st.error(f"Speech Recognition service error: {e}")
        return None
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

def generate_clear_speech(text, language_key):
    """Generate clear, strong TTS audio from text in the selected language. Returns audio bytes."""
    lang_config = LANGUAGE_CONFIG.get(language_key, LANGUAGE_CONFIG["English"])
    tts_lang = lang_config["tts"]
    
    try:
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.getvalue()
    except Exception as e:
        st.error(f"TTS generation error: {e}")
        return None

def get_audio_download_link(audio_bytes, filename="clear_speech.mp3"):
    """Generate a download link for audio bytes."""
    b64 = base64.b64encode(audio_bytes).decode()
    return f'<a class="download-link" href="data:audio/mp3;base64,{b64}" download="{filename}">⬇️ Download Clear Audio</a>'

def get_whatsapp_share_link(text):
    """Generate a WhatsApp share link with the transcribed text."""
    import urllib.parse
    encoded_text = urllib.parse.quote(text)
    return f"https://api.whatsapp.com/send?text={encoded_text}"

# ──────────────────────────────────────────────────────────────────
# Full Speech Enhancement Pipeline (shared across all 3 modes)
# ──────────────────────────────────────────────────────────────────

def run_enhancement_pipeline(audio_bytes, language_key, mode="face_to_face"):
    """
    Full pipeline:
    1. Classify speech (dysarthric vs clear)
    2. Transcribe slurred speech (STT)
    3. Re-synthesize as clear, strong speech (TTS)
    Returns dict with all results.
    """
    scaler, model = load_ml_artifacts()
    if scaler is None or model is None:
        st.error("⚠️ ML Artifacts not found! Run main.ipynb first to generate scaler.pkl and dysarthria_model.pkl.")
        return None
    
    result = {}
    lang_config = LANGUAGE_CONFIG.get(language_key, LANGUAGE_CONFIG["English"])
    
    # ── Step 1: Classification ──
    st.markdown(f"""
    <div class="pipeline-step active">
        <span>🔬</span> <strong>{_t("step1_msg")}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    is_dysarthric, confidence = classify_speech(audio_bytes, scaler, model)
    if is_dysarthric is None:
        st.error("Could not process audio. Please try again.")
        return None
    
    result['is_dysarthric'] = is_dysarthric
    result['confidence'] = confidence
    
    status_text = _t("dysarthric") if is_dysarthric else _t("clear_speech")
    st.markdown(f"""
    <div class="pipeline-step done">
        <span>✅</span> <strong>{_t("step1_done", status=status_text, conf=int(confidence*100))}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Step 2: Transcription (STT) ──
    st.markdown(f"""
    <div class="pipeline-step active">
        <span>📝</span> <strong>{_t("step2_msg", lang=lang_config['label'])}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    transcribed_text = transcribe_audio(audio_bytes, language_key)
    
    if transcribed_text:
        result['transcribed_text'] = transcribed_text
        st.markdown(f"""
        <div class="pipeline-step done">
            <span>✅</span> <strong>{_t("step2_done", text=transcribed_text)}</strong>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="pipeline-step" style="border-left: 3px solid #ef4444; background: rgba(239,68,68,0.1);">
            <span>⚠️</span> <strong>{_t("step2_fail")}</strong>
        </div>
        """, unsafe_allow_html=True)
        result['transcribed_text'] = None
        return result
    
    # ── Step 3: Clear Speech Generation (TTS) ──
    if is_dysarthric and transcribed_text:
        st.markdown(f"""
        <div class="pipeline-step active">
            <span>🔊</span> <strong>{_t("step3_msg", lang=lang_config['label'])}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        clear_audio = generate_clear_speech(transcribed_text, language_key)
        
        if clear_audio:
            result['clear_audio'] = clear_audio
            st.markdown(f"""
            <div class="pipeline-step done">
                <span>✅</span> <strong>{_t("step3_done")}</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pipeline-step" style="border-left: 3px solid #ef4444;">
                <span>❌</span> <strong>{_t("step3_fail")}</strong>
            </div>
            """, unsafe_allow_html=True)
    elif not is_dysarthric:
        # Even if clear speech, still offer TTS enhancement
        st.markdown(f"""
        <div class="pipeline-step active">
            <span>🔊</span> <strong>{_t("step3_clear")}</strong>
        </div>
        """, unsafe_allow_html=True)
        clear_audio = generate_clear_speech(transcribed_text, language_key)
        if clear_audio:
            result['clear_audio'] = clear_audio
            st.markdown(f"""
            <div class="pipeline-step done">
                <span>✅</span> <strong>{_t("step3_msg", lang=lang_config['label'])}</strong>
            </div>
            """, unsafe_allow_html=True)

    
    return result

# ──────────────────────────────────────────────────────────────────
# UI Rendering Functions
# ──────────────────────────────────────────────────────────────────

def render_language_selection():
    st.markdown(f'<div class="main-header" style="margin-top: 50px;">{_t("choose_lang")}</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align:center; color:#94a3b8; font-size:1.1rem; margin-bottom: 40px;">{_t("select_pref")}</p>', unsafe_allow_html=True)
    
    languages = list(LANGUAGE_CONFIG.keys())
    
    cols = st.columns(2)
    for i, lang in enumerate(languages):
        with cols[i % 2]:
            if st.button(lang, use_container_width=True, key=f"lang_{i}"):
                st.session_state.language = lang
                st.session_state.app_state = 'home'
                st.rerun()


def render_home():
    st.markdown(f'<div class="main-header">{_t("title")}</div>', unsafe_allow_html=True)
    
    lang_label = LANGUAGE_CONFIG.get(st.session_state.language, {}).get('label', 'English')
    
    st.markdown(f"""
    <div class="stats-container">
        <div class="stat-item">
            <div class="stat-value">4M+</div>
            <div class="stat-label">{_t("users")}</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">6</div>
            <div class="stat-label">{_t("languages")}</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{lang_label}</div>
            <div class="stat-label">{_t("selected")}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f'<h3 style="color:white; margin-bottom: 16px; font-weight: 600;">{_t("choose_mode")}</h3>', unsafe_allow_html=True)
    
    # Mode 1: Face-to-Face
    st.markdown(f"""
    <div class="mode-card">
        <div class="mode-index">01</div>
        <span style="background: rgba(59, 130, 246, 0.2); color: #3b82f6; padding: 4px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; margin-bottom: 12px; display: inline-block;">{_t("ai_powered")}</span>
        <div class="mode-title">{_t("f2f_title")}</div>
        <div class="mode-desc">{_t("f2f_desc")}</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button(_t("f2f_btn"), key="f2f", use_container_width=True):
        st.session_state.app_state = 'face_to_face'
        st.rerun()

    # Mode 2: WhatsApp / Share
    st.markdown(f"""
    <div class="mode-card" style="margin-top:20px;">
        <div class="mode-index">02</div>
        <span style="background: rgba(16, 185, 129, 0.2); color: #10b981; padding: 4px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; margin-bottom: 12px; display: inline-block;">{_t("integration")}</span>
        <div class="mode-title">{_t("wa_title")}</div>
        <div class="mode-desc">{_t("wa_desc")}</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button(_t("wa_btn"), key="wa", use_container_width=True):
        st.session_state.app_state = 'whatsapp_mode'
        st.rerun()

    # Mode 3: Phone Call
    st.markdown(f"""
    <div class="mode-card" style="margin-top:20px;">
        <div class="mode-index">03</div>
        <span style="background: rgba(168, 85, 247, 0.2); color: #a855f7; padding: 4px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; margin-bottom: 12px; display: inline-block;">{_t("voice_only")}</span>
        <div class="mode-title">{_t("phone_title")}</div>
        <div class="mode-desc">{_t("phone_desc")}</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button(_t("phone_btn"), key="phone", use_container_width=True):
        st.session_state.app_state = 'phone_mode'
        st.rerun()

    if st.button(_t("back_lang"), key="back_lang"):
        st.session_state.app_state = 'language_selection'
        st.rerun()



# ──────────────────────────────────────────────────────────────────
# MODE 1: Face-to-Face
# ──────────────────────────────────────────────────────────────────
def render_face_to_face():
    if st.button(_t("back_dash")):
        st.session_state.app_state = 'home'
        st.rerun()
    
    lang_label = LANGUAGE_CONFIG.get(st.session_state.language, {}).get('label', 'English')
    
    st.markdown(f'<div class="main-header" style="font-size: 1.8rem; margin-top:20px;">🗣️ {_t("f2f_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align:center; color:#94a3b8; font-size:0.95rem;">{_t("speak_into_mic")} <strong style="color:#10b981;">{lang_label}</strong></p>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Audio input
    audio_value = st.audio_input(_t("tap_to_speak"), key="f2f_audio")
    
    if audio_value is not None:
        audio_bytes = audio_value.getvalue()
        
        st.markdown("---")
        st.markdown(f'<p style="color: #94a3b8; font-size:0.8rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px;">{_t("processing_pipeline")}</p>', unsafe_allow_html=True)
        
        with st.spinner(""):
            result = run_enhancement_pipeline(audio_bytes, st.session_state.language, mode="face_to_face")
        
        if result and result.get('clear_audio'):
            st.markdown("---")
            
            # Show enhancement results
            st.markdown(f"""
            <div class="enhancement-card">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                    <div style="background: linear-gradient(135deg, #3b82f6, #10b981); border-radius: 50%; width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; flex-shrink:0;">🔊</div>
                    <div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">{_t("clear_ready")}</div>
                        <div style="color: #94a3b8; font-size: 0.85rem;">{_t("enhanced_voice")}</div>
                    </div>
                </div>
                <div style="background: rgba(2, 6, 23, 0.5); padding: 16px; border-radius: 12px; margin-bottom: 16px; border-left: 4px solid #10b981;">
                    <span style="display:block; font-size: 0.75rem; color:#94a3b8; margin-bottom: 4px; text-transform: uppercase;">{_t("transcription_label", lang=lang_label)}</span>
                    <span style="font-size: 1.1rem; color: #f8fafc; font-weight: 500;">"{result['transcribed_text']}"</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Play the clear audio
            st.markdown(f'<p style="color: #10b981; font-weight: 600; font-size: 0.85rem; margin-top: 16px;">{_t("play_listener")}</p>', unsafe_allow_html=True)
            st.audio(result['clear_audio'], format="audio/mp3", autoplay=True)
            
        elif result and result.get('transcribed_text') and not result.get('is_dysarthric'):
            st.markdown("---")
            st.markdown(f"""
            <div class="enhancement-card" style="border-color: rgba(16, 185, 129, 0.3);">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                    <span style="font-size: 1.5rem;">✅</span>
                    <div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #10b981;">{_t("clear_detected_title")}</div>
                        <div style="color: #94a3b8; font-size: 0.85rem;">{_t("no_enhancement")}</div>
                    </div>
                </div>
                <div style="background: rgba(2, 6, 23, 0.5); padding: 16px; border-radius: 12px; border-left: 4px solid #10b981;">
                    <span style="display:block; font-size: 0.75rem; color:#94a3b8; margin-bottom: 4px; text-transform: uppercase;">{_t("transcription_label", lang=lang_label)}</span>
                    <span style="font-size: 1.05rem; color: #f8fafc;">"{result['transcribed_text']}"</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if result.get('clear_audio'):
                st.audio(result['clear_audio'], format="audio/mp3")
    else:
        # Instructions
        st.markdown(f"""
        <div style="margin-top: 40px;">
            <p style="color:#94a3b8; font-weight:600; font-size: 0.85rem; margin-bottom: 12px; text-transform:uppercase; letter-spacing: 0.05em;">{_t("how_it_works")}</p>
            <div style="display:flex; align-items:flex-start; margin-bottom: 16px;">
                <div style="background:#3b82f6; color:white; border-radius:50%; width: 28px; height: 28px; display:flex; align-items:center; justify-content:center; font-size: 0.8rem; font-weight:bold; margin-right: 12px; flex-shrink: 0;">1</div>
                <div style="color:#cbd5e1; font-size:0.9rem;"><strong>{_t("how_step1")}</strong></div>
            </div>
            <div style="display:flex; align-items:flex-start; margin-bottom: 16px;">
                <div style="background:#3b82f6; color:white; border-radius:50%; width: 28px; height: 28px; display:flex; align-items:center; justify-content:center; font-size: 0.8rem; font-weight:bold; margin-right: 12px; flex-shrink: 0;">2</div>
                <div style="color:#cbd5e1; font-size:0.9rem;"><strong>{_t("how_step2")}</strong></div>
            </div>
            <div style="display:flex; align-items:flex-start; margin-bottom: 16px;">
                <div style="background:#10b981; color:white; border-radius:50%; width: 28px; height: 28px; display:flex; align-items:center; justify-content:center; font-size: 0.8rem; font-weight:bold; margin-right: 12px; flex-shrink: 0;">3</div>
                <div style="color:#cbd5e1; font-size:0.9rem;"><strong>{_t("how_step3")}</strong></div>
            </div>
            <div style="display:flex; align-items:flex-start;">
                <div style="background:#10b981; color:white; border-radius:50%; width: 28px; height: 28px; display:flex; align-items:center; justify-content:center; font-size: 0.8rem; font-weight:bold; margin-right: 12px; flex-shrink: 0;">4</div>
                <div style="color:#cbd5e1; font-size:0.9rem;"><strong>{_t("how_step4")}</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)



# ──────────────────────────────────────────────────────────────────
# MODE 2: WhatsApp & Share
# ──────────────────────────────────────────────────────────────────
def render_whatsapp_mode():
    if st.button(_t("back_dash"), key="back_wa"):
        st.session_state.app_state = 'home'
        st.rerun()
    
    lang_label = LANGUAGE_CONFIG.get(st.session_state.language, {}).get('label', 'English')
    
    st.markdown(f'<div class="main-header" style="font-size: 1.8rem; margin-top:20px;">💬 {_t("wa_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align:center; color:#94a3b8; font-size:0.95rem;">{_t("wa_desc")} <strong style="color:#10b981;">{lang_label}</strong></p>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    audio_value = st.audio_input(_t("tap_to_speak"), key="wa_audio")
    
    if audio_value is not None:
        audio_bytes = audio_value.getvalue()
        
        st.markdown("---")
        st.markdown(f'<p style="color: #94a3b8; font-size:0.8rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px;">{_t("processing_pipeline")}</p>', unsafe_allow_html=True)
        
        with st.spinner(""):
            result = run_enhancement_pipeline(audio_bytes, st.session_state.language, mode="whatsapp")
        
        if result and result.get('clear_audio') and result.get('transcribed_text'):
            st.markdown("---")
            
            st.markdown(f"""
            <div class="enhancement-card" style="border-color: rgba(16, 185, 129, 0.3);">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                    <div style="background: linear-gradient(135deg, #10b981, #059669); border-radius: 50%; width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; flex-shrink:0;">💬</div>
                    <div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">{_t("clear_ready")}</div>
                        <div style="color: #94a3b8; font-size: 0.85rem;">{_t("enhanced_voice")}</div>
                    </div>
                </div>
                <div style="background: rgba(2, 6, 23, 0.5); padding: 16px; border-radius: 12px; margin-bottom: 16px; border-left: 4px solid #10b981;">
                    <span style="display:block; font-size: 0.75rem; color:#94a3b8; margin-bottom: 4px; text-transform: uppercase;">{_t("transcription_label", lang=lang_label)}</span>
                    <span style="font-size: 1.1rem; color: #f8fafc; font-weight: 500;">"{result['transcribed_text']}"</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Play clear audio
            st.markdown(f'<p style="color: #10b981; font-weight: 600; font-size: 0.85rem; margin-top: 16px;">{_t("wa_preview")}</p>', unsafe_allow_html=True)
            st.audio(result['clear_audio'], format="audio/mp3")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Download + Share buttons
            col1, col2 = st.columns(2)
            with col1:
                # Download clear audio
                st.markdown(get_audio_download_link(result['clear_audio'], "bhasini_clear_message.mp3"), unsafe_allow_html=True)
            
            with col2:
                # Share text to WhatsApp
                wa_link = get_whatsapp_share_link(f"🗣️ {_t('title')} Message:\n\n\"{result['transcribed_text']}\"")
                st.markdown(f'<a class="download-link" href="{wa_link}" target="_blank" style="background: linear-gradient(90deg, #25D366, #128C7E);">{_t("share_wa")}</a>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<p style="color: #64748b; font-size: 0.8rem; text-align: center;">{_t("wa_tip")}</p>', unsafe_allow_html=True)
    
    else:
        st.markdown(f"""
        <div style="margin-top: 40px;">
            <p style="color:#94a3b8; font-weight:600; font-size: 0.85rem; margin-bottom: 12px; text-transform:uppercase; letter-spacing: 0.05em;">{_t("how_it_works")}</p>
            <div style="display:flex; align-items:flex-start; margin-bottom: 16px;">
                <div style="background:#10b981; color:white; border-radius:50%; width: 28px; height: 28px; display:flex; align-items:center; justify-content:center; font-size: 0.8rem; font-weight:bold; margin-right: 12px; flex-shrink: 0;">1</div>
                <div style="color:#cbd5e1; font-size:0.9rem;"><strong>{_t("how_step1")}</strong></div>
            </div>
            <div style="display:flex; align-items:flex-start; margin-bottom: 16px;">
                <div style="background:#10b981; color:white; border-radius:50%; width: 28px; height: 28px; display:flex; align-items:center; justify-content:center; font-size: 0.8rem; font-weight:bold; margin-right: 12px; flex-shrink: 0;">2</div>
                <div style="color:#cbd5e1; font-size:0.9rem;"><strong>{_t("how_step2")}</strong></div>
            </div>
            <div style="display:flex; align-items:flex-start; margin-bottom: 16px;">
                <div style="background:#10b981; color:white; border-radius:50%; width: 28px; height: 28px; display:flex; align-items:center; justify-content:center; font-size: 0.8rem; font-weight:bold; margin-right: 12px; flex-shrink: 0;">3</div>
                <div style="color:#cbd5e1; font-size:0.9rem;"><strong>{_t("how_step1")}</strong></div>
            </div>
            <div style="display:flex; align-items:flex-start;">
                <div style="background:#25D366; color:white; border-radius:50%; width: 28px; height: 28px; display:flex; align-items:center; justify-content:center; font-size: 0.8rem; font-weight:bold; margin-right: 12px; flex-shrink: 0;">4</div>
                <div style="color:#cbd5e1; font-size:0.9rem;"><strong>{_t("how_step4")}</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)



# ──────────────────────────────────────────────────────────────────
# MODE 3: Phone Calls
# ──────────────────────────────────────────────────────────────────
def render_phone_mode():
    if st.button(_t("back_dash"), key="back_phone"):
        st.session_state.app_state = 'home'
        st.rerun()
    
    lang_label = LANGUAGE_CONFIG.get(st.session_state.language, {}).get('label', 'English')
    
    st.markdown(f'<div class="main-header" style="font-size: 1.8rem; margin-top:20px;">📞 {_t("phone_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align:center; color:#94a3b8; font-size:0.95rem;">{_t("phone_mode_banner")} <strong style="color:#a855f7;">{lang_label}</strong></p>', unsafe_allow_html=True)
    
    # Phone call simulation UI
    st.markdown(f"""
    <div style="background: linear-gradient(145deg, rgba(168, 85, 247, 0.1), rgba(15, 23, 42, 0.8)); border: 1px solid rgba(168, 85, 247, 0.2); border-radius: 20px; padding: 20px; margin: 20px 0; text-align: center;">
        <div style="font-size: 0.75rem; color: #a855f7; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">{_t("active_call")}</div>
        <div style="font-size: 1.1rem; color: #f8fafc; font-weight: 600;">{_t("phone_instruction")}</div>
        <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 8px;">{_t("phone_tip_box")}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    audio_value = st.audio_input(_t("tap_to_speak"), key="phone_audio")
    
    if audio_value is not None:
        audio_bytes = audio_value.getvalue()
        
        st.markdown("---")
        st.markdown(f'<p style="color: #a855f7; font-size:0.8rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px;">{_t("phone_bridge")}</p>', unsafe_allow_html=True)
        
        with st.spinner(""):
            result = run_enhancement_pipeline(audio_bytes, st.session_state.language, mode="phone")
        
        if result and result.get('clear_audio') and result.get('transcribed_text'):
            st.markdown("---")
            
            st.markdown(f"""
            <div class="enhancement-card" style="border-color: rgba(168, 85, 247, 0.3);">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                    <div style="background: linear-gradient(135deg, #a855f7, #7c3aed); border-radius: 50%; width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; flex-shrink:0;">📞</div>
                    <div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">{_t("clear_ready")}</div>
                        <div style="color: #94a3b8; font-size: 0.85rem;">{_t("play_near_mic")}</div>
                    </div>
                </div>
                <div style="background: rgba(2, 6, 23, 0.5); padding: 16px; border-radius: 12px; margin-bottom: 16px; border-left: 4px solid #a855f7;">
                    <span style="display:block; font-size: 0.75rem; color:#94a3b8; margin-bottom: 4px; text-transform: uppercase;">{_t("transcription_label", lang=lang_label)}</span>
                    <span style="font-size: 1.1rem; color: #f8fafc; font-weight: 500;">"{result['transcribed_text']}"</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Play clear audio — the user holds the phone to the speaker
            st.markdown(f'<p style="color: #a855f7; font-weight: 600; font-size: 0.85rem; margin-top: 16px;">{_t("play_near_mic")}</p>', unsafe_allow_html=True)
            st.audio(result['clear_audio'], format="audio/mp3", autoplay=True)
            
            st.markdown(f'<p style="color: #64748b; font-size: 0.8rem; text-align: center; margin-top: 12px;">{_t("phone_tip")}</p>', unsafe_allow_html=True)
    
    else:
        st.markdown(f"""
        <div style="margin-top: 40px;">
            <p style="color:#94a3b8; font-weight:600; font-size: 0.85rem; margin-bottom: 12px; text-transform:uppercase; letter-spacing: 0.05em;">{_t("how_it_works")}</p>
            <div style="display:flex; align-items:flex-start; margin-bottom: 16px;">
                <div style="background:#a855f7; color:white; border-radius:50%; width: 28px; height: 28px; display:flex; align-items:center; justify-content:center; font-size: 0.8rem; font-weight:bold; margin-right: 12px; flex-shrink: 0;">1</div>
                <div style="color:#cbd5e1; font-size:0.9rem;"><strong>{_t("how_step1")}</strong></div>
            </div>
            <div style="display:flex; align-items:flex-start; margin-bottom: 16px;">
                <div style="background:#a855f7; color:white; border-radius:50%; width: 28px; height: 28px; display:flex; align-items:center; justify-content:center; font-size: 0.8rem; font-weight:bold; margin-right: 12px; flex-shrink: 0;">2</div>
                <div style="color:#cbd5e1; font-size:0.9rem;"><strong>{_t("how_step2")}</strong></div>
            </div>
            <div style="display:flex; align-items:flex-start; margin-bottom: 16px;">
                <div style="background:#a855f7; color:white; border-radius:50%; width: 28px; height: 28px; display:flex; align-items:center; justify-content:center; font-size: 0.8rem; font-weight:bold; margin-right: 12px; flex-shrink: 0;">3</div>
                <div style="color:#cbd5e1; font-size:0.9rem;"><strong>{_t("how_step3")}</strong></div>
            </div>
            <div style="display:flex; align-items:flex-start;">
                <div style="background:#a855f7; color:white; border-radius:50%; width: 28px; height: 28px; display:flex; align-items:center; justify-content:center; font-size: 0.8rem; font-weight:bold; margin-right: 12px; flex-shrink: 0;">4</div>
                <div style="color:#cbd5e1; font-size:0.9rem;"><strong>{_t("how_step4")}</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)



# ──────────────────────────────────────────────────────────────────
# Main Router
# ──────────────────────────────────────────────────────────────────
if st.session_state.app_state == 'language_selection':
    render_language_selection()
elif st.session_state.app_state == 'home':
    render_home()
elif st.session_state.app_state == 'face_to_face':
    render_face_to_face()
elif st.session_state.app_state == 'whatsapp_mode':
    render_whatsapp_mode()
elif st.session_state.app_state == 'phone_mode':
    render_phone_mode()
