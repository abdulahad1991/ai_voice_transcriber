from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from semantic_matcher import get_best_intent, extract_entities, update_conversation_state, suggest_similar_intents
from translate_utils import translate_to_english
import whisper
import os
import certifi
from reply_template import REPLY_TEMPLATES
os.environ['SSL_CERT_FILE'] = certifi.where()

# Add ffmpeg path manually for whisper to use it
ffmpeg_path = r"C:\Users\raqib\Downloads\ffmpeg-2025-06-28-git-cfd1f81e7d-essentials_build\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_path

app = FastAPI()
os.environ["HF_HOME"] = "C:/Users/raqib/.cache/huggingface"
model = whisper.load_model("medium")  

conversation_state = {}
conversation_history = {}

def get_language_code(detected_lang):
    if detected_lang in ["ur", "hi"]:
        return "ur"
    return "en"

def get_dual_language_message(intent, entities, state, detected_lang, clarification=None):
    msgs = {}
    for code in ['en', 'ur']:
        if clarification:
            msgs[code] = clarification[code] if isinstance(clarification, dict) else clarification
        else:
            msgs[code] = update_conversation_state('demo-user', intent, entities, state, code)
    return msgs

def build_success(intent, entities, lang):
    # Multi-step
    if intent == "REQUEST_FROM_PERSON" and "success" in REPLY_TEMPLATES[intent]:
        person = (entities.get("person_names") or ["someone"])[0]
        amount = (entities.get("amounts") or ["some amount"])[0]
        return REPLY_TEMPLATES[intent]["success"][lang].format(person=person, amount=amount)
    # All single-shot commands:
    elif intent in REPLY_TEMPLATES and "success" in REPLY_TEMPLATES[intent]:
        return REPLY_TEMPLATES[intent]["success"][lang]
    return f"✅ Detected intent: {intent}. Entities: {entities}"

@app.post("/voice-intent")
async def voice_intent_handler(file: UploadFile = File(...)):
    audio = await file.read()
    with open("temp.wav", "wb") as f:
        f.write(audio)
    result = model.transcribe("temp.wav", task="translate")

    original_text = result["text"]
    detected_lang = result.get("language", "unknown")

    # Always produce both translations
    urdu_translation = translate_to_english(original_text) if detected_lang in ["ur", "hi"] else original_text
    english_translation = original_text if detected_lang == "en" else urdu_translation

    user_id = "demo-user"
    user_context = conversation_state.get(user_id, {})
    last_intent = user_context.get("intent")

    # Extract intent/entities for both versions
    intent_en = get_best_intent(english_translation)
    intent_ur = get_best_intent(urdu_translation)
    entities_en = extract_entities(english_translation)
    entities_ur = extract_entities(urdu_translation)

    # Prefer English, fallback to Urdu if UNKNOWN
    intent = intent_en if intent_en != "UNKNOWN" else intent_ur
    entities = entities_en if intent_en != "UNKNOWN" else entities_ur

    history = conversation_history.get(user_id, [])
    history.append(english_translation)
    conversation_history[user_id] = history[-3:]

    # Multi-step flows
    if intent == "UNKNOWN" and last_intent:
        if last_intent == "LOGIN":
            if "phone_number" in entities:
                intent = "COMPLETE_LOGIN_STEP_1"
            elif "password" in entities:
                intent = "COMPLETE_LOGIN_STEP_2"
        elif last_intent == "REQUEST_FROM_PERSON":
            if "person_names" in entities and "amounts" in entities:
                intent = "COMPLETE_REQUEST_STEP"
        elif last_intent == "SIGNUP_HELP":
            if "phone_number" in entities and "password" in entities:
                intent = "COMPLETE_SIGNUP"

    confirm_required = intent in ["LOGOUT", "SEND_MONEY", "DELETE_ACCOUNT"]
    suggested_commands = []

    # --------- Dual-language message logic ---------

    def build_success(intent, entities, lang):
        # Multi-step entity formatting
        if intent == "REQUEST_FROM_PERSON" and "success" in REPLY_TEMPLATES[intent]:
            person = (entities.get("person_names") or ["someone"])[0]
            amount = (entities.get("amounts") or ["some amount"])[0]
            return REPLY_TEMPLATES[intent]["success"][lang].format(person=person, amount=amount)
        elif intent in REPLY_TEMPLATES and "success" in REPLY_TEMPLATES[intent]:
            return REPLY_TEMPLATES[intent]["success"][lang]
        return f"✅ Detected intent: {intent}. Entities: {entities}"

    # ENGLISH LOGIC
    clarification_en = None
    if intent == "REQUEST_FROM_PERSON":
        missing_en = []
        if not entities_en.get("person_names"):
            missing_en.append("the name of the person")
        if not entities_en.get("amounts"):
            missing_en.append("the amount")
        if missing_en:
            missing_str_en = ", and ".join(missing_en)
            clarification_en = REPLY_TEMPLATES["REQUEST_FROM_PERSON"]["clarification"]["en"].format(
                missing=missing_str_en or "required details"
            )
    elif intent == "LOGIN":
        if not entities_en.get("phone_number"):
            clarification_en = REPLY_TEMPLATES["LOGIN"]["clarification"]["en"]
    elif intent == "SIGNUP_HELP":
        if not (entities_en.get("phone_number") and entities_en.get("password")):
            clarification_en = REPLY_TEMPLATES["SIGNUP_HELP"]["clarification"]["en"]
    elif intent == "LOGOUT":
        clarification_en = REPLY_TEMPLATES["LOGOUT"]["confirm"]["en"]
    elif intent == "UNKNOWN":
        suggested_commands = suggest_similar_intents(english_translation)
        examples_en = []
        for sc in suggested_commands:
            phrases = COMMANDS.get(sc, [])
            for ex in phrases:
                if all(ord(ch) < 128 for ch in ex):
                    examples_en.append(ex)
                if len(examples_en) >= 2:
                    break
            if len(examples_en) >= 2:
                break
        if not examples_en:
            clarification_en = (
                "Sorry, I couldn't understand your command. "
                "You can try: 'Show my balance', 'Show my QR code', or 'Request money from Ali'."
            )
        else:
            clarification_en = REPLY_TEMPLATES["UNKNOWN"]["clarification"]["en"].format(
                examples="; ".join(f"'{ex}'" for ex in examples_en[:2])
            )
    # URDU LOGIC
    clarification_ur = None
    if intent == "REQUEST_FROM_PERSON":
        missing_ur = []
        if not entities_ur.get("person_names"):
            missing_ur.append("جس سے پیسے مانگنے ہیں اس کا نام")
        if not entities_ur.get("amounts"):
            missing_ur.append("کتنے پیسے")
        if missing_ur:
            missing_str_ur = " اور ".join(missing_ur)
            clarification_ur = REPLY_TEMPLATES["REQUEST_FROM_PERSON"]["clarification"]["ur"].format(
                missing=missing_str_ur or "درکار معلومات"
            )
    elif intent == "LOGIN":
        if not entities_ur.get("phone_number"):
            clarification_ur = REPLY_TEMPLATES["LOGIN"]["clarification"]["ur"]
    elif intent == "SIGNUP_HELP":
        if not (entities_ur.get("phone_number") and entities_ur.get("password")):
            clarification_ur = REPLY_TEMPLATES["SIGNUP_HELP"]["clarification"]["ur"]
    elif intent == "LOGOUT":
        clarification_ur = REPLY_TEMPLATES["LOGOUT"]["confirm"]["ur"]
    elif intent == "UNKNOWN":
        # Suggest Urdu commands
        examples_ur = []
        for sc in suggested_commands:
            phrases = COMMANDS.get(sc, [])
            for ex in phrases:
                if any(ord(ch) > 128 for ch in ex) or any(w in ex.lower() for w in ["dikhao", "paise", "krdo", "hai", "kahan", "mje", "ki", "se"]):
                    examples_ur.append(ex)
                if len(examples_ur) >= 2:
                    break
            if len(examples_ur) >= 2:
                break
        if not examples_ur:
            clarification_ur = (
                "معذرت، میں سمجھ نہیں سکا۔ آپ یہ کہہ سکتے ہیں: 'میرا بیلنس دکھاؤ'، 'میرا کیو آر کوڈ دکھاؤ'، یا 'علی سے پیسے مانگو'۔"
            )
        else:
            clarification_ur = REPLY_TEMPLATES["UNKNOWN"]["clarification"]["ur"].format(
                examples="؛ ".join(f"'{ex}'" for ex in examples_ur[:2])
            )

    # Construct the final messages
    messages = {
        "en": clarification_en or build_success(intent, entities_en, "en"),
        "ur": clarification_ur or build_success(intent, entities_ur, "ur")
    }

    return {
        "transcription": original_text,
        "language": detected_lang,
        "translated": english_translation,
        "intent": intent,
        "entities": entities,
        "context_from": last_intent,
        "messages": messages,
        "confirm_required": confirm_required,
        "suggested_commands": suggested_commands,
        "recent_memory": conversation_history[user_id]
    }


# Add this code to run the server directly with a custom port
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8282)
