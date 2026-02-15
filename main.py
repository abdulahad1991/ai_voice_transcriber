from command_db import COMMANDS
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from semantic_matcher import (
    get_best_intent, extract_entities, extract_entities_with_context,
    update_conversation_state, suggest_similar_intents,
    extract_partial_intent_clues
)
from translate_utils import translate_to_english
from conversation_manager import ConversationalManager
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

# Enhanced conversation management
conv_manager = ConversationalManager()
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
    
    # Get enhanced conversation context
    conv_context = conv_manager.get_or_create_context(user_id)
    conversation_turns = conv_context.get_recent_context(3)
    
    # Check for contextual references first ("that", "it", "same", etc.)
    contextual_intent = conv_manager.handle_contextual_reference(english_translation, conv_context)
    if contextual_intent:
        intent = contextual_intent
        entities = extract_entities_with_context(english_translation, conversation_turns)
    else:
        # Check if user is selecting from previous options
        selected_intent = conv_manager.handle_option_selection(english_translation, conv_context)
        if selected_intent:
            intent = selected_intent
            entities = extract_entities_with_context(english_translation, conversation_turns)
        else:
            # Extract intent/entities for both versions with context
            intent_en = get_best_intent(
                english_translation, 
                context_intent=last_intent,
                conversation_history=conversation_turns
            )
            intent_ur = get_best_intent(
                urdu_translation,
                context_intent=last_intent, 
                conversation_history=conversation_turns
            )
            entities_en = extract_entities_with_context(english_translation, conversation_turns)
            entities_ur = extract_entities_with_context(urdu_translation, conversation_turns)
            
            # Prefer English, fallback to Urdu if UNKNOWN
            intent = intent_en if intent_en != "UNKNOWN" else intent_ur
            entities = entities_en if intent_en != "UNKNOWN" else entities_ur

    # Handle incomplete commands
    is_incomplete, possible_intents = conv_manager.detect_incomplete_command(english_translation, intent)
    
    if is_incomplete and possible_intents:
        # Store suggestions for follow-up
        conv_context.suggested_intents = possible_intents
        conv_context.incomplete_command_count += 1
        
        # Generate clarification with options
        clarification_en = conv_manager.generate_clarification_options(possible_intents, "en")
        clarification_ur = conv_manager.generate_clarification_options(possible_intents, "ur")
        
        messages = {
            "en": clarification_en["message"],
            "ur": clarification_ur["message"]
        }
        
        # Add turn to conversation history
        response_data = {
            "transcription": original_text,
            "language": detected_lang,
            "translated": english_translation,
            "intent": "INCOMPLETE_COMMAND",
            "entities": entities,
            "context_from": last_intent,
            "messages": messages,
            "confirm_required": False,
            "suggested_commands": possible_intents,
            "recent_memory": conversation_history.get(user_id, []),
            "conversation_type": "clarification",
            "options": possible_intents
        }
        
        conv_context.add_turn(english_translation, "INCOMPLETE_COMMAND", entities, response_data)
        return response_data

    # Update conversation history
    history = conversation_history.get(user_id, [])
    history.append(english_translation)
    conversation_history[user_id] = history[-5:]  # Keep more history for context

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

    confirm_required = intent in ["SEND_MONEY", "DELETE_ACCOUNT"]
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
        # For logout, show success message instead of confirmation
        clarification_en = None  # This will use build_success instead
    elif intent == "UNKNOWN":
        # Enhanced unknown handling with conversation context
        suggested_commands = suggest_similar_intents(
            english_translation, 
            top_k=4,
            conversation_context=conversation_turns
        )
        
        # Try to extract partial intent clues
        partial_clues = extract_partial_intent_clues(english_translation)
        if partial_clues:
            suggested_commands = list(set(suggested_commands + partial_clues))[:4]
        
        # Store suggestions for potential follow-up
        conv_context.suggested_intents = suggested_commands
        
        examples_en = []
        for sc in suggested_commands:
            phrases = COMMANDS.get(sc, [])
            for ex in phrases:
                if all(ord(ch) < 128 for ch in ex):
                    examples_en.append(ex)
                if len(examples_en) >= 3:
                    break
            if len(examples_en) >= 3:
                break
                
        if not examples_en:
            if conv_context.incomplete_command_count > 0:
                clarification_en = (
                    "I'm still not sure what you want to do. Let me give you some options:\n"
                    "1. Check your balance\n"
                    "2. Show your QR code\n"
                    "3. View transactions\n"
                    "4. Request money from someone\n\n"
                    "Just say the number or tell me what you'd like to do."
                )
            else:
                clarification_en = (
                    "Sorry, I couldn't understand your command. "
                    "You can try: 'Show my balance', 'Show my QR code', or 'Request money from contacts'."
                )
        else:
            clarification_en = REPLY_TEMPLATES["UNKNOWN"]["clarification"]["en"].format(
                examples="; ".join(f"'{ex}'" for ex in examples_en[:3])
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
        # For logout, show success message instead of confirmation
        clarification_ur = None  # This will use build_success instead
    elif intent == "UNKNOWN":
        # Enhanced Urdu unknown handling
        examples_ur = []
        for sc in suggested_commands:
            phrases = COMMANDS.get(sc, [])
            for ex in phrases:
                if any(ord(ch) > 128 for ch in ex) or any(w in ex.lower() for w in ["dikhao", "paise", "krdo", "hai", "kahan", "mje", "ki", "se"]):
                    examples_ur.append(ex)
                if len(examples_ur) >= 3:
                    break
            if len(examples_ur) >= 3:
                break
                
        if not examples_ur:
            if conv_context.incomplete_command_count > 0:
                clarification_ur = (
                    "معذرت، اب بھی سمجھ نہیں آیا۔ یہ آپشنز ہیں:\n"
                    "1. اپنا بیلنس چیک کریں\n"
                    "2. اپنا QR کوڈ دیکھیں\n"
                    "3. ٹرانزیکشنز دیکھیں\n"
                    "4. کسی سے پیسے مانگیں\n\n"
                    "نمبر بولیں یا بتائیں کہ کیا کرنا چاہتے ہیں۔"
                )
            else:
                clarification_ur = (
                    "معذرت، میں سمجھ نہیں سکا۔ آپ یہ کہہ سکتے ہیں: 'میرا بیلنس دکھاؤ'، 'میرا کیو آر کوڈ دکھاؤ'، یا 'علی سے پیسے مانگو'۔"
                )
        else:
            clarification_ur = REPLY_TEMPLATES["UNKNOWN"]["clarification"]["ur"].format(
                examples="؛ ".join(f"'{ex}'" for ex in examples_ur[:3])
            )

    # Construct the final messages
    messages = {
        "en": clarification_en or build_success(intent, entities, "en"),
        "ur": clarification_ur or build_success(intent, entities, "ur")
    }
    
    # Determine conversation type
    conversation_type = "single_turn"
    if conv_manager.should_continue_conversation(intent, entities):
        conversation_type = "multi_turn"
    elif clarification_en or clarification_ur:
        conversation_type = "clarification"
    
    # Prepare response data
    response_data = {
        "transcription": original_text,
        "language": detected_lang,
        "translated": english_translation,
        "intent": intent,
        "entities": entities,
        "context_from": last_intent,
        "messages": messages,
        "confirm_required": confirm_required,
        "suggested_commands": suggested_commands,
        "recent_memory": conversation_history[user_id],
        "conversation_type": conversation_type,
        "conversation_id": conv_context.user_id,
        "turn_count": len(conv_context.conversation_flow) + 1
    }
    
    # Add this turn to conversation context
    conv_context.add_turn(english_translation, intent, entities, response_data)
    
    # Update conversation state for multi-step flows
    if intent not in ["UNKNOWN", "INCOMPLETE_COMMAND"]:
        conversation_state[user_id] = {"intent": intent, "entities": entities}
    
    return response_data


# Add this code to run the server directly with a custom port
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8282)
