from sentence_transformers import SentenceTransformer, util
from command_db import COMMANDS
from reply_template import REPLY_TEMPLATES
from transformers import pipeline
import re


model = SentenceTransformer("all-MiniLM-L6-v2")
intent_to_sentences = []
sentence_to_intent = []
for intent, examples in COMMANDS.items():
    for ex in examples:
        intent_to_sentences.append(ex)
        sentence_to_intent.append(intent)


command_embeddings = model.encode(intent_to_sentences, convert_to_tensor=True)


ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

def get_best_intent(user_text, threshold=0.50):
    query_embedding = model.encode(user_text, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, command_embeddings)[0]
    best_idx = scores.argmax().item()
    best_score = scores[best_idx].item()
    # Boost for obvious short commands
    text_lower = user_text.lower()
    if best_score < threshold:
        # Add heuristics for common partial phrases
        if "request" in text_lower and "money" in text_lower:
            return "REQUEST_FROM_PERSON"
    return sentence_to_intent[best_idx] if best_score >= threshold else "UNKNOWN"

def suggest_similar_intents(user_text, top_k=3):
    query_embedding = model.encode(user_text, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, command_embeddings)[0]
    top_indices = scores.argsort(descending=True)[:top_k]
    return [sentence_to_intent[i] for i in top_indices if scores[i].item() > 0.4]

def extract_entities(text):
    entities = {"person_names": [], "amounts": []}
    phone_match = re.search(r'(\d{10,15})', text)
    if phone_match: entities['phone_number'] = phone_match.group(1)
    password_match = re.search(r'password\s*is\s*(\d{6,})', text, re.IGNORECASE)
    if password_match: entities['password'] = password_match.group(1)
    amounts = re.findall(r'(\d{2,6})\s*(rs|rupees|rupay)?', text, re.IGNORECASE)
    if amounts: entities["amounts"] = [amt[0] for amt in amounts]

    results = ner(text)
    for item in results:
        if item["entity_group"] == "PER" and item["word"] not in entities["person_names"]:
            entities["person_names"].append(item["word"])
        elif item["entity_group"] == "LOC" and "location" not in entities:
            entities["location"] = item["word"]

    # Fallback: check common names if NER misses
    COMMON_NAMES = ["ali", "ahmed", "fatima", "sara", "raza", "hassan"]
    for word in text.split():
        if word.lower() in COMMON_NAMES and word not in entities["person_names"]:
            entities["person_names"].append(word)

    return entities

def update_conversation_state(user_id, intent, entities, state, lang_code):
    if user_id not in state:
        state[user_id] = {}

    state[user_id]["intent"] = intent

    # LOGIN
    if intent == "LOGIN":
        state[user_id]["pending"] = ["phone_number", "password"]
        return REPLY_TEMPLATES["LOGIN"]["clarification"][lang_code]

    elif intent == "COMPLETE_LOGIN_STEP_1":
        state[user_id]["pending"] = ["password"]
        state[user_id]["phone_number"] = entities.get("phone_number")
        return "Now, please provide your password." if lang_code == "en" else "اب، براہ کرم اپنا پاسورڈ بتائیں۔"

    elif intent == "COMPLETE_LOGIN_STEP_2":
        state[user_id]["pending"] = []
        state[user_id]["password"] = entities.get("password")
        return REPLY_TEMPLATES["LOGIN"]["success"][lang_code]

    # REQUEST_FROM_PERSON
    elif intent == "REQUEST_FROM_PERSON":
        state[user_id]["pending"] = ["person_names", "amounts"]
        # The clarification for missing handled in endpoint

    elif intent == "COMPLETE_REQUEST_STEP":
        state[user_id]["pending"] = []
        person = (entities.get("person_names") or ["someone"])[0]
        amount = (entities.get("amounts") or ["some amount"])[0]
        return REPLY_TEMPLATES["REQUEST_FROM_PERSON"]["success"][lang_code].format(
            person=person, amount=amount
        )

    # SIGNUP
    elif intent == "SIGNUP_HELP":
        state[user_id]["pending"] = ["phone_number", "password"]
        return REPLY_TEMPLATES["SIGNUP_HELP"]["clarification"][lang_code]

    elif intent == "COMPLETE_SIGNUP":
        state[user_id]["pending"] = []
        return REPLY_TEMPLATES["SIGNUP_HELP"]["success"][lang_code]

    # LOGOUT
    elif intent == "LOGOUT":
        return REPLY_TEMPLATES["LOGOUT"]["confirm"][lang_code]

    # Generic responses for simple commands
    elif intent in REPLY_TEMPLATES and "success" in REPLY_TEMPLATES[intent]:
        return REPLY_TEMPLATES[intent]["success"][lang_code]

    return f"✅ Detected intent: {intent}. Entities: {entities}"