from sentence_transformers import SentenceTransformer, util
from command_db import COMMANDS
from reply_template import REPLY_TEMPLATES
from transformers import pipeline
import re
from fuzzywuzzy import fuzz

# --- Build keyword and phrase mappings ---
phrase_to_intent = {}
intent_phrases = {}
for intent, examples in COMMANDS.items():
    cleaned_examples = set()
    for ex in examples:
        cleaned = ex.lower().strip()
        phrase_to_intent[cleaned] = intent
        cleaned_examples.add(cleaned)
    intent_phrases[intent] = cleaned_examples

# --- Prepare embeddings ---
model = SentenceTransformer("all-MiniLM-L6-v2")
intent_to_sentences = []
sentence_to_intent = []
for intent, examples in COMMANDS.items():
    for ex in examples:
        intent_to_sentences.append(ex)
        sentence_to_intent.append(intent)
command_embeddings = model.encode(intent_to_sentences, convert_to_tensor=True)

ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

# --- Intent matching functions ---

def normalize_input(text):
    """Optionally normalize/expand common abbreviations here."""
    text = text.lower().strip()
    return text

def keyword_match(user_text):
    text_lower = user_text.lower().strip()
    # 1. Exact match
    if text_lower in phrase_to_intent:
        return phrase_to_intent[text_lower], 1.0
    # 2. Partial/substring match
    for intent, phrases in intent_phrases.items():
        for phrase in phrases:
            if phrase in text_lower or text_lower in phrase:
                return intent, 0.8
    return None, 0

def semantic_match(user_text, threshold=0.50):
    query_embedding = model.encode(user_text, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, command_embeddings)[0]
    best_idx = scores.argmax().item()
    best_score = scores[best_idx].item()
    if best_score >= threshold:
        return sentence_to_intent[best_idx], best_score
    return None, best_score

def fuzzy_match(user_text, fuzzy_threshold=75):
    best_score = 0
    best_intent = None
    for i, example in enumerate(intent_to_sentences):
        ratio_score = fuzz.ratio(user_text.lower(), example.lower())
        partial_score = fuzz.partial_ratio(user_text.lower(), example.lower())
        token_score = fuzz.token_sort_ratio(user_text.lower(), example.lower())
        score = max(ratio_score, partial_score, token_score)
        if score > best_score and score >= fuzzy_threshold:
            best_score = score
            best_intent = sentence_to_intent[i]
    if best_intent:
        return best_intent, best_score / 100.0
    return None, 0

def get_best_intent(user_text, semantic_threshold=0.50, fuzzy_threshold=75):
    # 1. Keyword matching (high confidence)
    intent, confidence = keyword_match(user_text)
    if intent and confidence > 0.85:
        return intent
    # 2. Semantic matching (transformer)
    intent, confidence = semantic_match(user_text, semantic_threshold)
    if intent:
        return intent
    # 3. Fuzzy (typo-tolerant) matching fallback
    intent, confidence = fuzzy_match(user_text, fuzzy_threshold)
    if intent:
        return intent
    # 4. Still nothing? Try partial matches again (for safety)
    text_lower = user_text.lower()
    for intent, phrases in intent_phrases.items():
        for phrase in phrases:
            if phrase in text_lower or text_lower in phrase:
                return intent
    return "UNKNOWN"

def suggest_similar_intents(user_text, top_k=3):
    query_embedding = model.encode(user_text, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, command_embeddings)[0]
    top_indices = scores.argsort(descending=True)[:top_k]
    suggestions = []
    for i in top_indices:
        if scores[i].item() > 0.3:
            intent = sentence_to_intent[i]
            if intent not in suggestions:
                suggestions.append(intent)
    return suggestions[:top_k]

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
