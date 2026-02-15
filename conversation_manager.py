from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

class ConversationContext:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.current_intent = None
        self.pending_entities = []
        self.collected_entities = {}
        self.conversation_flow = []
        self.last_activity = datetime.now()
        self.context_memory = {}
        self.incomplete_command_count = 0
        self.suggested_intents = []
        
    def add_turn(self, user_input: str, intent: str, entities: dict, response: dict):
        turn = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "intent": intent,
            "entities": entities,
            "response": response
        }
        self.conversation_flow.append(turn)
        # Keep last 10 turns for context
        if len(self.conversation_flow) > 10:
            self.conversation_flow = self.conversation_flow[-10:]
        self.last_activity = datetime.now()
    
    def get_recent_context(self, turns: int = 3) -> List[dict]:
        return self.conversation_flow[-turns:] if self.conversation_flow else []
    
    def is_session_active(self, timeout_minutes: int = 30) -> bool:
        return datetime.now() - self.last_activity < timedelta(minutes=timeout_minutes)

class ConversationalManager:
    def __init__(self):
        self.contexts: Dict[str, ConversationContext] = {}
        
        # Enhanced partial command patterns
        self.partial_patterns = {
            "show": ["SHOW_QR", "SHOW_TRANSACTIONS", "SHOW_PAYMENTS", "CHECK_BALANCE"],
            "check": ["CHECK_BALANCE", "SHOW_TRANSACTIONS"],
            "send": ["REQUEST_FROM_PERSON"],
            "request": ["REQUEST_FROM_PERSON"],
            "money": ["REQUEST_FROM_PERSON", "CHECK_BALANCE"],
            "balance": ["CHECK_BALANCE"],
            "qr": ["SHOW_QR", "SHOW_QR_LOCATION"],
            "transaction": ["SHOW_TRANSACTIONS", "SHOW_QR_TRANSACTIONS"],
            "payment": ["SHOW_PAYMENTS"],
            "login": ["LOGIN"],
            "logout": ["LOGOUT"],
            "sign": ["SIGNUP_HELP"],
            "setting": ["OPEN_SETTINGS"],
            
            # Urdu patterns
            "dikhao": ["SHOW_QR", "SHOW_TRANSACTIONS", "SHOW_PAYMENTS", "CHECK_BALANCE"],
            "dikha": ["SHOW_QR", "SHOW_TRANSACTIONS", "SHOW_PAYMENTS", "CHECK_BALANCE"],
            "paise": ["CHECK_BALANCE", "REQUEST_FROM_PERSON"],
            "mangna": ["REQUEST_FROM_PERSON"],
            "bhejo": ["REQUEST_FROM_PERSON"],
            "balance": ["CHECK_BALANCE"],
            "setting": ["OPEN_SETTINGS"],
        }
        
        # Contextual follow-up patterns
        self.contextual_responses = {
            "that": "previous_intent",
            "it": "previous_intent", 
            "same": "previous_intent",
            "again": "previous_intent",
            "wahi": "previous_intent",
            "wohi": "previous_intent",
            "dubara": "previous_intent",
        }
        
    def get_or_create_context(self, user_id: str) -> ConversationContext:
        if user_id not in self.contexts:
            self.contexts[user_id] = ConversationContext(user_id)
        elif not self.contexts[user_id].is_session_active():
            # Reset context if session expired
            self.contexts[user_id] = ConversationContext(user_id)
        return self.contexts[user_id]
    
    def detect_incomplete_command(self, text: str, intent: str) -> Tuple[bool, List[str]]:
        """Detect if command is incomplete and suggest possible completions"""
        text_lower = text.lower().strip()
        words = text_lower.split()
        
        # Very short commands are likely incomplete
        if len(words) <= 2 and intent == "UNKNOWN":
            possible_intents = []
            for word in words:
                if word in self.partial_patterns:
                    possible_intents.extend(self.partial_patterns[word])
            
            if possible_intents:
                return True, list(set(possible_intents))
        
        # Check for common incomplete patterns
        incomplete_starters = [
            "i want", "mje", "please", "can you", "show me", "tell me", 
            "help me", "how to", "where is", "kahan hai", "kese", "btao"
        ]
        
        if any(text_lower.startswith(starter) for starter in incomplete_starters) and intent == "UNKNOWN":
            return True, self._guess_intent_from_context(text_lower)
        
        return False, []
    
    def _guess_intent_from_context(self, text: str) -> List[str]:
        """Guess possible intents based on partial text"""
        suggestions = []
        
        # Check for key words
        for word, intents in self.partial_patterns.items():
            if word in text:
                suggestions.extend(intents)
        
        # If no specific matches, suggest common actions
        if not suggestions:
            suggestions = ["CHECK_BALANCE", "SHOW_QR", "SHOW_TRANSACTIONS"]
        
        return list(set(suggestions))
    
    def handle_contextual_reference(self, text: str, context: ConversationContext) -> Optional[str]:
        """Handle references like 'that', 'it', 'same', 'again'"""
        text_lower = text.lower().strip()
        
        for ref_word in self.contextual_responses:
            if ref_word in text_lower:
                recent_turns = context.get_recent_context(3)
                if recent_turns:
                    # Find the last successful intent
                    for turn in reversed(recent_turns):
                        if turn["intent"] != "UNKNOWN":
                            return turn["intent"]
        
        return None
    
    def generate_clarification_options(self, possible_intents: List[str], lang: str = "en") -> dict:
        """Generate user-friendly clarification with options"""
        
        intent_descriptions = {
            "en": {
                "CHECK_BALANCE": "Check your account balance",
                "SHOW_QR": "Show your QR code",
                "SHOW_TRANSACTIONS": "View transaction history", 
                "SHOW_PAYMENTS": "View payment history",
                "REQUEST_FROM_PERSON": "Request money from someone",
                "LOGIN": "Log into your account",
                "LOGOUT": "Log out of your account",
                "OPEN_SETTINGS": "Open app settings"
            },
            "ur": {
                "CHECK_BALANCE": "اپنا بیلنس چیک کریں",
                "SHOW_QR": "اپنا QR کوڈ دیکھیں", 
                "SHOW_TRANSACTIONS": "ٹرانزیکشن ہسٹری دیکھیں",
                "SHOW_PAYMENTS": "پیمنٹ ہسٹری دیکھیں",
                "REQUEST_FROM_PERSON": "کسی سے پیسے مانگیں",
                "LOGIN": "اپنے اکاؤنٹ میں لاگ ان کریں",
                "LOGOUT": "اپنے اکاؤنٹ سے لاگ آؤٹ کریں",
                "OPEN_SETTINGS": "ایپ سیٹنگز کھولیں"
            }
        }
        
        descriptions = intent_descriptions.get(lang, intent_descriptions["en"])
        
        options = []
        for i, intent in enumerate(possible_intents[:4], 1):  # Limit to 4 options
            if intent in descriptions:
                options.append(f"{i}. {descriptions[intent]}")
        
        if lang == "en":
            message = f"I'm not sure what you want to do. Here are some options:\n" + "\n".join(options) + "\n\nYou can say the number or describe what you want to do."
        else:
            message = f"معذرت، سمجھ نہیں آیا۔ یہ آپشنز ہیں:\n" + "\n".join(options) + "\n\nآپ نمبر بول سکتے ہیں یا بتا سکتے ہیں کہ کیا کرنا چاہتے ہیں۔"
        
        return {
            "message": message,
            "options": possible_intents,
            "type": "multiple_choice"
        }
    
    def handle_option_selection(self, text: str, context: ConversationContext) -> Optional[str]:
        """Handle when user selects an option by number or confirmation"""
        text_lower = text.strip().lower()
        
        # Check if user selected a number
        if text_lower in ["1", "first", "pehla", "pahla"]:
            return context.suggested_intents[0] if context.suggested_intents else None
        elif text_lower in ["2", "second", "dusra", "dosra"]: 
            return context.suggested_intents[1] if len(context.suggested_intents) > 1 else None
        elif text_lower in ["3", "third", "tisra", "teesra"]:
            return context.suggested_intents[2] if len(context.suggested_intents) > 2 else None
        elif text_lower in ["4", "fourth", "chotha", "chautha"]:
            return context.suggested_intents[3] if len(context.suggested_intents) > 3 else None
        
        # Check for confirmation words
        confirmation_words = ["yes", "yeah", "ok", "okay", "haan", "han", "bilkul", "theek"]
        if any(word in text_lower for word in confirmation_words):
            return context.suggested_intents[0] if context.suggested_intents else None
            
        return None
    
    def should_continue_conversation(self, intent: str, entities: dict) -> bool:
        """Determine if this should continue the conversation vs end it"""
        # Always continue for incomplete commands
        if intent == "UNKNOWN":
            return True
            
        # Continue for multi-step processes
        multi_step_intents = ["LOGIN", "REQUEST_FROM_PERSON", "SIGNUP_HELP"]
        if intent in multi_step_intents:
            return True
            
        # Continue if entities are missing for the intent
        required_entities = {
            "REQUEST_FROM_PERSON": ["person_names", "amounts"],
            "LOGIN": ["phone_number"],
            "SIGNUP_HELP": ["phone_number", "password"]
        }
        
        if intent in required_entities:
            missing = [e for e in required_entities[intent] if e not in entities or not entities[e]]
            if missing:
                return True
        
        return False