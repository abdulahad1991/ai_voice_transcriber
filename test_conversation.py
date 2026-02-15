#!/usr/bin/env python3
"""
Test script to demonstrate enhanced conversational capabilities
"""

from conversation_manager import ConversationalManager
from semantic_matcher import (
    get_best_intent, extract_entities_with_context, 
    suggest_similar_intents, extract_partial_intent_clues
)

def test_conversation_scenarios():
    """Test various conversation scenarios"""
    conv_manager = ConversationalManager()
    user_id = "test-user"
    
    print("🤖 Enhanced Voice Q&A Conversational System Test")
    print("=" * 50)
    
    # Test Scenario 1: Incomplete commands
    print("\n📝 Test 1: Incomplete Commands")
    print("-" * 30)
    
    incomplete_commands = [
        "show me",
        "i want to check",
        "mje dikhao",
        "can you help me with",
        "please tell me"
    ]
    
    for cmd in incomplete_commands:
        intent = get_best_intent(cmd)
        is_incomplete, suggestions = conv_manager.detect_incomplete_command(cmd, intent)
        
        print(f"Input: '{cmd}'")
        print(f"Is incomplete: {is_incomplete}")
        if is_incomplete:
            clarification = conv_manager.generate_clarification_options(suggestions)
            print(f"Suggestions: {suggestions}")
            print(f"Response: {clarification['message'][:100]}...")
        print()
    
    # Test Scenario 2: Contextual references
    print("\n📝 Test 2: Contextual References")
    print("-" * 30)
    
    context = conv_manager.get_or_create_context(user_id)
    
    # Simulate previous interaction
    context.add_turn("show my balance", "CHECK_BALANCE", {}, {"intent": "CHECK_BALANCE"})
    
    contextual_commands = [
        "show that again",
        "do it again",
        "same thing",
        "wahi dikhao",
        "dubara"
    ]
    
    for cmd in contextual_commands:
        contextual_intent = conv_manager.handle_contextual_reference(cmd, context)
        print(f"Input: '{cmd}' -> Contextual Intent: {contextual_intent}")
    
    # Test Scenario 3: Progressive conversation
    print("\n📝 Test 3: Progressive Conversation")
    print("-" * 30)
    
    conversation_flow = [
        "i want money",
        "from ali",
        "500 rupees",
        "yes send it"
    ]
    
    context = conv_manager.get_or_create_context("progressive-user")
    
    for turn_num, user_input in enumerate(conversation_flow, 1):
        print(f"Turn {turn_num}: User says '{user_input}'")
        
        # Get conversation context
        recent_turns = context.get_recent_context(3)
        
        # Extract intent with context
        intent = get_best_intent(
            user_input,
            context_intent=context.current_intent,
            conversation_history=recent_turns
        )
        
        # Extract entities with context
        entities = extract_entities_with_context(user_input, recent_turns)
        
        print(f"  Intent: {intent}")
        print(f"  Entities: {entities}")
        
        # Check if incomplete
        is_incomplete, suggestions = conv_manager.detect_incomplete_command(user_input, intent)
        if is_incomplete:
            print(f"  Incomplete - Suggestions: {suggestions}")
        
        # Add to context
        context.add_turn(user_input, intent, entities, {"intent": intent})
        context.current_intent = intent
        
        print()
    
    # Test Scenario 4: Partial intent extraction
    print("\n📝 Test 4: Partial Intent Extraction")
    print("-" * 30)
    
    partial_commands = [
        "money",
        "balance check",
        "qr code",
        "paise dekho",
        "transaction"
    ]
    
    for cmd in partial_commands:
        clues = extract_partial_intent_clues(cmd)
        print(f"Input: '{cmd}' -> Clues: {clues}")
    
    # Test Scenario 5: Option selection
    print("\n📝 Test 5: Option Selection")
    print("-" * 30)
    
    context = conv_manager.get_or_create_context("option-user")
    context.suggested_intents = ["CHECK_BALANCE", "SHOW_QR", "SHOW_TRANSACTIONS"]
    
    option_responses = [
        "1",
        "first",
        "pehla",
        "yes",
        "haan"
    ]
    
    for response in option_responses:
        selected_intent = conv_manager.handle_option_selection(response, context)
        print(f"Response: '{response}' -> Selected: {selected_intent}")
    
    print("\n✅ All conversation tests completed!")

def demonstrate_conversational_flow():
    """Demonstrate a complete conversational flow"""
    print("\n🎭 Conversational Flow Demonstration")
    print("=" * 40)
    
    conv_manager = ConversationalManager()
    user_id = "demo-user"
    context = conv_manager.get_or_create_context(user_id)
    
    # Simulated conversation
    conversation = [
        ("User", "i want to"),
        ("System", "I'm not sure what you want to do. Here are some options:\n1. Check your balance\n2. Show your QR code\n3. View transactions\n4. Request money from someone"),
        ("User", "2"),
        ("System", "Here is your QR code."),
        ("User", "now show transactions"),
        ("System", "Here are your transactions."),
        ("User", "same thing again"),
        ("System", "Here are your transactions."),
        ("User", "balance check karo"),
        ("System", "Here is your balance.")
    ]
    
    print("Sample conversation:")
    for speaker, message in conversation:
        print(f"{speaker}: {message}")
        if speaker == "User":
            # Process the user input
            intent = get_best_intent(message, context_intent=context.current_intent)
            entities = extract_entities_with_context(message, context.get_recent_context(3))
            
            # Handle contextual references
            if intent == "UNKNOWN":
                contextual_intent = conv_manager.handle_contextual_reference(message, context)
                if contextual_intent:
                    intent = contextual_intent
            
            # Handle option selection
            if intent == "UNKNOWN":
                selected_intent = conv_manager.handle_option_selection(message, context)
                if selected_intent:
                    intent = selected_intent
            
            # Update context
            context.add_turn(message, intent, entities, {"intent": intent})
            context.current_intent = intent
        
        print()

if __name__ == "__main__":
    test_conversation_scenarios()
    demonstrate_conversational_flow()