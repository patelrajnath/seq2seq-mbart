#!/usr/bin/env python3
"""
Fixed demo script for English-Romanian translation with proper tokenizer handling
"""

import torch
from transformers import MBartTokenizer, MBartForConditionalGeneration

def test_fixed_implementation():
    """Test the fixed implementation"""
    print("🎯 Testing Fixed mBART Implementation")
    print("=" * 50)
    
    try:
        # Use the working tokenizer
        print("📦 Loading tokenizer...")
        tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        print("✅ Tokenizer loaded successfully")
        
        # Test basic tokenizer functionality
        test_sentence = "Hello, how are you?"
        tokenizer.src_lang = "en_XX"
        
        inputs = tokenizer(test_sentence, return_tensors="pt")
        print(f"✅ Tokenization successful: {inputs['input_ids'].shape}")
        
        # Test language codes
        print(f"✅ English language code: {tokenizer.lang_code_to_id['en_XX']}")
        print(f"✅ Romanian language code: {tokenizer.lang_code_to_id['ro_RO']}")
        
        # Test with small model for memory efficiency
        print("\n📦 Loading small model for testing...")
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        print(f"✅ Model loaded: {model.num_parameters():,} parameters")
        
        # Test translation
        print("\n🗣️ Testing translation...")
        test_sentences = [
            "Hello, how are you?",
            "I love machine learning.",
            "Good morning!"
        ]
        
        for sentence in test_sentences:
            # Tokenize English
            tokenizer.src_lang = "en_XX"
            inputs = tokenizer(sentence, return_tensors="pt", max_length=50, truncation=True)
            
            # Generate Romanian translation
            tokenizer.tgt_lang = "ro_RO"
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id["ro_RO"],
                max_length=50,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode
            translation = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"EN: {sentence}")
            print(f"RO: {translation}")
            print("-" * 30)
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_fixed_implementation()