#!/usr/bin/env python3
"""
Working demo for English-Romanian mBART translation
"""

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

def test_mbert_en_ro():
    """Test mBART with English-Romanian translation"""
    
    print("🚀 Testing mBART English-Romanian Translation")
    print("=" * 60)
    
    # Initialize model and tokenizer
    print("📦 Loading model and tokenizer...")
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    
    try:
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test sentences
    test_sentences = [
        "Hello, how are you?",
        "I love machine learning and artificial intelligence.",
        "The weather is beautiful today in London.",
        "Thank you very much for your help.",
        "Good morning! I hope you have a great day.",
        "Can you tell me where the nearest restaurant is?",
        "I'm studying computer science at university.",
        "The conference was very interesting and informative."
    ]
    
    print(f"\n🎯 Testing {len(test_sentences)} English → Romanian translations:")
    print("-" * 60)
    
    # Set source and target languages
    tokenizer.src_lang = "en_XX"
    target_lang = "ro_RO"
    
    # Process each sentence
    for i, sentence in enumerate(test_sentences, 1):
        try:
            # Tokenize input
            inputs = tokenizer(
                sentence, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=128
            )
            
            # Generate translation
            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
                    max_length=128,
                    num_beams=4,
                    early_stopping=True,
                    length_penalty=1.0
                )
            
            # Decode translation
            translation = tokenizer.decode(
                generated_tokens[0], 
                skip_special_tokens=True
            )
            
            print(f"{i}. EN: {sentence}")
            print(f"   RO: {translation}")
            print()
            
        except Exception as e:
            print(f"{i}. ❌ Error translating: {sentence}")
            print(f"   Error: {e}")
    
    print("✅ Translation complete!")

def test_batch_translation():
    """Test batch translation for efficiency"""
    
    print("\n📦 Testing Batch Translation")
    print("=" * 40)
    
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    
    tokenizer.src_lang = "en_XX"
    target_lang = "ro_RO"
    
    # Batch of sentences
    batch_sentences = [
        "Hello world",
        "Good morning",
        "Thank you",
        "How are you"
    ]
    
    print("Batch translating:", batch_sentences)
    
    # Tokenize batch
    inputs = tokenizer(
        batch_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    )
    
    # Generate translations
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
            max_length=64,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode all translations
    translations = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )
    
    for original, translated in zip(batch_sentences, translations):
        print(f"EN: {original} → RO: {translated}")

if __name__ == "__main__":
    test_mbert_en_ro()
    test_batch_translation()