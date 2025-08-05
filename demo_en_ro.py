#!/usr/bin/env python3
"""
Demo script for English-Romanian translation with mBART
This uses the Hugging Face transformers library directly
"""

# Try to import dependencies
try:
    from transformers import MBartForConditionalGeneration, MBartTokenizer
    import torch
    print("✅ Dependencies available")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Install with: pip3 install transformers torch")
    exit(1)

# Initialize model and tokenizer
print("📦 Loading mBART model...")
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBartTokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Test English to Romanian translation
test_sentences = [
    "Hello, how are you?",
    "I love machine learning.",
    "The weather is nice today.",
    "Thank you very much.",
    "Good morning!"
]

print("🎯 Testing English → Romanian translation:")
print("=" * 50)

for sentence in test_sentences:
    # Set source language to English
    tokenizer.src_lang = "en_XX"
    
    # Tokenize input
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    
    # Generate translation to Romanian
    tokenizer.tgt_lang = "ro_RO"
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["ro_RO"],
        max_length=50,
        num_beams=4,
        early_stopping=True
    )
    
    # Decode translation
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    print(f"EN: {sentence}")
    print(f"RO: {translation}")
    print("-" * 50)

print("✅ Demo complete!")
