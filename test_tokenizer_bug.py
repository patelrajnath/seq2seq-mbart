#!/usr/bin/env python3
"""
Test script to reproduce and fix the tokenizer bug
"""

def test_tokenizer_fix():
    """Test different approaches to fix tokenizer issue"""
    print("🔧 Testing tokenizer fixes...")
    
    # Test 1: Try MBartTokenizer (older approach)
    try:
        from transformers import MBartTokenizer
        tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        print("✅ MBartTokenizer works")
        return "MBartTokenizer"
    except Exception as e:
        print(f"❌ MBartTokenizer failed: {e}")
    
    # Test 2: Try MBart50TokenizerFast with specific model
    try:
        from transformers import MBart50TokenizerFast
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        print("✅ MBart50TokenizerFast works")
        return "MBart50TokenizerFast"
    except Exception as e:
        print(f"❌ MBart50TokenizerFast failed: {e}")
    
    # Test 3: Try without fast tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", use_fast=False)
        print("✅ AutoTokenizer (slow) works")
        return "AutoTokenizer"
    except Exception as e:
        print(f"❌ AutoTokenizer failed: {e}")
    
    return None

if __name__ == "__main__":
    test_tokenizer_fix()