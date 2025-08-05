#!/usr/bin/env python3
"""
Test script to verify mBART implementation works correctly
"""

import torch
from src.model import MultilingualTranslationModel
from src.data import DataProcessor
from src.evaluation import TranslationEvaluator
import json

def test_basic_functionality():
    """Test basic model functionality"""
    print("Testing basic functionality...")
    
    # Test model loading
    model = MultilingualTranslationModel("facebook/mbart-large-50")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Test tokenizer
    tokenizer = model.tokenizer
    test_text = "Hello, how are you?"
    tokens = tokenizer(test_text, return_tensors="pt")
    
    print(f"✓ Model loaded successfully")
    print(f"✓ Tokenizer working: {test_text} -> {tokens['input_ids'].shape}")
    
    return model, tokenizer, device

def test_translation():
    """Test translation functionality"""
    print("\nTesting translation...")
    
    model, tokenizer, device = test_basic_functionality()
    
    # Test translation
    source_texts = ["Hello, how are you?", "I love machine learning."]
    
    evaluator = TranslationEvaluator(model, tokenizer, device)
    translations = evaluator.translate_batch(source_texts)
    
    print("✓ Translation working:")
    for src, tgt in zip(source_texts, translations):
        print(f"  {src} -> {tgt}")
    
    return True

def test_data_loading():
    """Test data loading functionality"""
    print("\nTesting data loading...")
    
    processor = DataProcessor()
    
    # Test translation data
    train_data = processor.load_wmt_en_ro("train")[:10]
    print(f"✓ Loaded {len(train_data)} training samples")
    
    # Test monolingual data
    en_texts = processor.load_monolingual_data("en", 5)
    ro_texts = processor.load_monolingual_data("ro", 5)
    
    print(f"✓ Loaded {len(en_texts)} English monolingual samples")
    print(f"✓ Loaded {len(ro_texts)} Romanian monolingual samples")
    
    return True

def test_evaluation():
    """Test evaluation functionality"""
    print("\nTesting evaluation...")
    
    model, tokenizer, device = test_basic_functionality()
    evaluator = TranslationEvaluator(model, tokenizer, device)
    
    # Test evaluation with dummy data
    source_texts = ["Hello", "Thank you"]
    target_texts = ["Salut", "Mulțumesc"]
    
    results, hypotheses, references = evaluator.evaluate_dataset(
        source_texts, target_texts, batch_size=2
    )
    
    print("✓ Evaluation working:")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
    
    return True

def create_sample_data():
    """Create sample data for testing"""
    sample_data = [
        {"source": "Hello, how are you?", "target": "Salut, ce mai faci?"},
        {"source": "I love machine learning.", "target": "Îmi place învățarea automată."},
        {"source": "This is a test sentence.", "target": "Aceasta este o propoziție de test."},
        {"source": "The weather is nice today.", "target": "Vremea este frumoasă astăzi."},
        {"source": "Thank you very much.", "target": "Mulțumesc foarte mult."}
    ]
    
    with open("sample_test_data.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print("✓ Created sample test data")
    return sample_data

def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("Running mBART Implementation Tests")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_translation()
        test_data_loading()
        test_evaluation()
        create_sample_data()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed successfully!")
        print("=" * 50)
        print("\nTo start training:")
        print("  python scripts/train.py --mode finetune --batch_size 8 --max_epochs 1")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()