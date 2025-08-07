#!/usr/bin/env python3
"""
Quick test to identify critical training bugs
"""

import json
import os
import sys

def quick_check():
    """Quick analysis of potential training bugs"""
    print("🔍 Quick Training Bug Analysis")
    print("=" * 50)
    
    bugs_found = []
    
    # Check 1: Configuration consistency
    print("\n1. Configuration Analysis:")
    try:
        with open("configs/translation_config.json", "r") as f:
            trans = json.load(f)
        with open("configs/pretrain_config.json", "r") as f:
            pretrain = json.load(f)
        
        print(f"   Translation LR: {trans['training']['learning_rate']}")
        print(f"   Pretrain LR: {pretrain['training']['learning_rate']}")
        
        if trans['training']['learning_rate'] > 1e-4:
            bugs_found.append("Translation LR might be too high")
        if pretrain['training']['learning_rate'] > 1e-3:
            bugs_found.append("Pretrain LR might be too high")
            
    except Exception as e:
        bugs_found.append(f"Config error: {e}")
    
    # Check 2: Data pipeline issues
    print("\n2. Data Pipeline Analysis:")
    try:
        # Check if dummy data exists
        dummy_train = os.path.exists("data/dummy/train.json")
        dummy_val = os.path.exists("data/dummy/val.json")
        
        if not (dummy_train or dummy_val):
            bugs_found.append("Missing dummy data files")
        
        print(f"   Dummy train data: {'✅' if dummy_train else '❌'}")
        print(f"   Dummy val data: {'✅' if dummy_val else '❌'}")
        
    except Exception as e:
        bugs_found.append(f"Data pipeline error: {e}")
    
    # Check 3: File structure issues
    print("\n3. File Structure Analysis:")
    required_files = [
        "src/model.py", "src/data.py", "src/trainer.py", 
        "scripts/train.py", "requirements.txt"
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        bugs_found.append(f"Missing files: {missing}")
    
    print(f"   Missing files: {len(missing)}")
    
    # Check 4: Memory estimation
    print("\n4. Memory Requirements:")
    print("   mBART-large: ~1.7GB parameters + ~3-5GB activations")
    print("   Recommended: 8GB+ GPU memory")
    
    # Check 5: Common training issues
    print("\n5. Common Training Issues:")
    common_issues = [
        "Learning rate too high (try 3e-5 for translation)",
        "Batch size too large for GPU memory",
        "Missing language tokens in tokenizer setup",
        "Incorrect decoder_input_ids generation",
        "Gradient accumulation not implemented",
        "No early stopping mechanism"
    ]
    
    for issue in common_issues:
        print(f"   ⚠️  {issue}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 BUG DETECTION SUMMARY")
    print("=" * 50)
    
    if bugs_found:
        print("❌ Bugs found:")
        for bug in bugs_found:
            print(f"   • {bug}")
    else:
        print("✅ No critical bugs detected")
    
    print("\n🎯 Recommendations:")
    print("   1. Use learning rate 3e-5 for translation fine-tuning")
    print("   2. Start with batch_size=4 for memory constraints")
    print("   3. Monitor GPU memory usage")
    print("   4. Use gradient checkpointing for large models")
    print("   5. Implement validation every few steps")
    
    return len(bugs_found) == 0

if __name__ == "__main__":
    quick_check()