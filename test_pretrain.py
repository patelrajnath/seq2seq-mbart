#!/usr/bin/env python3
"""
Quick test for pretraining functionality
"""

import sys
sys.path.insert(0, ".")

from src.data import DataProcessor
from src.trainer import run_pretraining

def test_pretrain_setup():
    """Test pretraining setup"""
    print("🧪 Testing Pretraining Setup")
    print("=" * 40)
    
    try:
        processor = DataProcessor()
        print("✅ Data processor initialized")
        
        # Test pretrain dataloader creation
        dataloader = processor.create_pretrain_dataloaders(
            batch_size=2,
            max_length=64,
            num_samples=20
        )
        
        # Test one batch
        batch = next(iter(dataloader))
        print(f"✅ Batch created: {batch['input_ids'].shape}")
        print(f"✅ Input IDs shape: {batch['input_ids'].shape}")
        print(f"✅ Labels shape: {batch['labels'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pretrain_setup()
    if success:
        print("\n🎉 Pretraining setup working!")
    else:
        print("\n⚠️  Pretraining needs debugging")