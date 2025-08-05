#!/usr/bin/env python3
"""
Training script for mBART implementation
"""

import argparse
import json
import os
import torch
from pathlib import Path

from src.trainer import run_pretraining, run_finetuning
from src.model import MultilingualTranslationModel
from src.data import DataProcessor
from src.evaluation import evaluate_model

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_directories(config: dict):
    """Create necessary directories"""
    os.makedirs(config["output"]["output_dir"], exist_ok=True)
    os.makedirs(config["output"]["log_dir"], exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Train mBART model")
    parser.add_argument("--mode", choices=["pretrain", "finetune", "evaluate"], 
                       default="finetune", help="Training mode")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--data_size", type=int, default=10000, 
                       help="Number of samples to use (for faster experimentation)")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Use default configs
        if args.mode == "pretrain":
            config = load_config("configs/pretrain_config.json")
        else:
            config = load_config("configs/translation_config.json")
    
    # Override config with command line arguments
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.max_steps and args.mode == "pretrain":
        config["training"]["max_steps"] = args.max_steps
    if args.max_epochs and args.mode == "finetune":
        config["training"]["max_epochs"] = args.max_epochs
    if args.output_dir:
        config["output"]["output_dir"] = args.output_dir
    
    # Setup directories
    setup_directories(config)
    
    # Run training/evaluation
    if args.mode == "pretrain":
        print("Starting denoising pre-training...")
        run_pretraining(
            batch_size=config["training"]["batch_size"],
            max_steps=config["training"]["max_steps"],
            learning_rate=config["training"]["learning_rate"],
            warmup_steps=config["training"]["warmup_steps"],
            max_length=config["data"]["max_length"],
            output_dir=config["output"]["output_dir"]
        )
        
    elif args.mode == "finetune":
        print("Starting English-Romanian translation fine-tuning...")
        run_finetuning(
            batch_size=config["training"]["batch_size"],
            max_epochs=config["training"]["max_epochs"],
            learning_rate=config["training"]["learning_rate"],
            warmup_steps=config["training"]["warmup_steps"],
            max_length=config["data"]["max_length"],
            output_dir=config["output"]["output_dir"]
        )
        
    elif args.mode == "evaluate":
        print("Evaluating model...")
        
        if not args.model_path:
            args.model_path = os.path.join(config["output"]["output_dir"], "best_translation_model.pt")
        
        results = evaluate_model(
            model_path=args.model_path,
            output_dir="evaluation_results"
        )
        
        print("\nFinal Results:")
        for metric, value in results.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")

if __name__ == "__main__":
    main()