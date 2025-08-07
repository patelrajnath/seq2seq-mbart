import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, Tuple
import os
import json
from datetime import datetime

from .model import MultilingualDenoisingPretraining, MultilingualTranslationModel, NoiseGenerator
from .data import DataProcessor

class BaseTrainer:
    """
    Base trainer class with common functionality
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device: torch.device,
        output_dir: str,
        log_interval: int = 50,
        eval_interval: int = 500,
        save_interval: int = 1000,
    ):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        os.makedirs(output_dir, exist_ok=True)
        
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'step': self.step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, os.path.join(self.output_dir, filename))
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']

class DenoisingPretrainingTrainer(BaseTrainer):
    """
    Trainer for multilingual denoising pre-training
    """
    
    def __init__(
        self,
        model: MultilingualDenoisingPretraining,
        train_loader,
        val_loader,
        device: torch.device,
        output_dir: str,
        config: Dict,
        noise_generator: Optional[NoiseGenerator] = None,
    ):
        self.noise_generator = noise_generator
        
        training_config = config.get("training", {})
        optimizer = AdamW(
            model.parameters(),
            lr=training_config.get("learning_rate", 5e-4),
            weight_decay=training_config.get("weight_decay", 0.01),
            betas=(0.9, 0.98),
            eps=1e-6
        )
        
        max_steps = training_config.get("max_steps", 10000)
        warmup_steps = training_config.get("warmup_steps", 1000)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps
        )
        
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_dir=output_dir,
        )
        
        self.max_steps = config.get("training", {}).get("max_steps", 10000)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Add noise to input
        if self.noise_generator:
            batch["input_ids"] = self.noise_generator.add_noise(
                batch["input_ids"], noise_type="span_masking"
            )
        
        # Forward pass
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            decoder_attention_mask=batch["decoder_attention_mask"],
            labels=batch["labels"]
        )
        
        loss = outputs["loss"]
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def validate(self) -> float:
        """Validation step"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Add noise to input
                if self.noise_generator:
                    batch["input_ids"] = self.noise_generator.add_noise(
                        batch["input_ids"], noise_type="span_masking"
                    )
                
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=batch["decoder_input_ids"],
                    decoder_attention_mask=batch["decoder_attention_mask"],
                    labels=batch["labels"]
                )
                
                total_loss += outputs["loss"].item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self):
        """Training loop"""
        print(f"Starting pre-training for {self.max_steps} steps...")
        
        try:
            import wandb
            wandb.init(
                project="mbart-pretraining",
                name=f"denoising_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            use_wandb = True
        except Exception:
            print("⚠️  wandb not available, using console logging only")
            use_wandb = False
        
        progress_bar = tqdm(total=self.max_steps, initial=self.step)
        
        while self.step < self.max_steps:
            for batch in self.train_loader:
                if self.step >= self.max_steps:
                    break
                
                loss = self.train_step(batch)
                
                if self.step % self.log_interval == 0:
                    if use_wandb:
                        wandb.log({
                            "train_loss": loss,
                            "learning_rate": self.scheduler.get_last_lr()[0],
                            "step": self.step
                        })
                    else:
                        print(f"Step {self.step}: loss={loss:.4f}, lr={self.scheduler.get_last_lr()[0]:.2e}")
                
                if self.step % self.eval_interval == 0:
                    val_loss = self.validate()
                    if use_wandb:
                        wandb.log({"val_loss": val_loss, "step": self.step})
                    else:
                        print(f"Validation: loss={val_loss:.4f}")
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best_pretrain_model.pt")
                
                if self.step % self.save_interval == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.step}.pt")
                
                self.step += 1
                progress_bar.update(1)
        
        wandb.finish()
        self.save_checkpoint("final_pretrain_model.pt")

class TranslationTrainer(BaseTrainer):
    """
    Trainer for English-Romanian translation fine-tuning
    """
    
    def __init__(
        self,
        model: MultilingualTranslationModel,
        train_loader,
        val_loader,
        device: torch.device,
        output_dir: str,
        config: Dict,
    ):
        training_config = config.get("training", {})
        optimizer = AdamW(
            model.parameters(),
            lr=training_config.get("learning_rate", 3e-5),
            weight_decay=training_config.get("weight_decay", 0.01),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        max_epochs = training_config.get("max_epochs", 3)
        warmup_steps = training_config.get("warmup_steps", 500)
        total_steps = len(train_loader) * max_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_dir=output_dir,
        )
        
        self.max_epochs = config.get("training", {}).get("max_epochs", 3)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        loss = outputs["loss"]
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def validate(self) -> float:
        """Validation step"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                total_loss += outputs["loss"].item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self):
        """Training loop"""
        print(f"Starting fine-tuning for {self.max_epochs} epochs...")
        
        try:
            import wandb
            wandb.init(
                project="mbart-translation",
                name=f"en_ro_translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            use_wandb = True
        except Exception:
            print("⚠️  wandb not available, using console logging only")
            use_wandb = False
        
        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            epoch_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.max_epochs}")
            
            for batch in progress_bar:
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                
                if self.step % self.log_interval == 0:
                    wandb.log({
                        "train_loss": loss,
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "step": self.step
                    })
                
                progress_bar.set_postfix({"loss": loss})
                self.step += 1
            
            # Validation
            val_loss = self.validate()
            avg_train_loss = epoch_loss / num_batches
            
            wandb.log({
                "epoch_train_loss": avg_train_loss,
                "epoch_val_loss": val_loss,
                "epoch": epoch
            })
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_translation_model.pt")
            
            self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
        
        if use_wandb:
            wandb.finish()
        self.save_checkpoint("final_translation_model.pt")

def run_pretraining(config_path: str = "configs/pretrain_config.json"):
    """
    Run denoising pre-training
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    output_config = config.get("output", {})
    
    # Initialize components
    processor = DataProcessor()
    train_loader = processor.create_pretrain_dataloaders(
        batch_size=training_config.get("batch_size", 8),
        max_length=data_config.get("max_length", 512),
        num_samples=data_config.get("num_samples", 100)
    )
    val_loader = processor.create_pretrain_dataloaders(
        batch_size=training_config.get("batch_size", 8),
        max_length=data_config.get("max_length", 512),
        num_samples=data_config.get("num_samples", 20)
    )
    
    # Initialize model
    from transformers import MBartConfig
    processor = DataProcessor()
    vocab_size = len(processor.tokenizer)
    mbart_config = MBartConfig(
        vocab_size=vocab_size,
        d_model=model_config.get("d_model", 1024),
        encoder_layers=model_config.get("encoder_layers", 12),
        decoder_layers=model_config.get("decoder_layers", 12),
        encoder_attention_heads=model_config.get("encoder_attention_heads", 16),
        decoder_attention_heads=model_config.get("decoder_attention_heads", 16),
        encoder_ffn_dim=model_config.get("encoder_ffn_dim", 4096),
        decoder_ffn_dim=model_config.get("decoder_ffn_dim", 4096),
        max_position_embeddings=model_config.get("max_position_embeddings", 1024),
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        bos_token_id=processor.tokenizer.bos_token_id,
        dropout=model_config.get("dropout", 0.1),
        attention_dropout=model_config.get("attention_dropout", 0.1),
        activation_dropout=model_config.get("activation_dropout", 0.0),
    )
    
    model = MultilingualDenoisingPretraining(mbart_config)
    noise_generator = NoiseGenerator(processor.tokenizer)
    
    # Initialize trainer
    trainer = DenoisingPretrainingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_config.get("output_dir", "checkpoints/pretrain"),
        config=config,
        noise_generator=noise_generator
    )
    
    trainer.train()

def run_finetuning(config_path: str = "configs/translation_config.json"):
    """
    Run English-Romanian translation fine-tuning
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    output_config = config.get("output", {})
    model_config = config.get("model", {})
    
    # Initialize components
    processor = DataProcessor()
    train_loader, val_loader, test_loader = processor.create_dataloaders(
        batch_size=training_config.get("batch_size", 8),
        max_length=data_config.get("max_length", 128),
        data_size=data_config.get("train_size", 1000)  # Small dataset for testing
    )
    
    # Initialize model
    pretrained_model = model_config.get("pretrained_model", "facebook/mbart-large-50")
    model = MultilingualTranslationModel(pretrained_model)
    
    # Initialize trainer
    trainer = TranslationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_config.get("output_dir", "checkpoints/translation"),
        config=config
    )
    
    trainer.train()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pretrain", "finetune"], default="finetune")
    parser.add_argument("--config", type=str, help="Path to config file")
    
    args = parser.parse_args()
    
    if args.mode == "pretrain":
        config_path = args.config or "configs/pretrain_config.json"
        run_pretraining(config_path=config_path)
    else:
        config_path = args.config or "configs/translation_config.json"
        run_finetuning(config_path=config_path)