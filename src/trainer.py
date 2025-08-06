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
        learning_rate: float = 5e-4,
        warmup_steps: int = 1000,
        max_steps: int = 10000,
        weight_decay: float = 0.01,
        noise_generator: Optional[NoiseGenerator] = None,
    ):
        self.noise_generator = noise_generator
        
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6
        )
        
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
        
        self.max_steps = max_steps
    
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
        
        wandb.init(
            project="mbart-pretraining",
            name=f"denoising_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        progress_bar = tqdm(total=self.max_steps, initial=self.step)
        
        while self.step < self.max_steps:
            for batch in self.train_loader:
                if self.step >= self.max_steps:
                    break
                
                loss = self.train_step(batch)
                
                if self.step % self.log_interval == 0:
                    wandb.log({
                        "train_loss": loss,
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "step": self.step
                    })
                
                if self.step % self.eval_interval == 0:
                    val_loss = self.validate()
                    wandb.log({"val_loss": val_loss, "step": self.step})
                    
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
        learning_rate: float = 3e-5,
        warmup_steps: int = 500,
        max_epochs: int = 3,
        weight_decay: float = 0.01,
    ):
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
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
        
        self.max_epochs = max_epochs
    
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
        
        wandb.init(
            project="mbart-translation",
            name=f"en_ro_translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
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
        
        wandb.finish()
        self.save_checkpoint("final_translation_model.pt")

def run_pretraining(
    batch_size: int = 8,
    max_steps: int = 5000,
    learning_rate: float = 5e-4,
    warmup_steps: int = 500,
    max_length: int = 512,
    output_dir: str = "checkpoints/pretrain"
):
    """
    Run denoising pre-training
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize components
    processor = DataProcessor()
    train_loader = processor.create_pretrain_dataloaders(
        batch_size=batch_size,
        max_length=max_length,
        num_samples=10000
    )
    val_loader = processor.create_pretrain_dataloaders(
        batch_size=batch_size,
        max_length=max_length,
        num_samples=1000
    )
    
    # Initialize model
    from transformers import MBartConfig
    config = MBartConfig(
        vocab_size=250027,
        d_model=1024,
        encoder_layers=12,
        decoder_layers=12,
        encoder_attention_heads=16,
        decoder_attention_heads=16,
        encoder_ffn_dim=4096,
        decoder_ffn_dim=4096,
        max_position_embeddings=1024,
    )
    
    model = MultilingualDenoisingPretraining(config)
    noise_generator = NoiseGenerator(processor.tokenizer)
    
    # Initialize trainer
    trainer = DenoisingPretrainingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        noise_generator=noise_generator
    )
    
    trainer.train()

def run_finetuning(
    batch_size: int = 8,
    max_epochs: int = 3,
    learning_rate: float = 3e-5,
    warmup_steps: int = 500,
    max_length: int = 128,
    output_dir: str = "checkpoints/translation"
):
    """
    Run English-Romanian translation fine-tuning
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize components
    processor = DataProcessor()
    train_loader, val_loader, test_loader = processor.create_dataloaders(
        batch_size=batch_size,
        max_length=max_length
    )
    
    # Initialize model
    model = MultilingualTranslationModel("facebook/mbart-large-50")
    
    # Initialize trainer
    trainer = TranslationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        max_epochs=max_epochs
    )
    
    trainer.train()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pretrain", "finetune"], default="finetune")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--max_epochs", type=int, default=3)
    
    args = parser.parse_args()
    
    if args.mode == "pretrain":
        run_pretraining(
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate
        )
    else:
        run_finetuning(
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate
        )