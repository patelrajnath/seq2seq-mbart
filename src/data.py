import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MBart50TokenizerFast
from typing import List, Dict, Tuple, Optional
import pandas as pd
from datasets import load_dataset
import random
import numpy as np

class TranslationDataset(Dataset):
    """
    Dataset for English-Romanian translation
    """
    
    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: MBart50TokenizerFast,
        max_length: int = 128,
        src_lang: str = "en_XX",
        tgt_lang: str = "ro_RO"
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # Set source and target languages
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Tokenize source
        source = self.tokenizer(
            item["source"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target
        target = self.tokenizer(
            item["target"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create decoder_input_ids by shifting labels right and adding decoder_start_token_id
        labels = target["input_ids"].squeeze(0)
        decoder_input_ids = torch.cat([
            torch.tensor([self.tokenizer.lang_code_to_id[self.tgt_lang]]),
            labels[:-1]
        ])
        
        return {
            "input_ids": source["input_ids"].squeeze(0),
            "attention_mask": source["attention_mask"].squeeze(0),
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": target["attention_mask"].squeeze(0)
        }

class DenoisingPretrainDataset(Dataset):
    """
    Dataset for multilingual denoising pre-training
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: MBart50TokenizerFast,
        max_length: int = 512,
        languages: List[str] = None
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.languages = languages or ["en_XX", "ro_RO"]
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # Randomly select language
        lang = random.choice(self.languages)
        
        # Add language token
        text = f"{lang} {text}"
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # For denoising, input and target are the same
        labels = encoded["input_ids"].squeeze(0)
        
        # Create decoder_input_ids by shifting labels right and adding decoder_start_token_id
        # Use the language token as start token for denoising
        lang_id = self.tokenizer.lang_code_to_id[lang]
        decoder_input_ids = torch.cat([
            torch.tensor([lang_id]),
            labels[:-1]
        ])
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": encoded["attention_mask"].squeeze(0)
        }

class DataProcessor:
    """
    Data processing utilities for English-Romanian translation
    """
    
    def __init__(self, tokenizer_name: str = "facebook/mbart-large-50-many-to-many-mmt"):
        self.tokenizer = MBart50TokenizerFast.from_pretrained(tokenizer_name)
    
    def load_wmt_en_ro(self, split: str = "train") -> List[Dict[str, str]]:
        """
        Load WMT English-Romanian dataset
        """
        try:
            dataset = load_dataset("wmt16", "ro-en", split=split)
            
            data = []
            for item in dataset:
                data.append({
                    "source": item["translation"]["en"],
                    "target": item["translation"]["ro"]
                })
            
            return data
        except Exception as e:
            print(f"Error loading WMT dataset: {e}")
            # Fallback to smaller dataset
            return self.load_ted_talks(split)
    
    def load_ted_talks(self, split: str = "train") -> List[Dict[str, str]]:
        """
        Load TED talks dataset as fallback
        """
        try:
            dataset = load_dataset("ted_talks_iwslt", "en_ro", split=split)
            
            data = []
            for item in dataset:
                data.append({
                    "source": item["translation"]["en"],
                    "target": item["translation"]["ro"]
                })
            
            return data
        except Exception as e:
            print(f"Error loading TED talks: {e}")
            # Return dummy data for testing
            return self._create_dummy_data()
    
    def load_monolingual_data(self, lang: str = "en", num_samples: int = 10000) -> List[str]:
        """
        Load monolingual data for pre-training
        """
        try:
            if lang == "en":
                dataset = load_dataset("cc100", lang="en", split="train", streaming=True)
            elif lang == "ro":
                dataset = load_dataset("cc100", lang="ro", split="train", streaming=True)
            else:
                dataset = load_dataset("cc100", lang="en", split="train", streaming=True)
            
            texts = []
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                texts.append(item["text"])
            
            return texts
        except Exception as e:
            print(f"Error loading monolingual data: {e}")
            return self._create_dummy_monolingual_data(lang, num_samples)
    
    def _create_dummy_data(self) -> List[Dict[str, str]]:
        """
        Create dummy data for testing
        """
        return [
            {"source": "Hello, how are you?", "target": "Salut, ce mai faci?"},
            {"source": "I love machine learning.", "target": "Îmi place învățarea automată."},
            {"source": "This is a test sentence.", "target": "Aceasta este o propoziție de test."},
            {"source": "The weather is nice today.", "target": "Vremea este frumoasă astăzi."},
            {"source": "Thank you very much.", "target": "Mulțumesc foarte mult."}
        ]
    
    def _create_dummy_monolingual_data(self, lang: str, num_samples: int) -> List[str]:
        """
        Create dummy monolingual data for testing
        """
        if lang == "ro":
            texts = [
                "Aceasta este o propoziție de test în limba română.",
                "Vremea este frumoasă astăzi în București.",
                "Îmi place să învăț despre inteligența artificială.",
                "România este o țară frumoasă din Europa de Est.",
                "Limba română este o limbă romanică."
            ]
        else:
            texts = [
                "This is a test sentence in English.",
                "The weather is nice today in London.",
                "I love learning about artificial intelligence.",
                "England is a beautiful country in Western Europe.",
                "English is a Germanic language."
            ]
        
        return texts * (num_samples // len(texts) + 1)[:num_samples]
    
    def create_dataloaders(
        self,
        batch_size: int = 8,
        max_length: int = 128,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test dataloaders
        """
        # Load datasets
        train_data = self.load_wmt_en_ro("train")
        val_data = self.load_wmt_en_ro("validation")
        test_data = self.load_wmt_en_ro("test")
        
        # Limit data sizes for faster experimentation
        train_data = train_data[:10000]
        val_data = val_data[:1000]
        test_data = test_data[:1000]
        
        # Create datasets
        train_dataset = TranslationDataset(
            train_data, self.tokenizer, max_length=max_length
        )
        val_dataset = TranslationDataset(
            val_data, self.tokenizer, max_length=max_length
        )
        test_dataset = TranslationDataset(
            test_data, self.tokenizer, max_length=max_length
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def create_pretrain_dataloaders(
        self,
        batch_size: int = 8,
        max_length: int = 512,
        num_samples: int = 10000,
        num_workers: int = 4
    ) -> DataLoader:
        """
        Create pre-training dataloader
        """
        # Load monolingual data
        en_texts = self.load_monolingual_data("en", num_samples // 2)
        ro_texts = self.load_monolingual_data("ro", num_samples // 2)
        
        all_texts = en_texts + ro_texts
        random.shuffle(all_texts)
        
        # Create dataset
        dataset = DenoisingPretrainDataset(
            all_texts, self.tokenizer, max_length=max_length
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return dataloader