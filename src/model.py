import torch
import torch.nn as nn
from transformers import MBartConfig, MBartForConditionalGeneration, MBart50TokenizerFast, MBartTokenizer
from typing import Dict, List, Optional, Tuple

class MultilingualDenoisingPretraining(nn.Module):
    """
    Implementation of multilingual denoising pre-training as described in mBART paper
    """
    
    def __init__(self, config: MBartConfig):
        super().__init__()
        self.config = config
        self.model = MBartForConditionalGeneration(config)
        self.tokenizer = None
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for denoising pre-training
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits,
            'last_hidden_state': outputs.encoder_last_hidden_state
        }

class MultilingualTranslationModel(nn.Module):
    """
    Fine-tuned model for English-Romanian translation
    """
    
    def __init__(self, pretrained_model_name: str = "facebook/mbart-large-50-many-to-many-mmt"):
        super().__init__()
        self.model = MBartForConditionalGeneration.from_pretrained(pretrained_model_name)
        try:
            self.tokenizer = MBart50TokenizerFast.from_pretrained(pretrained_model_name)
        except Exception as e:
            print(f"Warning: MBart50TokenizerFast failed, trying MBartTokenizer: {e}")
            self.tokenizer = MBartTokenizer.from_pretrained(pretrained_model_name)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor = None,
        decoder_attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for translation
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            return_dict=True
        )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
    ) -> torch.Tensor:
        """
        Generate translations
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
        )

class NoiseGenerator:
    """
    Implements noise functions for denoising pre-training
    """
    
    def __init__(self, tokenizer, mask_prob: float = 0.35, poisson_lambda: float = 3.5):
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.poisson_lambda = poisson_lambda
        self.mask_token_id = tokenizer.mask_token_id
        
    def span_masking(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply span masking as described in the paper
        """
        batch_size, seq_len = input_ids.shape
        masked_input = input_ids.clone()
        
        for i in range(batch_size):
            valid_length = (input_ids[i] != self.tokenizer.pad_token_id).sum().item()
            position = 1  # Skip <s> token
            
            while position < valid_length:
                span_length = min(
                    torch.poisson(torch.tensor(float(self.poisson_lambda))).item(),
                    valid_length - position
                )
                span_length = max(int(span_length), 1)
                
                if torch.rand(1).item() < self.mask_prob:
                    masked_input[i, position:position + span_length] = self.mask_token_id
                
                position += span_length
                
        return masked_input
    
    def sentence_permutation(self, input_ids: torch.Tensor, sentence_end_token: int = 2) -> torch.Tensor:
        """
        Apply sentence permutation
        """
        batch_size, seq_len = input_ids.shape
        permuted_input = input_ids.clone()
        
        for i in range(batch_size):
            # Find sentence boundaries
            sentence_ends = (input_ids[i] == sentence_end_token).nonzero().flatten()
            if len(sentence_ends) == 0:
                continue
                
            sentences = []
            start = 1  # Skip <s> token
            
            for end in sentence_ends:
                sentences.append(input_ids[i, start:end+1])
                start = end + 1
            
            # Shuffle sentences
            indices = torch.randperm(len(sentences))
            
            # Reconstruct sequence
            new_sequence = [input_ids[i, 0].unsqueeze(0)]  # <s> token
            for idx in indices:
                new_sequence.append(sentences[idx])
            
            new_sequence = torch.cat(new_sequence)
            
            # Pad or truncate to original length
            if len(new_sequence) < seq_len:
                padding = torch.full((seq_len - len(new_sequence),), self.tokenizer.pad_token_id)
                new_sequence = torch.cat([new_sequence, padding])
            else:
                new_sequence = new_sequence[:seq_len]
                
            permuted_input[i] = new_sequence
            
        return permuted_input
    
    def add_noise(self, input_ids: torch.Tensor, noise_type: str = "span_masking") -> torch.Tensor:
        """
        Add noise to input sequences
        """
        if noise_type == "span_masking":
            return self.span_masking(input_ids)
        elif noise_type == "sentence_permutation":
            return self.sentence_permutation(input_ids)
        elif noise_type == "both":
            masked = self.span_masking(input_ids)
            return self.sentence_permutation(masked)
        else:
            return input_ids