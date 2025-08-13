"""
Pytest configuration and fixtures for mBART seq-to-seq tests
"""

import pytest
import torch
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import MagicMock

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.model import MultilingualDenoisingPretraining, MultilingualTranslationModel, NoiseGenerator
from src.data import DataProcessor, TranslationDataset, DenoisingPretrainDataset
from src.evaluation import TranslationEvaluator
from transformers import MBartConfig


@pytest.fixture
def device():
    """Get appropriate device for testing"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_mbart_config():
    """Small mBART config for testing"""
    return MBartConfig(
        vocab_size=1000,
        d_model=128,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=256,
        decoder_ffn_dim=256,
        max_position_embeddings=128,
        pad_token_id=1,
        eos_token_id=2,
        bos_token_id=0,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
    )


@pytest.fixture
def test_config():
    """Test configuration dictionary"""
    return {
        "model": {
            "vocab_size": 1000,
            "d_model": 128,
            "encoder_layers": 2,
            "decoder_layers": 2,
            "encoder_attention_heads": 4,
            "decoder_attention_heads": 4,
            "encoder_ffn_dim": 256,
            "decoder_ffn_dim": 256,
            "max_position_embeddings": 128,
            "dropout": 0.1,
            "attention_dropout": 0.0,
            "activation_dropout": 0.0
        },
        "training": {
            "batch_size": 2,
            "max_steps": 10,
            "max_epochs": 1,
            "learning_rate": 1e-4,
            "warmup_steps": 2,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "save_interval": 5,
            "eval_interval": 5,
            "log_interval": 2
        },
        "data": {
            "max_length": 64,
            "num_samples": 10,
            "languages": ["en_XX", "ro_RO"],
            "noise_types": ["span_masking"],
            "mask_prob": 0.15,
            "poisson_lambda": 2.0
        },
        "output": {
            "output_dir": "test_checkpoints",
            "log_dir": "test_logs"
        }
    }


@pytest.fixture
def sample_translation_data():
    """Sample English-Romanian translation data"""
    return [
        {"source": "Hello world", "target": "Salut lume"},
        {"source": "Good morning", "target": "Bună dimineața"},
        {"source": "Thank you", "target": "Mulțumesc"},
        {"source": "How are you?", "target": "Ce mai faci?"},
        {"source": "See you later", "target": "Ne vedem mai târziu"}
    ]


@pytest.fixture
def sample_monolingual_data():
    """Sample monolingual data for pretraining"""
    return [
        "This is a test sentence in English.",
        "Another English sentence for testing.",
        "Machine learning is fascinating.",
        "Aceasta este o propoziție de test în română.",
        "O altă propoziție românească pentru testare.",
        "Învățarea automată este fascinantă."
    ]


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing"""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.bos_token_id = 0
    tokenizer.mask_token_id = 50264
    tokenizer.lang_code_to_id = {"en_XX": 250004, "ro_RO": 250020}
    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = "ro_RO"
    
    # Mock tokenization
    def mock_call(*args, **kwargs):
        text = args[0] if args else ""
        if isinstance(text, str):
            tokens = [0, 100, 200, 2, 1, 1]  # Mock token sequence
        else:  # list of strings
            tokens = [[0, 100, 200, 2, 1, 1] for _ in text]
        
        if kwargs.get("return_tensors") == "pt":
            if isinstance(tokens[0], list):
                return {
                    "input_ids": torch.tensor(tokens),
                    "attention_mask": torch.ones(len(tokens), len(tokens[0]))
                }
            else:
                return {
                    "input_ids": torch.tensor([tokens]),
                    "attention_mask": torch.ones(1, len(tokens))
                }
        return {"input_ids": tokens}
    
    def mock_decode(token_ids, **kwargs):
        if hasattr(token_ids, 'tolist'):
            if token_ids.dim() > 1:
                return ["Mock translation" for _ in range(token_ids.size(0))]
            else:
                return "Mock translation"
        return "Mock translation"
    
    def mock_batch_decode(token_ids, **kwargs):
        return ["Mock translation" for _ in range(len(token_ids))]
    
    tokenizer.side_effect = mock_call
    tokenizer.__call__ = mock_call
    tokenizer.decode = mock_decode
    tokenizer.batch_decode = mock_batch_decode
    tokenizer.__len__ = lambda: 1000
    
    return tokenizer


@pytest.fixture
def data_processor(mock_tokenizer):
    """DataProcessor instance with mock tokenizer"""
    processor = DataProcessor.__new__(DataProcessor)
    processor.tokenizer = mock_tokenizer
    return processor


@pytest.fixture
def temp_config_file(test_config):
    """Temporary config file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_checkpoint_dir():
    """Temporary directory for checkpoints"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_batch():
    """Sample training batch"""
    batch_size = 2
    seq_len = 10
    
    return {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "labels": torch.randint(0, 1000, (batch_size, seq_len)),
        "decoder_input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "decoder_attention_mask": torch.ones(batch_size, seq_len)
    }


@pytest.fixture
def pretrain_model(small_mbart_config):
    """Small pretraining model for testing"""
    return MultilingualDenoisingPretraining(small_mbart_config)


@pytest.fixture
def noise_generator(mock_tokenizer):
    """NoiseGenerator instance"""
    return NoiseGenerator(mock_tokenizer)


class MockModel:
    """Mock model for testing"""
    def __init__(self):
        self.training = True
        
    def train(self):
        self.training = True
        
    def eval(self):
        self.training = False
        
    def parameters(self):
        return [torch.randn(10, 10, requires_grad=True)]
        
    def state_dict(self):
        return {"test_param": torch.randn(10, 10)}
        
    def load_state_dict(self, state_dict):
        pass
        
    def to(self, device):
        return self
        
    def forward(self, **kwargs):
        batch_size = kwargs.get("input_ids", torch.tensor([[1]])).size(0)
        return {
            "loss": torch.tensor(0.5, requires_grad=True),
            "logits": torch.randn(batch_size, 10, 1000)
        }
        
    def generate(self, **kwargs):
        batch_size = kwargs.get("input_ids", torch.tensor([[1]])).size(0)
        return torch.randint(0, 1000, (batch_size, 20))


@pytest.fixture
def mock_model():
    """Mock model for testing"""
    return MockModel()


# Test utilities
def assert_tensor_close(actual, expected, rtol=1e-5, atol=1e-8):
    """Assert two tensors are close"""
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), \
        f"Tensors not close: {actual} vs {expected}"


def assert_tensor_shape(tensor, expected_shape):
    """Assert tensor has expected shape"""
    assert tensor.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {tensor.shape}"


def create_mock_dataset(size=10, seq_len=20):
    """Create mock dataset for testing"""
    data = []
    for i in range(size):
        data.append({
            "source": f"Test source sentence {i}",
            "target": f"Test target sentence {i}"
        })
    return data


# Parametrized fixtures for different test scenarios
@pytest.fixture(params=["cpu", "cuda"])
def all_devices(request):
    """Parametrize tests across available devices"""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


@pytest.fixture(params=[1, 2, 4])
def batch_sizes(request):
    """Parametrize tests across different batch sizes"""
    return request.param


@pytest.fixture(params=[16, 32, 64])
def sequence_lengths(request):
    """Parametrize tests across different sequence lengths"""
    return request.param