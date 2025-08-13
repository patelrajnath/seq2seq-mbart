"""
Unit tests for data processing components
"""

import pytest
import torch
import json
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader

from src.data import TranslationDataset, DenoisingPretrainDataset, DataProcessor


class TestTranslationDataset:
    """Test TranslationDataset class"""

    @pytest.mark.unit
    def test_initialization(self, sample_translation_data, mock_tokenizer):
        """Test dataset initialization"""
        dataset = TranslationDataset(
            data=sample_translation_data,
            tokenizer=mock_tokenizer,
            max_length=128
        )
        
        assert len(dataset) == len(sample_translation_data)
        assert dataset.tokenizer == mock_tokenizer
        assert dataset.max_length == 128
        assert dataset.src_lang == "en_XX"
        assert dataset.tgt_lang == "ro_RO"

    @pytest.mark.unit
    def test_initialization_custom_languages(self, sample_translation_data, mock_tokenizer):
        """Test dataset initialization with custom languages"""
        dataset = TranslationDataset(
            data=sample_translation_data,
            tokenizer=mock_tokenizer,
            src_lang="fr_XX",
            tgt_lang="de_DE"
        )
        
        assert dataset.src_lang == "fr_XX"
        assert dataset.tgt_lang == "de_DE"

    @pytest.mark.unit
    def test_len(self, sample_translation_data, mock_tokenizer):
        """Test __len__ method"""
        dataset = TranslationDataset(sample_translation_data, mock_tokenizer)
        assert len(dataset) == 5

    @pytest.mark.unit
    def test_getitem(self, sample_translation_data, mock_tokenizer):
        """Test __getitem__ method"""
        dataset = TranslationDataset(sample_translation_data, mock_tokenizer)
        
        item = dataset[0]
        
        # Check required keys
        required_keys = ["input_ids", "attention_mask", "labels", 
                        "decoder_input_ids", "decoder_attention_mask"]
        for key in required_keys:
            assert key in item
            assert isinstance(item[key], torch.Tensor)

    @pytest.mark.unit
    def test_getitem_tensor_shapes(self, sample_translation_data, mock_tokenizer):
        """Test tensor shapes in __getitem__"""
        max_length = 64
        dataset = TranslationDataset(sample_translation_data, mock_tokenizer, max_length=max_length)
        
        item = dataset[0]
        
        # All tensors should have same length (max_length)
        assert item["input_ids"].shape[0] == max_length
        assert item["attention_mask"].shape[0] == max_length
        assert item["labels"].shape[0] == max_length
        assert item["decoder_input_ids"].shape[0] == max_length
        assert item["decoder_attention_mask"].shape[0] == max_length

    @pytest.mark.unit
    def test_tokenizer_language_settings(self, sample_translation_data, mock_tokenizer):
        """Test that tokenizer languages are set correctly"""
        dataset = TranslationDataset(
            sample_translation_data, 
            mock_tokenizer,
            src_lang="fr_XX",
            tgt_lang="de_DE"
        )
        
        # Access item to trigger tokenizer usage
        _ = dataset[0]
        
        assert mock_tokenizer.src_lang == "fr_XX"
        assert mock_tokenizer.tgt_lang == "de_DE"

    @pytest.mark.unit
    def test_empty_dataset(self, mock_tokenizer):
        """Test empty dataset handling"""
        empty_data = []
        dataset = TranslationDataset(empty_data, mock_tokenizer)
        
        assert len(dataset) == 0

    @pytest.mark.unit 
    def test_single_item_dataset(self, mock_tokenizer):
        """Test dataset with single item"""
        single_data = [{"source": "Hello", "target": "Salut"}]
        dataset = TranslationDataset(single_data, mock_tokenizer)
        
        assert len(dataset) == 1
        item = dataset[0]
        assert "input_ids" in item

    @pytest.mark.unit
    @pytest.mark.parametrize("max_length", [32, 64, 128, 256])
    def test_different_max_lengths(self, sample_translation_data, mock_tokenizer, max_length):
        """Test dataset with different max lengths"""
        dataset = TranslationDataset(sample_translation_data, mock_tokenizer, max_length=max_length)
        item = dataset[0]
        
        assert item["input_ids"].shape[0] == max_length
        assert item["labels"].shape[0] == max_length


class TestDenoisingPretrainDataset:
    """Test DenoisingPretrainDataset class"""

    @pytest.mark.unit
    def test_initialization(self, sample_monolingual_data, mock_tokenizer):
        """Test dataset initialization"""
        dataset = DenoisingPretrainDataset(
            texts=sample_monolingual_data,
            tokenizer=mock_tokenizer,
            max_length=512
        )
        
        assert len(dataset) == len(sample_monolingual_data)
        assert dataset.tokenizer == mock_tokenizer
        assert dataset.max_length == 512
        assert dataset.languages == ["en_XX", "ro_RO"]

    @pytest.mark.unit
    def test_initialization_custom_languages(self, sample_monolingual_data, mock_tokenizer):
        """Test dataset initialization with custom languages"""
        languages = ["fr_XX", "de_DE", "es_XX"]
        dataset = DenoisingPretrainDataset(
            sample_monolingual_data, 
            mock_tokenizer,
            languages=languages
        )
        
        assert dataset.languages == languages

    @pytest.mark.unit
    def test_len(self, sample_monolingual_data, mock_tokenizer):
        """Test __len__ method"""
        dataset = DenoisingPretrainDataset(sample_monolingual_data, mock_tokenizer)
        assert len(dataset) == 6

    @pytest.mark.unit
    def test_getitem(self, sample_monolingual_data, mock_tokenizer):
        """Test __getitem__ method"""
        dataset = DenoisingPretrainDataset(sample_monolingual_data, mock_tokenizer)
        
        item = dataset[0]
        
        # Check required keys
        required_keys = ["input_ids", "attention_mask", "labels", 
                        "decoder_input_ids", "decoder_attention_mask"]
        for key in required_keys:
            assert key in item
            assert isinstance(item[key], torch.Tensor)

    @pytest.mark.unit
    def test_getitem_random_language_selection(self, sample_monolingual_data, mock_tokenizer):
        """Test that random language selection works"""
        dataset = DenoisingPretrainDataset(
            sample_monolingual_data, 
            mock_tokenizer,
            languages=["en_XX", "ro_RO", "fr_XX"]
        )
        
        # Get multiple items to test randomness
        languages_used = set()
        for _ in range(10):
            # Mock random.choice to cycle through languages
            with patch('src.data.random.choice') as mock_choice:
                mock_choice.return_value = "en_XX"
                item = dataset[0]
                languages_used.add("en_XX")
        
        # At least one language should be used
        assert len(languages_used) >= 1

    @pytest.mark.unit
    def test_getitem_tensor_shapes(self, sample_monolingual_data, mock_tokenizer):
        """Test tensor shapes in __getitem__"""
        max_length = 256
        dataset = DenoisingPretrainDataset(sample_monolingual_data, mock_tokenizer, max_length=max_length)
        
        item = dataset[0]
        
        # All tensors should have same length (max_length)
        assert item["input_ids"].shape[0] == max_length
        assert item["attention_mask"].shape[0] == max_length
        assert item["labels"].shape[0] == max_length
        assert item["decoder_input_ids"].shape[0] == max_length
        assert item["decoder_attention_mask"].shape[0] == max_length

    @pytest.mark.unit
    def test_empty_dataset(self, mock_tokenizer):
        """Test empty dataset handling"""
        empty_data = []
        dataset = DenoisingPretrainDataset(empty_data, mock_tokenizer)
        
        assert len(dataset) == 0


class TestDataProcessor:
    """Test DataProcessor class"""

    @pytest.mark.unit
    @patch('src.data.MBart50TokenizerFast.from_pretrained')
    def test_initialization_success(self, mock_tokenizer_class):
        """Test successful initialization"""
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.return_value = mock_tokenizer
        
        processor = DataProcessor()
        
        assert processor.tokenizer == mock_tokenizer
        mock_tokenizer_class.assert_called_once_with("facebook/mbart-large-50-many-to-many-mmt")

    @pytest.mark.unit
    @patch('src.data.MBart50TokenizerFast.from_pretrained')
    @patch('src.data.MBartTokenizer.from_pretrained')
    def test_tokenizer_fallback(self, mock_fallback_tokenizer, mock_fast_tokenizer):
        """Test tokenizer fallback mechanism"""
        mock_fast_tokenizer.side_effect = Exception("Fast tokenizer failed")
        mock_fallback = MagicMock()
        mock_fallback_tokenizer.return_value = mock_fallback
        
        processor = DataProcessor()
        
        assert processor.tokenizer == mock_fallback
        mock_fallback_tokenizer.assert_called_once()

    @pytest.mark.unit
    @patch('src.data.load_dataset')
    def test_load_wmt_en_ro_success(self, mock_load_dataset, data_processor):
        """Test successful WMT dataset loading"""
        # Mock dataset structure
        mock_dataset = [
            {"translation": {"en": "Hello", "ro": "Salut"}},
            {"translation": {"en": "Goodbye", "ro": "La revedere"}}
        ]
        mock_load_dataset.return_value = mock_dataset
        
        data = data_processor.load_wmt_en_ro("train")
        
        assert len(data) == 2
        assert data[0]["source"] == "Hello"
        assert data[0]["target"] == "Salut"
        mock_load_dataset.assert_called_once_with("wmt16", "ro-en", split="train")

    @pytest.mark.unit
    @patch('src.data.load_dataset')
    def test_load_wmt_en_ro_fallback_to_ted_talks(self, mock_load_dataset, data_processor):
        """Test fallback to TED talks dataset"""
        # First call (WMT) fails
        mock_load_dataset.side_effect = [
            Exception("WMT loading failed"),
            [{"translation": {"en": "Test", "ro": "Test"}}]  # TED talks succeeds
        ]
        
        data = data_processor.load_wmt_en_ro("train")
        
        assert len(data) == 1
        assert mock_load_dataset.call_count == 2

    @pytest.mark.unit
    @patch('src.data.load_dataset')
    def test_load_wmt_en_ro_fallback_to_dummy(self, mock_load_dataset, data_processor):
        """Test fallback to dummy data"""
        # Both WMT and TED talks fail
        mock_load_dataset.side_effect = Exception("Loading failed")
        
        data = data_processor.load_wmt_en_ro("train")
        
        # Should return dummy data
        assert len(data) > 0
        assert all("source" in item and "target" in item for item in data)

    @pytest.mark.unit
    def test_create_dummy_data(self, data_processor):
        """Test dummy data creation"""
        dummy_data = data_processor._create_dummy_data()
        
        assert len(dummy_data) == 5
        assert all("source" in item and "target" in item for item in dummy_data)
        assert all(isinstance(item["source"], str) and isinstance(item["target"], str) 
                  for item in dummy_data)

    @pytest.mark.unit
    def test_create_dummy_monolingual_data_english(self, data_processor):
        """Test dummy monolingual data creation for English"""
        data = data_processor._create_dummy_monolingual_data("en", 10)
        
        assert len(data) == 10
        assert all(isinstance(text, str) for text in data)
        # Should contain English text
        assert any("English" in text for text in data)

    @pytest.mark.unit
    def test_create_dummy_monolingual_data_romanian(self, data_processor):
        """Test dummy monolingual data creation for Romanian"""
        data = data_processor._create_dummy_monolingual_data("ro", 8)
        
        assert len(data) == 8
        assert all(isinstance(text, str) for text in data)
        # Should contain Romanian text
        assert any("română" in text for text in data)

    @pytest.mark.unit
    @patch('src.data.load_dataset')
    def test_load_monolingual_data_success(self, mock_load_dataset, data_processor):
        """Test successful monolingual data loading"""
        mock_dataset = [
            {"text": "First sentence."},
            {"text": "Second sentence."},
            {"text": "Third sentence."}
        ]
        mock_load_dataset.return_value = mock_dataset
        
        data = data_processor.load_monolingual_data("en", 2)
        
        assert len(data) == 2
        assert data[0] == "First sentence."
        assert data[1] == "Second sentence."

    @pytest.mark.unit
    @patch('src.data.load_dataset')
    def test_load_monolingual_data_fallback(self, mock_load_dataset, data_processor):
        """Test monolingual data fallback to dummy data"""
        mock_load_dataset.side_effect = Exception("Loading failed")
        
        data = data_processor.load_monolingual_data("en", 5)
        
        assert len(data) == 5
        # Should be dummy data
        assert all(isinstance(text, str) for text in data)

    @pytest.mark.unit
    def test_create_dataloaders(self, data_processor):
        """Test dataloader creation"""
        with patch.object(data_processor, 'load_wmt_en_ro') as mock_load:
            # Mock data loading
            mock_load.side_effect = [
                [{"source": f"src{i}", "target": f"tgt{i}"} for i in range(100)],  # train
                [{"source": f"src{i}", "target": f"tgt{i}"} for i in range(20)],   # val
                [{"source": f"src{i}", "target": f"tgt{i}"} for i in range(20)]    # test
            ]
            
            train_loader, val_loader, test_loader = data_processor.create_dataloaders(
                batch_size=4, max_length=64, data_size=50
            )
            
            assert isinstance(train_loader, DataLoader)
            assert isinstance(val_loader, DataLoader)
            assert isinstance(test_loader, DataLoader)
            
            assert train_loader.batch_size == 4
            assert val_loader.batch_size == 4
            assert test_loader.batch_size == 4

    @pytest.mark.unit
    def test_create_dataloaders_without_data_size_limit(self, data_processor):
        """Test dataloader creation without data size limit"""
        with patch.object(data_processor, 'load_wmt_en_ro') as mock_load:
            # Mock large dataset
            mock_load.side_effect = [
                [{"source": f"src{i}", "target": f"tgt{i}"} for i in range(15000)],  # train
                [{"source": f"src{i}", "target": f"tgt{i}"} for i in range(2000)],   # val
                [{"source": f"src{i}", "target": f"tgt{i}"} for i in range(2000)]    # test
            ]
            
            train_loader, val_loader, test_loader = data_processor.create_dataloaders(
                batch_size=8, max_length=128
            )
            
            # Should limit to default sizes
            assert len(train_loader.dataset) == 10000  # Default train limit
            assert len(val_loader.dataset) == 1000     # Default val limit
            assert len(test_loader.dataset) == 1000    # Default test limit

    @pytest.mark.unit
    def test_create_pretrain_dataloaders(self, data_processor):
        """Test pretraining dataloader creation"""
        dataloader = data_processor.create_pretrain_dataloaders(
            batch_size=4, max_length=256, num_samples=20
        )
        
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 4
        
        # Check that dataset is created correctly
        assert len(dataloader.dataset) == 20

    @pytest.mark.unit
    def test_create_pretrain_dataloaders_default_params(self, data_processor):
        """Test pretraining dataloader with default parameters"""
        dataloader = data_processor.create_pretrain_dataloaders()
        
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 8  # Default batch size

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    def test_dataloaders_different_batch_sizes(self, data_processor, batch_size):
        """Test dataloaders with different batch sizes"""
        with patch.object(data_processor, 'load_wmt_en_ro') as mock_load:
            mock_load.side_effect = [
                [{"source": f"src{i}", "target": f"tgt{i}"} for i in range(20)]
                for _ in range(3)  # train, val, test
            ]
            
            train_loader, val_loader, test_loader = data_processor.create_dataloaders(
                batch_size=batch_size, data_size=20
            )
            
            assert train_loader.batch_size == batch_size
            assert val_loader.batch_size == batch_size
            assert test_loader.batch_size == batch_size

    @pytest.mark.unit
    def test_dataset_iteration(self, data_processor):
        """Test that datasets can be iterated"""
        with patch.object(data_processor, 'load_wmt_en_ro') as mock_load:
            mock_data = [{"source": "Hello", "target": "Salut"}]
            mock_load.return_value = mock_data
            
            train_loader, _, _ = data_processor.create_dataloaders(
                batch_size=1, data_size=1
            )
            
            # Should be able to iterate through dataloader
            batch = next(iter(train_loader))
            
            assert isinstance(batch, dict)
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "labels" in batch