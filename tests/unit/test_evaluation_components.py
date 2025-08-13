"""
Unit tests for evaluation components
"""

import pytest
import torch
import json
import os
import tempfile
from unittest.mock import patch, MagicMock, call
import numpy as np

from src.evaluation import TranslationEvaluator, MBRougeScorer, evaluate_model


class TestTranslationEvaluator:
    """Test TranslationEvaluator class"""

    @pytest.fixture
    def evaluator(self, mock_model, mock_tokenizer, device):
        """Create evaluator instance"""
        return TranslationEvaluator(mock_model, mock_tokenizer, device)

    @pytest.mark.unit
    def test_initialization(self, mock_model, mock_tokenizer, device):
        """Test evaluator initialization"""
        evaluator = TranslationEvaluator(mock_model, mock_tokenizer, device)
        
        assert evaluator.model == mock_model
        assert evaluator.tokenizer == mock_tokenizer
        assert evaluator.device == device
        assert hasattr(evaluator, 'scorer')

    @pytest.mark.unit
    def test_translate_batch_basic(self, evaluator):
        """Test basic batch translation"""
        input_texts = ["Hello world", "Good morning"]
        
        # Mock model generate method
        evaluator.model.generate.return_value = torch.tensor([[100, 200, 2], [300, 400, 2]])
        
        translations = evaluator.translate_batch(input_texts)
        
        assert isinstance(translations, list)
        assert len(translations) == 2
        assert all(isinstance(t, str) for t in translations)

    @pytest.mark.unit
    def test_translate_batch_language_settings(self, evaluator):
        """Test that language settings are applied correctly"""
        input_texts = ["Hello"]
        
        evaluator.translate_batch(
            input_texts,
            src_lang="fr_XX",
            tgt_lang="de_DE"
        )
        
        assert evaluator.tokenizer.src_lang == "fr_XX"
        assert evaluator.tokenizer.tgt_lang == "de_DE"

    @pytest.mark.unit
    def test_translate_batch_generation_params(self, evaluator):
        """Test that generation parameters are passed correctly"""
        input_texts = ["Hello"]
        
        evaluator.translate_batch(
            input_texts,
            max_length=256,
            num_beams=8,
            length_penalty=0.8
        )
        
        # Verify generate was called with correct parameters
        call_args = evaluator.model.generate.call_args
        assert call_args.kwargs['max_length'] == 256
        assert call_args.kwargs['num_beams'] == 8
        assert call_args.kwargs['length_penalty'] == 0.8

    @pytest.mark.unit
    def test_translate_batch_model_eval_mode(self, evaluator):
        """Test that model is set to eval mode during translation"""
        input_texts = ["Hello"]
        
        evaluator.translate_batch(input_texts)
        
        evaluator.model.eval.assert_called_once()

    @pytest.mark.unit
    def test_translate_batch_empty_input(self, evaluator):
        """Test translation with empty input"""
        translations = evaluator.translate_batch([])
        
        assert translations == []

    @pytest.mark.unit
    @patch('src.evaluation.sacrebleu.corpus_bleu')
    def test_compute_bleu(self, mock_corpus_bleu, evaluator):
        """Test BLEU score computation"""
        mock_bleu_result = MagicMock()
        mock_bleu_result.score = 25.5
        mock_bleu_result.signature = "BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.5.0"
        mock_corpus_bleu.return_value = mock_bleu_result
        
        hypotheses = ["Hello world", "Good morning"]
        references = ["Bonjour monde", "Bonjour matin"]
        
        result = evaluator.compute_bleu(hypotheses, references)
        
        assert result['bleu'] == 25.5
        assert 'bleu_signature' in result
        
        # Verify sacrebleu was called with correct format
        mock_corpus_bleu.assert_called_once()
        call_args = mock_corpus_bleu.call_args[0]
        assert call_args[0] == hypotheses  # hypotheses
        assert call_args[1] == [["Bonjour monde"], ["Bonjour matin"]]  # references as list of lists

    @pytest.mark.unit
    @patch('src.evaluation.rouge_scorer.RougeScorer')
    def test_compute_rouge(self, mock_rouge_scorer_class, evaluator):
        """Test ROUGE score computation"""
        mock_scorer = MagicMock()
        mock_rouge_scorer_class.return_value = mock_scorer
        
        # Mock individual scores
        mock_score = MagicMock()
        mock_score.fmeasure = 0.75
        mock_scores = {
            'rouge1': mock_score,
            'rouge2': mock_score,
            'rougeL': mock_score
        }
        mock_scorer.score.return_value = mock_scores
        
        # Create new evaluator to use mocked scorer
        evaluator_with_mock = TranslationEvaluator(evaluator.model, evaluator.tokenizer, evaluator.device)
        
        hypotheses = ["Hello world", "Good morning"]
        references = ["Hello earth", "Good evening"]
        
        result = evaluator_with_mock.compute_rouge(hypotheses, references)
        
        assert result['rouge1'] == 0.75
        assert result['rouge2'] == 0.75
        assert result['rougeL'] == 0.75

    @pytest.mark.unit
    @patch('src.evaluation.sacrebleu.corpus_chrf')
    def test_compute_chrf(self, mock_corpus_chrf, evaluator):
        """Test chrF++ score computation"""
        mock_chrf_result = MagicMock()
        mock_chrf_result.score = 55.2
        mock_corpus_chrf.return_value = mock_chrf_result
        
        hypotheses = ["Hello world"]
        references = ["Hello earth"]
        
        result = evaluator.compute_chrf(hypotheses, references)
        
        assert result['chrf'] == 55.2
        mock_corpus_chrf.assert_called_once()

    @pytest.mark.unit
    def test_compute_exact_match(self, evaluator):
        """Test exact match computation"""
        hypotheses = ["Hello world", "Good morning", "Test sentence"]
        references = ["Hello world", "Good evening", "Test sentence"]
        
        result = evaluator.compute_exact_match(hypotheses, references)
        
        # 2 out of 3 match exactly
        assert result['exact_match'] == 2/3

    @pytest.mark.unit
    def test_compute_exact_match_empty(self, evaluator):
        """Test exact match with empty lists"""
        result = evaluator.compute_exact_match([], [])
        
        assert result['exact_match'] == 0.0

    @pytest.mark.unit
    def test_compute_exact_match_whitespace_handling(self, evaluator):
        """Test exact match handles whitespace correctly"""
        hypotheses = [" Hello world ", "Good morning"]
        references = ["Hello world", " Good morning "]
        
        result = evaluator.compute_exact_match(hypotheses, references)
        
        # Both should match after stripping
        assert result['exact_match'] == 1.0

    @pytest.mark.unit
    def test_evaluate_dataset_structure(self, evaluator):
        """Test evaluate_dataset method structure"""
        source_texts = ["Hello", "Goodbye"]
        target_texts = ["Bonjour", "Au revoir"]
        
        # Mock all metric computations
        with patch.object(evaluator, 'translate_batch') as mock_translate:
            mock_translate.return_value = ["Salut", "Adieu"]
            
            with patch.object(evaluator, 'compute_bleu') as mock_bleu, \
                 patch.object(evaluator, 'compute_rouge') as mock_rouge, \
                 patch.object(evaluator, 'compute_chrf') as mock_chrf, \
                 patch.object(evaluator, 'compute_exact_match') as mock_em:
                
                mock_bleu.return_value = {'bleu': 20.0}
                mock_rouge.return_value = {'rouge1': 0.5, 'rouge2': 0.3, 'rougeL': 0.4}
                mock_chrf.return_value = {'chrf': 45.0}
                mock_em.return_value = {'exact_match': 0.0}
                
                results, hypotheses, references = evaluator.evaluate_dataset(
                    source_texts, target_texts, batch_size=1
                )
                
                # Check that all metrics are included
                assert 'bleu' in results
                assert 'rouge1' in results
                assert 'rouge2' in results
                assert 'rougeL' in results
                assert 'chrf' in results
                assert 'exact_match' in results
                
                # Check return format
                assert len(hypotheses) == 2
                assert len(references) == 2

    @pytest.mark.unit
    @patch('src.evaluation.tqdm')
    def test_evaluate_dataset_batching(self, mock_tqdm, evaluator):
        """Test that evaluation processes in batches correctly"""
        mock_tqdm.return_value = range(0, 4, 2)  # Mock tqdm to return range
        
        source_texts = ["Hello", "Goodbye", "Thank you", "See you"]
        target_texts = ["Bonjour", "Au revoir", "Merci", "A bientot"]
        
        with patch.object(evaluator, 'translate_batch') as mock_translate:
            mock_translate.side_effect = [
                ["Salut", "Adieu"],      # First batch
                ["Merci", "A bientot"]   # Second batch
            ]
            
            with patch.object(evaluator, 'compute_bleu') as mock_bleu, \
                 patch.object(evaluator, 'compute_rouge') as mock_rouge, \
                 patch.object(evaluator, 'compute_chrf') as mock_chrf, \
                 patch.object(evaluator, 'compute_exact_match') as mock_em:
                
                mock_bleu.return_value = {'bleu': 20.0}
                mock_rouge.return_value = {'rouge1': 0.5, 'rouge2': 0.3, 'rougeL': 0.4}
                mock_chrf.return_value = {'chrf': 45.0}
                mock_em.return_value = {'exact_match': 0.0}
                
                results, hypotheses, references = evaluator.evaluate_dataset(
                    source_texts, target_texts, batch_size=2
                )
                
                # Verify translate_batch was called twice with correct batch sizes
                assert mock_translate.call_count == 2
                assert len(hypotheses) == 4
                assert len(references) == 4

    @pytest.mark.unit
    def test_save_results(self, evaluator):
        """Test results saving functionality"""
        results = {'bleu': 25.5, 'rouge1': 0.75, 'chrf': 55.2}
        hypotheses = ["Hello world", "Good morning"]
        references = ["Bonjour monde", "Bonjour matin"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator.save_results(results, hypotheses, references, temp_dir)
            
            # Check that files were created
            metrics_path = os.path.join(temp_dir, "metrics.json")
            translations_path = os.path.join(temp_dir, "translations.json")
            
            assert os.path.exists(metrics_path)
            assert os.path.exists(translations_path)
            
            # Verify content
            with open(metrics_path, 'r') as f:
                saved_metrics = json.load(f)
            assert saved_metrics == results
            
            with open(translations_path, 'r') as f:
                saved_translations = json.load(f)
            assert len(saved_translations) == 2
            assert saved_translations[0]['hypothesis'] == "Hello world"
            assert saved_translations[0]['reference'] == "Bonjour monde"

    @pytest.mark.unit
    def test_save_results_creates_directory(self, evaluator):
        """Test that save_results creates output directory if it doesn't exist"""
        results = {'bleu': 25.5}
        hypotheses = ["Hello"]
        references = ["Bonjour"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "new_output")
            
            evaluator.save_results(results, hypotheses, references, output_dir)
            
            assert os.path.exists(output_dir)
            assert os.path.exists(os.path.join(output_dir, "metrics.json"))


class TestMBRougeScorer:
    """Test MBRougeScorer class"""

    @pytest.mark.unit
    def test_initialization(self):
        """Test scorer initialization"""
        scorer = MBRougeScorer()
        
        assert hasattr(scorer, 'scorer')

    @pytest.mark.unit
    @patch('src.evaluation.rouge_scorer.RougeScorer')
    def test_score_method(self, mock_rouge_scorer_class):
        """Test score method"""
        mock_scorer = MagicMock()
        mock_rouge_scorer_class.return_value = mock_scorer
        
        # Mock score return value
        mock_score = MagicMock()
        mock_score.fmeasure = 0.8
        mock_scores = {
            'rouge1': mock_score,
            'rouge2': mock_score,
            'rougeL': mock_score
        }
        mock_scorer.score.return_value = mock_scores
        
        scorer = MBRougeScorer()
        result = scorer.score("reference text", "hypothesis text")
        
        assert result['rouge1'] == 0.8
        assert result['rouge2'] == 0.8
        assert result['rougeL'] == 0.8


class TestEvaluateModelFunction:
    """Test evaluate_model function"""

    @pytest.mark.unit
    @patch('src.evaluation.torch.load')
    @patch('src.evaluation.MultilingualTranslationModel')
    @patch('src.evaluation.DataProcessor')
    @patch('src.evaluation.TranslationEvaluator')
    def test_evaluate_model_with_checkpoint(
        self, 
        mock_evaluator_class, 
        mock_processor_class, 
        mock_model_class, 
        mock_torch_load
    ):
        """Test evaluate_model function with checkpoint loading"""
        # Setup mocks
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        mock_processor = MagicMock()
        mock_processor.load_wmt_en_ro.return_value = [
            {"source": "Hello", "target": "Bonjour"}
        ]
        mock_processor_class.return_value = mock_processor
        
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_dataset.return_value = (
            {'bleu': 25.0, 'rouge1': 0.5},
            ["Salut"],
            ["Bonjour"]
        )
        mock_evaluator.save_results = MagicMock()
        mock_evaluator_class.return_value = mock_evaluator
        
        mock_checkpoint = {'model_state_dict': {}}
        mock_torch_load.return_value = mock_checkpoint
        
        with tempfile.NamedTemporaryFile() as temp_model_file:
            results = evaluate_model(
                model_path=temp_model_file.name,
                batch_size=4,
                output_dir="test_output"
            )
            
            # Verify checkpoint was loaded
            mock_torch_load.assert_called_once()
            mock_model.model.load_state_dict.assert_called_once()
            
            # Verify evaluation was performed
            mock_evaluator.evaluate_dataset.assert_called_once()
            mock_evaluator.save_results.assert_called_once()
            
            # Verify results format
            assert 'bleu' in results
            assert 'rouge1' in results

    @pytest.mark.unit
    @patch('src.evaluation.MultilingualTranslationModel')
    @patch('src.evaluation.DataProcessor')
    @patch('src.evaluation.TranslationEvaluator')
    def test_evaluate_model_without_checkpoint(
        self, 
        mock_evaluator_class, 
        mock_processor_class, 
        mock_model_class
    ):
        """Test evaluate_model function without checkpoint"""
        # Setup mocks
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        mock_processor = MagicMock()
        mock_processor.load_wmt_en_ro.return_value = [
            {"source": "Hello", "target": "Bonjour"}
        ]
        mock_processor_class.return_value = mock_processor
        
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_dataset.return_value = (
            {'bleu': 25.0},
            ["Salut"],
            ["Bonjour"]
        )
        mock_evaluator.save_results = MagicMock()
        mock_evaluator_class.return_value = mock_evaluator
        
        results = evaluate_model(
            model_path=None,  # No checkpoint
            batch_size=8
        )
        
        # Should not try to load checkpoint
        mock_model.model.load_state_dict.assert_not_called()
        
        # Should still perform evaluation
        mock_evaluator.evaluate_dataset.assert_called_once()

    @pytest.mark.unit
    @patch('src.evaluation.MultilingualTranslationModel')
    @patch('src.evaluation.DataProcessor')
    @patch('src.evaluation.TranslationEvaluator')
    def test_evaluate_model_with_custom_test_data(
        self, 
        mock_evaluator_class, 
        mock_processor_class, 
        mock_model_class
    ):
        """Test evaluate_model with custom test data file"""
        # Setup mocks
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor
        
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_dataset.return_value = (
            {'bleu': 30.0},
            ["Custom translation"],
            ["Custom reference"]
        )
        mock_evaluator.save_results = MagicMock()
        mock_evaluator_class.return_value = mock_evaluator
        
        # Create temporary test data file
        test_data = [{"source": "Custom source", "target": "Custom target"}]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            test_data_path = f.name
        
        try:
            results = evaluate_model(
                model_path=None,
                test_data_path=test_data_path
            )
            
            # Should not call processor.load_wmt_en_ro
            mock_processor.load_wmt_en_ro.assert_not_called()
            
            # Should still perform evaluation
            mock_evaluator.evaluate_dataset.assert_called_once()
            call_args = mock_evaluator.evaluate_dataset.call_args[0]
            assert call_args[0] == ["Custom source"]  # source_texts
            assert call_args[1] == ["Custom target"]   # target_texts
            
        finally:
            os.unlink(test_data_path)

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    def test_evaluate_model_different_batch_sizes(self, batch_size):
        """Test evaluate_model with different batch sizes"""
        with patch('src.evaluation.MultilingualTranslationModel') as mock_model_class, \
             patch('src.evaluation.DataProcessor') as mock_processor_class, \
             patch('src.evaluation.TranslationEvaluator') as mock_evaluator_class:
            
            # Setup basic mocks
            mock_model_class.return_value = MagicMock()
            mock_processor = MagicMock()
            mock_processor.load_wmt_en_ro.return_value = [{"source": "test", "target": "test"}]
            mock_processor_class.return_value = mock_processor
            
            mock_evaluator = MagicMock()
            mock_evaluator.evaluate_dataset.return_value = ({}, [], [])
            mock_evaluator.save_results = MagicMock()
            mock_evaluator_class.return_value = mock_evaluator
            
            evaluate_model(model_path=None, batch_size=batch_size)
            
            # Verify batch_size was passed to evaluate_dataset
            call_args = mock_evaluator.evaluate_dataset.call_args
            assert call_args.kwargs['batch_size'] == batch_size