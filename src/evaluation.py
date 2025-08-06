import torch
import sacrebleu
from rouge_score import rouge_scorer
from transformers import MBart50TokenizerFast
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import json
import os

class TranslationEvaluator:
    """
    Comprehensive evaluation for English-Romanian translation
    """
    
    def __init__(self, model, tokenizer: MBart50TokenizerFast, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def translate_batch(
        self,
        input_texts: List[str],
        src_lang: str = "en_XX",
        tgt_lang: str = "ro_RO",
        max_length: int = 128,
        num_beams: int = 5,
        length_penalty: float = 1.0
    ) -> List[str]:
        """
        Translate a batch of texts from English to Romanian
        """
        self.model.eval()
        
        # Set language tokens
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        
        # Tokenize input
        inputs = self.tokenizer(
            input_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate translations
        with torch.no_grad():
            generated_tokens = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=True,
                decoder_start_token_id=self.tokenizer.lang_code_to_id[tgt_lang]
            )
        
        # Decode translations
        translations = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )
        
        return translations
    
    def compute_bleu(
        self,
        hypotheses: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute BLEU score using sacrebleu
        """
        # sacrebleu expects references as list of lists
        refs = [[ref] for ref in references]
        
        bleu = sacrebleu.corpus_bleu(hypotheses, refs)
        
        return {
            "bleu": bleu.score,
            "bleu_signature": bleu.signature
        }
    
    def compute_rouge(
        self,
        hypotheses: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores
        """
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for hyp, ref in zip(hypotheses, references):
            scores = self.scorer.score(ref, hyp)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            "rouge1": np.mean(rouge1_scores),
            "rouge2": np.mean(rouge2_scores),
            "rougeL": np.mean(rougeL_scores)
        }
    
    def compute_chrf(
        self,
        hypotheses: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute chrF++ score
        """
        refs = [[ref] for ref in references]
        chrf = sacrebleu.corpus_chrf(hypotheses, refs)
        
        return {
            "chrf": chrf.score
        }
    
    def compute_exact_match(
        self,
        hypotheses: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute exact match accuracy
        """
        exact_matches = sum(1 for hyp, ref in zip(hypotheses, references) if hyp.strip() == ref.strip())
        accuracy = exact_matches / len(hypotheses) if hypotheses else 0.0
        
        return {
            "exact_match": accuracy
        }
    
    def evaluate_dataset(
        self,
        source_texts: List[str],
        target_texts: List[str],
        batch_size: int = 8,
        **generation_kwargs
    ) -> Dict[str, float]:
        """
        Evaluate model on a complete dataset
        """
        print("Starting evaluation...")
        
        all_hypotheses = []
        all_references = []
        
        # Process in batches
        for i in tqdm(range(0, len(source_texts), batch_size), desc="Translating"):
            batch_sources = source_texts[i:i+batch_size]
            batch_references = target_texts[i:i+batch_size]
            
            # Translate batch
            batch_hypotheses = self.translate_batch(batch_sources, **generation_kwargs)
            
            all_hypotheses.extend(batch_hypotheses)
            all_references.extend(batch_references)
        
        # Compute metrics
        bleu_scores = self.compute_bleu(all_hypotheses, all_references)
        rouge_scores = self.compute_rouge(all_hypotheses, all_references)
        chrf_scores = self.compute_chrf(all_hypotheses, all_references)
        em_scores = self.compute_exact_match(all_hypotheses, all_references)
        
        results = {
            **bleu_scores,
            **rouge_scores,
            **chrf_scores,
            **em_scores
        }
        
        return results, all_hypotheses, all_references
    
    def save_results(
        self,
        results: Dict[str, float],
        hypotheses: List[str],
        references: List[str],
        output_dir: str
    ):
        """
        Save evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save translations
        translations_path = os.path.join(output_dir, "translations.json")
        translations_data = []
        for hyp, ref in zip(hypotheses, references):
            translations_data.append({
                "hypothesis": hyp,
                "reference": ref
            })
        
        with open(translations_path, 'w', encoding='utf-8') as f:
            json.dump(translations_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_dir}")

class MBRougeScorer:
    """
    Custom ROUGE scorer for Romanian
    """
    
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def score(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Score a single pair
        """
        scores = self.scorer.score(reference, hypothesis)
        return {
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure
        }

def evaluate_model(
    model_path: str,
    test_data_path: str = None,
    source_lang: str = "en",
    target_lang: str = "ro",
    batch_size: int = 8,
    output_dir: str = "evaluation_results"
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer
    from .model import MultilingualTranslationModel
    model = MultilingualTranslationModel("facebook/mbart-large-50")
    model = model.to(device)
    tokenizer = model.tokenizer
    
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.model = model.model.to(device)
    
    # Load test data
    from .data import DataProcessor
    processor = DataProcessor()
    
    if test_data_path:
        # Load custom test data
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        source_texts = [item['source'] for item in test_data]
        target_texts = [item['target'] for item in test_data]
    else:
        # Use built-in test data
        test_data = processor.load_wmt_en_ro("test")[:100]  # Limit for speed
        source_texts = [item['source'] for item in test_data]
        target_texts = [item['target'] for item in test_data]
    
    # Evaluate
    evaluator = TranslationEvaluator(model, tokenizer, device)
    results, hypotheses, references = evaluator.evaluate_dataset(
        source_texts,
        target_texts,
        batch_size=batch_size
    )
    
    # Save results
    evaluator.save_results(results, hypotheses, references, output_dir)
    
    print("Evaluation Results:")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/translation/best_translation_model.pt")
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--batch_size", type=int, default=8)
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )