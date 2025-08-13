"""
Test data fixtures and synthetic datasets for comprehensive testing
"""

import json
import os
import random
from typing import List, Dict, Tuple
from pathlib import Path


class SyntheticDataGenerator:
    """Generate synthetic data for testing purposes"""
    
    # Base vocabulary for generating synthetic text
    ENGLISH_WORDS = [
        "the", "and", "is", "in", "to", "of", "a", "that", "it", "with",
        "for", "as", "was", "on", "are", "you", "all", "can", "had", "her",
        "what", "oil", "its", "now", "find", "he", "his", "has", "but", "word",
        "hello", "world", "good", "morning", "thank", "very", "much", "please",
        "sorry", "excuse", "help", "yes", "no", "maybe", "today", "tomorrow",
        "yesterday", "time", "water", "food", "house", "car", "book", "school",
        "work", "family", "friend", "love", "happy", "sad", "big", "small"
    ]
    
    ROMANIAN_WORDS = [
        "și", "de", "în", "cu", "la", "pe", "că", "se", "din", "să",
        "nu", "un", "pentru", "ce", "o", "sunt", "este", "mai", "lui", "sau",
        "salut", "lume", "bună", "dimineața", "mulțumesc", "foarte", "mult", "vă rog",
        "scuze", "ajutor", "da", "nu", "poate", "astăzi", "mâine",
        "ieri", "timp", "apă", "mâncare", "casă", "mașină", "carte", "școală",
        "muncă", "familie", "prieten", "dragoste", "fericit", "trist", "mare", "mic"
    ]
    
    # Common sentence patterns
    ENGLISH_PATTERNS = [
        "The {noun} is {adjective}.",
        "I {verb} {noun} {adverb}.",
        "Can you {verb} the {noun}?",
        "This is a {adjective} {noun}.",
        "We {verb} {noun} every {time}.",
        "{adjective} {noun} {verb} {adverb}.",
        "How {adjective} is the {noun}?",
        "Please {verb} the {noun}.",
    ]
    
    ROMANIAN_PATTERNS = [
        "{noun} este {adjective}.",
        "Eu {verb} {noun} {adverb}.",
        "Poți să {verb} {noun}?",
        "Aceasta este o {noun} {adjective}.",
        "Noi {verb} {noun} în fiecare {time}.",
        "{noun} {adjective} {verb} {adverb}.",
        "Cât de {adjective} este {noun}?",
        "Te rog să {verb} {noun}.",
    ]
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility"""
        random.seed(seed)
    
    def generate_word_pairs(self, count: int) -> List[Tuple[str, str]]:
        """Generate simple English-Romanian word pairs"""
        pairs = [
            ("hello", "salut"),
            ("world", "lume"),
            ("good", "bun"),
            ("morning", "dimineață"),
            ("thank", "mulțumesc"),
            ("you", "tu"),
            ("yes", "da"),
            ("no", "nu"),
            ("please", "te rog"),
            ("sorry", "scuze"),
            ("water", "apă"),
            ("food", "mâncare"),
            ("house", "casă"),
            ("car", "mașină"),
            ("book", "carte"),
            ("school", "școală"),
            ("work", "muncă"),
            ("family", "familie"),
            ("friend", "prieten"),
            ("love", "dragoste"),
            ("happy", "fericit"),
            ("sad", "trist"),
            ("big", "mare"),
            ("small", "mic"),
            ("time", "timp")
        ]
        
        # Repeat and shuffle to get desired count
        result = []
        while len(result) < count:
            result.extend(pairs)
        
        random.shuffle(result)
        return result[:count]
    
    def generate_sentence_pairs(self, count: int) -> List[Tuple[str, str]]:
        """Generate English-Romanian sentence pairs"""
        base_pairs = [
            ("Hello, how are you?", "Salut, ce mai faci?"),
            ("Good morning!", "Bună dimineața!"),
            ("Thank you very much.", "Mulțumesc foarte mult."),
            ("I love machine learning.", "Îmi place învățarea automată."),
            ("The weather is nice today.", "Vremea este frumoasă astăzi."),
            ("Can you help me?", "Mă poți ajuta?"),
            ("Where is the library?", "Unde este biblioteca?"),
            ("I am learning Romanian.", "Învăț limba română."),
            ("This book is very interesting.", "Această carte este foarte interesantă."),
            ("We are going to school.", "Mergem la școală."),
            ("The car is red.", "Mașina este roșie."),
            ("I like to read books.", "Îmi place să citesc cărți."),
            ("My family is important.", "Familia mea este importantă."),
            ("The house is big.", "Casa este mare."),
            ("Water is essential for life.", "Apa este esențială pentru viață."),
            ("I work every day.", "Lucrez în fiecare zi."),
            ("Friends are precious.", "Prietenii sunt prețioși."),
            ("Time flies quickly.", "Timpul trece repede."),
            ("Food tastes good.", "Mâncarea are gust bun."),
            ("Education is important.", "Educația este importantă."),
        ]
        
        # Generate additional pairs using patterns
        extended_pairs = base_pairs.copy()
        
        # Add some pattern-based sentences
        nouns = ["book", "car", "house", "school", "work", "friend", "time"]
        adjectives = ["good", "bad", "big", "small", "nice", "important"]
        
        for i in range(count - len(base_pairs)):
            noun = random.choice(nouns)
            adj = random.choice(adjectives)
            en_sent = f"The {noun} is {adj}."
            ro_sent = f"{noun.capitalize()} este {adj}."
            extended_pairs.append((en_sent, ro_sent))
        
        random.shuffle(extended_pairs)
        return extended_pairs[:count]
    
    def generate_monolingual_text(self, language: str, count: int) -> List[str]:
        """Generate monolingual text for pretraining"""
        if language.lower() in ['en', 'english']:
            words = self.ENGLISH_WORDS
            patterns = [
                "The {w1} and the {w2} are {w3}.",
                "I think {w1} is very {w2}.",
                "Can you {w1} the {w2}?",
                "This {w1} is {w2} and {w3}.",
                "We need to {w1} more {w2}.",
                "{w1} is important for {w2}.",
                "Every {w1} should {w2}.",
                "The best {w1} is {w2}.",
            ]
        elif language.lower() in ['ro', 'romanian']:
            words = self.ROMANIAN_WORDS
            patterns = [
                "{w1} și {w2} sunt {w3}.",
                "Cred că {w1} este foarte {w2}.",
                "Poți să {w1} {w2}?",
                "Acest {w1} este {w2} și {w3}.",
                "Trebuie să {w1} mai mult {w2}.",
                "{w1} este important pentru {w2}.",
                "Fiecare {w1} ar trebui să {w2}.",
                "Cel mai bun {w1} este {w2}.",
            ]
        else:
            raise ValueError(f"Unsupported language: {language}")
        
        texts = []
        for _ in range(count):
            pattern = random.choice(patterns)
            w1, w2, w3 = random.choices(words, k=3)
            text = pattern.format(w1=w1, w2=w2, w3=w3)
            texts.append(text)
        
        return texts
    
    def generate_translation_dataset(self, size: int, train_ratio: float = 0.8, 
                                   val_ratio: float = 0.1) -> Dict[str, List[Dict]]:
        """Generate a complete translation dataset split"""
        all_pairs = self.generate_sentence_pairs(size)
        
        train_size = int(size * train_ratio)
        val_size = int(size * val_ratio)
        test_size = size - train_size - val_size
        
        train_pairs = all_pairs[:train_size]
        val_pairs = all_pairs[train_size:train_size + val_size]
        test_pairs = all_pairs[train_size + val_size:train_size + val_size + test_size]
        
        def pairs_to_dict(pairs):
            return [{"source": src, "target": tgt} for src, tgt in pairs]
        
        return {
            "train": pairs_to_dict(train_pairs),
            "validation": pairs_to_dict(val_pairs),
            "test": pairs_to_dict(test_pairs)
        }
    
    def generate_multilingual_pretrain_data(self, size: int, 
                                          languages: List[str] = None) -> List[str]:
        """Generate multilingual pretraining data"""
        if languages is None:
            languages = ['en', 'ro']
        
        texts = []
        texts_per_lang = size // len(languages)
        
        for lang in languages:
            lang_texts = self.generate_monolingual_text(lang, texts_per_lang)
            texts.extend(lang_texts)
        
        # Add remaining texts if size doesn't divide evenly
        remaining = size - len(texts)
        if remaining > 0:
            extra_texts = self.generate_monolingual_text(languages[0], remaining)
            texts.extend(extra_texts)
        
        random.shuffle(texts)
        return texts
    
    def generate_evaluation_data(self, size: int = 100) -> Dict:
        """Generate data specifically for evaluation testing"""
        # Generate pairs with known properties
        perfect_matches = []
        partial_matches = []
        no_matches = []
        
        # Perfect matches (identical)
        for i in range(size // 3):
            text = f"Test sentence {i}"
            perfect_matches.append({
                "source": f"Source: {text}",
                "reference": f"Source: {text}",
                "hypothesis": f"Source: {text}"
            })
        
        # Partial matches (similar but not identical)
        for i in range(size // 3):
            base_text = f"Test sentence {i}"
            partial_matches.append({
                "source": f"Source: {base_text}",
                "reference": f"Reference: {base_text}",
                "hypothesis": f"Hypothesis: {base_text}"
            })
        
        # No matches (completely different)
        for i in range(size - len(perfect_matches) - len(partial_matches)):
            no_matches.append({
                "source": f"Source text {i}",
                "reference": f"Different reference {i}",
                "hypothesis": f"Unrelated hypothesis {i}"
            })
        
        all_data = perfect_matches + partial_matches + no_matches
        random.shuffle(all_data)
        
        return {
            "evaluation_data": all_data,
            "perfect_match_count": len(perfect_matches),
            "partial_match_count": len(partial_matches),
            "no_match_count": len(no_matches)
        }
    
    def generate_edge_case_data(self) -> Dict:
        """Generate edge case data for testing"""
        return {
            "empty_strings": [
                {"source": "", "target": ""},
                {"source": "Hello", "target": ""},
                {"source": "", "target": "Bonjour"}
            ],
            "single_chars": [
                {"source": "a", "target": "b"},
                {"source": "1", "target": "2"},
                {"source": "!", "target": "?"}
            ],
            "very_long": [
                {
                    "source": "This is a very long sentence. " * 50,
                    "target": "Aceasta este o propoziție foarte lungă. " * 50
                }
            ],
            "special_chars": [
                {"source": "Hello 🌍", "target": "Salut 🌍"},
                {"source": "Café naïve", "target": "Cafea naivă"},
                {"source": "Test@#$%", "target": "Test@#$%"}
            ],
            "numbers": [
                {"source": "I have 5 apples", "target": "Am 5 mere"},
                {"source": "Price: $19.99", "target": "Preț: $19.99"},
                {"source": "Year 2024", "target": "Anul 2024"}
            ],
            "mixed_scripts": [
                {"source": "Hello 世界", "target": "Salut monde"},
                {"source": "Test العربية", "target": "Test Arabic"},
                {"source": "123 αβγ", "target": "456 abc"}
            ]
        }


class TestDataFixtures:
    """Predefined test data fixtures"""
    
    @staticmethod
    def get_minimal_config() -> Dict:
        """Get minimal configuration for testing"""
        return {
            "model": {
                "d_model": 128,
                "encoder_layers": 2,
                "decoder_layers": 2,
                "encoder_attention_heads": 4,
                "decoder_attention_heads": 4,
                "encoder_ffn_dim": 256,
                "decoder_ffn_dim": 256,
                "max_position_embeddings": 128,
                "dropout": 0.1
            },
            "training": {
                "batch_size": 2,
                "max_steps": 5,
                "max_epochs": 1,
                "learning_rate": 1e-4,
                "warmup_steps": 1,
                "weight_decay": 0.01
            },
            "data": {
                "max_length": 32,
                "num_samples": 10,
                "train_size": 10
            },
            "output": {
                "output_dir": "test_output",
                "log_dir": "test_logs"
            }
        }
    
    @staticmethod
    def get_production_config() -> Dict:
        """Get production-like configuration"""
        return {
            "model": {
                "d_model": 1024,
                "encoder_layers": 12,
                "decoder_layers": 12,
                "encoder_attention_heads": 16,
                "decoder_attention_heads": 16,
                "encoder_ffn_dim": 4096,
                "decoder_ffn_dim": 4096,
                "max_position_embeddings": 1024,
                "dropout": 0.1
            },
            "training": {
                "batch_size": 32,
                "max_steps": 100000,
                "max_epochs": 10,
                "learning_rate": 5e-4,
                "warmup_steps": 10000,
                "weight_decay": 0.01
            },
            "data": {
                "max_length": 512,
                "num_samples": 1000000,
                "train_size": 1000000
            },
            "output": {
                "output_dir": "checkpoints/production",
                "log_dir": "logs/production"
            }
        }
    
    @staticmethod
    def get_sample_translation_pairs() -> List[Dict[str, str]]:
        """Get sample English-Romanian translation pairs"""
        return [
            {"source": "Hello, how are you?", "target": "Salut, ce mai faci?"},
            {"source": "Good morning!", "target": "Bună dimineața!"},
            {"source": "Thank you very much.", "target": "Mulțumesc foarte mult."},
            {"source": "I love machine learning.", "target": "Îmi place învățarea automată."},
            {"source": "The weather is nice today.", "target": "Vremea este frumoasă astăzi."},
            {"source": "Can you help me?", "target": "Mă poți ajuta?"},
            {"source": "Where is the library?", "target": "Unde este biblioteca?"},
            {"source": "I am learning Romanian.", "target": "Învăț limba română."},
            {"source": "This book is interesting.", "target": "Această carte este interesantă."},
            {"source": "We are going to school.", "target": "Mergem la școală."}
        ]
    
    @staticmethod
    def get_sample_monolingual_data() -> Dict[str, List[str]]:
        """Get sample monolingual data for pretraining"""
        return {
            "english": [
                "This is a sample English sentence.",
                "Machine learning is transforming how we process language.",
                "Natural language processing enables computers to understand text.",
                "Artificial intelligence continues to advance rapidly.",
                "Deep learning models require large amounts of data.",
                "Translation models help break down language barriers.",
                "Technology makes global communication easier.",
                "Research in AI benefits many industries.",
                "Data science combines statistics and computer science.",
                "Innovation drives progress in language technology."
            ],
            "romanian": [
                "Aceasta este o propoziție de exemplu în română.",
                "Învățarea automată transformă modul în care procesăm limba.",
                "Procesarea limbajului natural permite calculatoarelor să înțeleagă textul.",
                "Inteligența artificială continuă să avanseze rapid.",
                "Modelele de învățare profundă necesită cantități mari de date.",
                "Modelele de traducere ajută la eliminarea barierelor lingvistice.",
                "Tehnologia face comunicarea globală mai ușoară.",
                "Cercetarea în IA beneficiază multe industrii.",
                "Știința datelor combină statistica și informatica.",
                "Inovația impulsionează progresul în tehnologia limbajului."
            ]
        }
    
    @staticmethod
    def create_test_files(output_dir: str):
        """Create test data files in the specified directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        generator = SyntheticDataGenerator()
        
        # Create translation dataset
        translation_data = generator.generate_translation_dataset(100)
        for split, data in translation_data.items():
            filepath = os.path.join(output_dir, f"translation_{split}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Create monolingual data
        en_texts = generator.generate_monolingual_text("en", 50)
        ro_texts = generator.generate_monolingual_text("ro", 50)
        
        with open(os.path.join(output_dir, "monolingual_en.json"), 'w', encoding='utf-8') as f:
            json.dump(en_texts, f, indent=2, ensure_ascii=False)
        
        with open(os.path.join(output_dir, "monolingual_ro.json"), 'w', encoding='utf-8') as f:
            json.dump(ro_texts, f, indent=2, ensure_ascii=False)
        
        # Create evaluation data
        eval_data = generator.generate_evaluation_data(30)
        with open(os.path.join(output_dir, "evaluation_data.json"), 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)
        
        # Create edge case data
        edge_data = generator.generate_edge_case_data()
        with open(os.path.join(output_dir, "edge_cases.json"), 'w', encoding='utf-8') as f:
            json.dump(edge_data, f, indent=2, ensure_ascii=False)
        
        # Create configs
        configs = {
            "minimal": TestDataFixtures.get_minimal_config(),
            "production": TestDataFixtures.get_production_config()
        }
        
        for name, config in configs.items():
            filepath = os.path.join(output_dir, f"config_{name}.json")
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
        
        print(f"Test data files created in: {output_dir}")
        return output_dir


def create_comprehensive_test_data(base_dir: str = "tests/data"):
    """Create comprehensive test data for the project"""
    data_dir = Path(base_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create different sizes of datasets for different test scenarios
    generator = SyntheticDataGenerator()
    
    # Small dataset for unit tests
    small_data = generator.generate_translation_dataset(20, 0.7, 0.2)
    small_dir = data_dir / "small"
    small_dir.mkdir(exist_ok=True)
    
    for split, data in small_data.items():
        filepath = small_dir / f"{split}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Medium dataset for integration tests
    medium_data = generator.generate_translation_dataset(200, 0.8, 0.1)
    medium_dir = data_dir / "medium"
    medium_dir.mkdir(exist_ok=True)
    
    for split, data in medium_data.items():
        filepath = medium_dir / f"{split}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Create specialized test datasets
    TestDataFixtures.create_test_files(str(data_dir / "fixtures"))
    
    print(f"Comprehensive test data created in: {data_dir}")
    return str(data_dir)


if __name__ == "__main__":
    # Create test data when run as script
    create_comprehensive_test_data()