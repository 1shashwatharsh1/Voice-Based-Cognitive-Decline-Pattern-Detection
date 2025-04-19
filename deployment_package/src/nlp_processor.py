import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from typing import Dict, List
import re

class NLPProcessor:
    def __init__(self):
        # Initialize NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        
        # Define hesitation markers
        self.hesitation_markers = {
            'uh', 'um', 'er', 'ah', 'eh', 'like', 'you know', 'well',
            'actually', 'basically', 'literally', 'sort of', 'kind of'
        }
        
    def analyze_text(self, text: str) -> Dict:
        """Analyze transcribed text for cognitive indicators."""
        if not text:
            return {}
            
        features = {}
        
        # Basic text statistics
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        features['sentence_count'] = len(sentences)
        features['word_count'] = len(words)
        features['avg_words_per_sentence'] = len(words) / len(sentences) if sentences else 0
        
        # Hesitation analysis
        features['hesitation_count'] = self._count_hesitations(text)
        features['hesitation_rate'] = features['hesitation_count'] / len(words) if words else 0
        
        # Word recall analysis
        features['word_repetitions'] = self._count_word_repetitions(words)
        features['word_substitutions'] = self._detect_word_substitutions(text)
        
        # Sentence complexity
        features['avg_sentence_length'] = self._calculate_avg_sentence_length(sentences)
        features['sentence_complexity'] = self._analyze_sentence_complexity(sentences)
        
        return features
        
    def _count_hesitations(self, text: str) -> int:
        """Count occurrences of hesitation markers."""
        count = 0
        text_lower = text.lower()
        for marker in self.hesitation_markers:
            count += text_lower.count(marker)
        return count
        
    def _count_word_repetitions(self, words: List[str]) -> int:
        """Count repeated words within a short window."""
        repetitions = 0
        window_size = 5
        
        for i in range(len(words) - window_size):
            window = words[i:i + window_size]
            for word in set(window):
                if window.count(word) > 1:
                    repetitions += 1
                    
        return repetitions
        
    def _detect_word_substitutions(self, text: str) -> List[str]:
        """Detect potential word substitutions or incorrect word usage."""
        # This is a simplified version - in practice, you'd want a more sophisticated approach
        # using word embeddings or language models
        substitutions = []
        words = word_tokenize(text)
        tagged = pos_tag(words)
        
        for i in range(len(tagged) - 1):
            current_word, current_pos = tagged[i]
            next_word, next_pos = tagged[i + 1]
            
            # Look for unusual word combinations
            if current_pos.startswith('NN') and next_pos.startswith('NN'):
                if not self._is_valid_noun_phrase(current_word, next_word):
                    substitutions.append(f"{current_word} -> {next_word}")
                    
        return substitutions
        
    def _calculate_avg_sentence_length(self, sentences: List[str]) -> float:
        """Calculate average sentence length in words."""
        if not sentences:
            return 0
        total_words = sum(len(word_tokenize(sent)) for sent in sentences)
        return total_words / len(sentences)
        
    def _analyze_sentence_complexity(self, sentences: List[str]) -> float:
        """Analyze sentence complexity based on various metrics."""
        if not sentences:
            return 0
            
        complexity_scores = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            tagged = pos_tag(words)
            
            # Count different parts of speech
            pos_counts = {}
            for _, pos in tagged:
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
                
            # Calculate complexity score
            score = (
                len(words) * 0.3 +  # Length factor
                len(set(words)) / len(words) * 0.3 +  # Vocabulary diversity
                sum(1 for pos in pos_counts if pos.startswith(('VB', 'JJ', 'RB'))) * 0.4  # Verb/adjective/adverb complexity
            )
            complexity_scores.append(score)
            
        return sum(complexity_scores) / len(complexity_scores)
        
    def _is_valid_noun_phrase(self, word1: str, word2: str) -> bool:
        """Check if two nouns form a valid noun phrase."""
        # This is a simplified check - in practice, you'd want a more sophisticated approach
        common_noun_phrases = {
            'coffee cup', 'computer screen', 'phone call', 'car keys',
            'house door', 'book shelf', 'water bottle', 'phone number'
        }
        return f"{word1} {word2}".lower() in common_noun_phrases 