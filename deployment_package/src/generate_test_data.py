import os
import numpy as np
import soundfile as sf
from scipy.io.wavfile import write
import librosa
from gtts import gTTS
import tempfile
import random
from typing import List, Dict

class TestDataGenerator:
    def __init__(self, output_dir: str = "data/audio_samples"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Define test scenarios
        self.scenarios = {
            'normal': {
                'hesitation_rate': 0.05,
                'pause_duration': 0.2,
                'word_substitution_rate': 0.0,
                'speech_rate': 1.0
            },
            'mild_impairment': {
                'hesitation_rate': 0.15,
                'pause_duration': 0.5,
                'word_substitution_rate': 0.1,
                'speech_rate': 0.8
            },
            'moderate_impairment': {
                'hesitation_rate': 0.25,
                'pause_duration': 1.0,
                'word_substitution_rate': 0.2,
                'speech_rate': 0.6
            }
        }
        
        # Define test sentences
        self.sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "I went to the store to buy some groceries.",
            "My favorite color is blue because it reminds me of the ocean.",
            "Yesterday, I visited my friend who lives in the city.",
            "The weather is beautiful today, perfect for a walk in the park."
        ]
        
        # Word association test items
        self.word_association_pairs = [
            ("dog", "cat"),
            ("pen", "paper"),
            ("bread", "butter"),
            ("salt", "pepper"),
            ("knife", "fork")
        ]
        
        # Common word substitutions
        self.common_substitutions = {
            "dog": ["pet", "animal", "puppy"],
            "store": ["shop", "market", "mall"],
            "friend": ["buddy", "pal", "companion"],
            "weather": ["climate", "temperature", "conditions"],
            "ocean": ["sea", "water", "waves"]
        }
        
    def generate_hesitation(self, text: str, rate: float) -> str:
        """Add hesitation markers to text based on rate."""
        hesitation_markers = ["uh", "um", "er", "ah", "well", "you know"]
        words = text.split()
        num_hesitations = int(len(words) * rate)
        
        for _ in range(num_hesitations):
            pos = random.randint(0, len(words))
            marker = random.choice(hesitation_markers)
            words.insert(pos, marker)
            
        return " ".join(words)
        
    def generate_pauses(self, audio: np.ndarray, sr: int, duration: float) -> np.ndarray:
        """Add pauses to audio."""
        pause_samples = int(duration * sr)
        pause = np.zeros(pause_samples)
        
        # Insert pause at random position
        pos = random.randint(0, len(audio))
        return np.concatenate([audio[:pos], pause, audio[pos:]])
        
    def generate_word_substitution(self, text: str, rate: float) -> str:
        """Substitute words based on rate."""
        words = text.split()
        num_substitutions = int(len(words) * rate)
        
        for _ in range(num_substitutions):
            for i, word in enumerate(words):
                if word.lower() in self.common_substitutions and random.random() < rate:
                    words[i] = random.choice(self.common_substitutions[word.lower()])
                    
        return " ".join(words)
        
    def adjust_speech_rate(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """Adjust speech rate using librosa."""
        return librosa.effects.time_stretch(audio, rate=rate)
        
    def generate_test_sample(self, scenario: str, sentence: str) -> str:
        """Generate a test audio sample with specified cognitive patterns."""
        params = self.scenarios[scenario]
        
        # Apply text modifications
        modified_text = self.generate_hesitation(sentence, params['hesitation_rate'])
        modified_text = self.generate_word_substitution(modified_text, params['word_substitution_rate'])
        
        # Generate speech using gTTS
        tts = gTTS(text=modified_text, lang='en', slow=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            tts.save(temp_file.name)
            audio, sr = librosa.load(temp_file.name, sr=None)
            
        # Apply audio modifications
        audio = self.generate_pauses(audio, sr, params['pause_duration'])
        audio = self.adjust_speech_rate(audio, params['speech_rate'])
        
        # Save final audio
        output_file = os.path.join(
            self.output_dir,
            f"{scenario}_{len(os.listdir(self.output_dir))}.wav"
        )
        write(output_file, sr, audio)
        
        return output_file
        
    def generate_word_association_test(self, scenario: str) -> List[str]:
        """Generate word association test samples."""
        samples = []
        for word, association in self.word_association_pairs:
            prompt = f"What word comes to mind when you hear the word {word}?"
            sample = self.generate_test_sample(scenario, prompt)
            samples.append(sample)
        return samples
        
    def generate_all_samples(self, num_samples: int = 5) -> Dict[str, List[str]]:
        """Generate a complete set of test samples."""
        samples = {
            'normal': [],
            'mild_impairment': [],
            'moderate_impairment': []
        }
        
        # Generate regular speech samples
        for scenario in samples.keys():
            for _ in range(num_samples):
                sentence = random.choice(self.sentences)
                sample = self.generate_test_sample(scenario, sentence)
                samples[scenario].append(sample)
                
        # Generate word association tests
        for scenario in samples.keys():
            association_samples = self.generate_word_association_test(scenario)
            samples[scenario].extend(association_samples)
            
        return samples

def main():
    generator = TestDataGenerator()
    samples = generator.generate_all_samples(num_samples=5)
    
    print("Generated test samples:")
    for scenario, files in samples.items():
        print(f"\n{scenario.capitalize()} samples:")
        for file in files:
            print(f"- {os.path.basename(file)}")

if __name__ == '__main__':
    main() 