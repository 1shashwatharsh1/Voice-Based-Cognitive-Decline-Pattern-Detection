import librosa
import numpy as np
from pydub import AudioSegment
import speech_recognition as sr
from typing import Dict, Tuple, List
import os
import tempfile
from pathlib import Path

class AudioProcessor:
    def __init__(self):
        # Set ffmpeg path
        ffmpeg_path = r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"
        if os.path.exists(ffmpeg_path):
            AudioSegment.converter = ffmpeg_path
            AudioSegment.ffmpeg = ffmpeg_path
            AudioSegment.ffprobe = ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe")
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return waveform and sample rate."""
        try:
            # First try loading with librosa
            audio, sr = librosa.load(file_path, sr=None)
            return audio, sr
        except Exception as e:
            print(f"Error loading with librosa: {str(e)}")
            try:
                # If that fails, try converting with pydub first
                audio = AudioSegment.from_file(file_path)
                # Convert to WAV format
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    audio.export(temp_file.name, format="wav")
                    # Now try loading the converted file
                    audio, sr = librosa.load(temp_file.name, sr=None)
                    os.unlink(temp_file.name)  # Clean up temp file
                    return audio, sr
            except Exception as e:
                print(f"Error loading audio file {file_path}: {str(e)}")
                return None, None

    def extract_audio_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract various audio features from the waveform."""
        features = {}
        
        try:
            # Pitch features
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            valid_pitches = pitches[magnitudes > 0]
            features['pitch_mean'] = float(np.mean(valid_pitches)) if len(valid_pitches) > 0 else 0.0
            features['pitch_std'] = float(np.std(valid_pitches)) if len(valid_pitches) > 0 else 0.0
            features['pitch_range'] = float(np.max(valid_pitches) - np.min(valid_pitches)) if len(valid_pitches) > 0 else 0.0
            
            # Speech rate (syllables per second)
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            features['speech_rate'] = float(len(librosa.onset.onset_detect(onset_envelope=onset_env)) / (len(audio) / sr))
            
            # Energy features
            rms = librosa.feature.rms(y=audio)
            features['energy_mean'] = float(np.mean(rms))
            features['energy_std'] = float(np.std(rms))
            features['energy_range'] = float(np.max(rms) - np.min(rms))
            
            # Zero crossing rate (indicator of speech vs silence)
            zcr = librosa.feature.zero_crossing_rate(y=audio)
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
            features['spectral_centroid_std'] = float(np.std(spectral_centroid))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            # Return default values if feature extraction fails
            features = {
                'pitch_mean': 0.0,
                'pitch_std': 0.0,
                'pitch_range': 0.0,
                'speech_rate': 0.0,
                'energy_mean': 0.0,
                'energy_std': 0.0,
                'energy_range': 0.0,
                'zcr_mean': 0.0,
                'zcr_std': 0.0,
                'spectral_centroid_mean': 0.0,
                'spectral_centroid_std': 0.0,
                'spectral_rolloff_mean': 0.0,
                'spectral_rolloff_std': 0.0
            }
            
        return features

    def detect_pauses(self, audio: np.ndarray, sr: int, threshold_db: float = -40) -> List[Tuple[float, float]]:
        """Detect pauses in speech using energy thresholding."""
        try:
            # Convert to dB
            db = librosa.amplitude_to_db(np.abs(audio))
            
            # Find segments below threshold
            pauses = []
            is_pause = False
            start_time = 0
            
            for i, value in enumerate(db):
                if value < threshold_db and not is_pause:
                    is_pause = True
                    start_time = i / sr
                elif value >= threshold_db and is_pause:
                    is_pause = False
                    end_time = i / sr
                    if end_time - start_time > 0.1:  # Only count pauses longer than 100ms
                        pauses.append((float(start_time), float(end_time)))
                        
            return pauses
        except Exception as e:
            print(f"Error detecting pauses: {str(e)}")
            return []

    def process_audio_file(self, file_path: str) -> Dict:
        """Process an audio file and return all extracted features."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        # Load audio
        audio, sr = self.load_audio(file_path)
        if audio is None:
            return {}
            
        # Extract features
        features = self.extract_audio_features(audio, sr)
        
        # Detect pauses
        pauses = self.detect_pauses(audio, sr)
        features['pause_count'] = len(pauses)
        features['pause_durations'] = [float(end - start) for start, end in pauses]
        features['pause_duration_mean'] = float(np.mean(features['pause_durations'])) if pauses else 0.0
        features['pause_duration_std'] = float(np.std(features['pause_durations'])) if pauses else 0.0
        
        # Add file name for reference
        features['file_name'] = os.path.basename(file_path)
        
        return features

    def _calculate_speech_rate(self, file_path):
        """Calculate speech rate in words per second."""
        try:
            # Convert to WAV if needed
            if not file_path.endswith('.wav'):
                audio = AudioSegment.from_file(file_path)
                wav_path = str(Path(file_path).with_suffix('.wav'))
                audio.export(wav_path, format='wav')
                file_path = wav_path
            
            # Transcribe audio
            recognizer = sr.Recognizer()
            with sr.AudioFile(file_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
            
            # Calculate speech rate
            words = len(text.split())
            duration = librosa.get_duration(filename=file_path)
            return words / duration if duration > 0 else 0
            
        except Exception as e:
            print(f"Error calculating speech rate: {str(e)}")
            return 0
    
    def _detect_pauses(self, audio, sr, threshold=0.01):
        """Detect pauses in audio."""
        # Calculate energy
        energy = librosa.feature.rms(y=audio)[0]
        
        # Find segments below threshold
        pauses = []
        in_pause = False
        start_time = 0
        
        for i, e in enumerate(energy):
            if e < threshold and not in_pause:
                in_pause = True
                start_time = i
            elif e >= threshold and in_pause:
                in_pause = False
                duration = (i - start_time) * (len(audio) / len(energy) / sr)
                if duration > 0.1:  # Only count pauses longer than 100ms
                    pauses.append(duration)
        
        return np.array(pauses) if pauses else np.array([0])