import os
import sys
import unittest
from pathlib import Path
import json
import numpy as np
from src.audio_processor import AudioProcessor
from src.ml_processor import MLProcessor

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create test directories
        cls.test_dir = Path("tests/test_data")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processors
        cls.audio_processor = AudioProcessor()
        cls.ml_processor = MLProcessor()
    
    def test_audio_processing(self):
        """Test audio processing functionality."""
        # Test with a sample audio file
        test_file = "data/audio_samples/normal_0.wav"
        if os.path.exists(test_file):
            features = self.audio_processor.process_audio_file(test_file)
            self.assertIsInstance(features, dict)
            self.assertTrue(len(features) > 0)
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        # Create test audio data
        test_audio = np.random.rand(16000)  # 1 second of audio at 16kHz
        features = self.audio_processor.extract_audio_features(test_audio, 16000)
        
        # Check feature types and ranges
        self.assertIsInstance(features, dict)
        for key, value in features.items():
            self.assertIsInstance(value, float)
            self.assertFalse(np.isnan(value))
    
    def test_ml_analysis(self):
        """Test machine learning analysis."""
        # Create test features
        test_features = {
            'pitch_mean': 150.0,
            'pitch_std': 20.0,
            'speech_rate': 4.5,
            'pause_duration_std': 0.3,
            'zcr_mean': 0.1
        }
        
        # Test feature preparation
        df = self.ml_processor.prepare_features([test_features])
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 1)
        
        # Test analysis
        results = self.ml_processor.analyze_features(df)
        self.assertIsNotNone(results)
        self.assertIn('pca_result', results)
        self.assertIn('clusters', results)
    
    def test_visualization(self):
        """Test visualization generation."""
        # Create test data
        test_features = [{
            'pitch_mean': 150.0,
            'pitch_std': 20.0,
            'speech_rate': 4.5,
            'pause_duration_std': 0.3,
            'zcr_mean': 0.1
        }]
        
        df = self.ml_processor.prepare_features(test_features)
        results = self.ml_processor.analyze_features(df)
        
        # Test visualization
        output_file = self.test_dir / "test_visualization.png"
        self.ml_processor.visualize_results(df, results, str(output_file))
        self.assertTrue(output_file.exists())
    
    def test_report_generation(self):
        """Test report generation."""
        # Create test data
        test_features = [{
            'pitch_mean': 150.0,
            'pitch_std': 20.0,
            'speech_rate': 4.5,
            'pause_duration_std': 0.3,
            'zcr_mean': 0.1
        }]
        
        df = self.ml_processor.prepare_features(test_features)
        results = self.ml_processor.analyze_features(df)
        
        # Test report generation
        report = self.ml_processor.generate_report(df, results)
        self.assertIsInstance(report, str)
        self.assertTrue(len(report) > 0)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Remove test directory
        if cls.test_dir.exists():
            for file in cls.test_dir.iterdir():
                file.unlink()
            cls.test_dir.rmdir()

if __name__ == '__main__':
    unittest.main() 