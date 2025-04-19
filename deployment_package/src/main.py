import os
import json
import argparse
from audio_processor import AudioProcessor
from ml_processor import MLProcessor

def process_audio_files(audio_dir, output_dir):
    """Process all audio files in the directory and save results."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize processors
    audio_processor = AudioProcessor()
    ml_processor = MLProcessor()
    
    # Process each audio file
    features_list = []
    for filename in os.listdir(audio_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(audio_dir, filename)
            print(f"Processing {filename}...")
            
            # Extract features
            features = audio_processor.process_audio_file(file_path)
            features['file_name'] = filename
            features_list.append(features)
    
    # Save raw features
    with open(os.path.join(output_dir, 'raw_features.json'), 'w') as f:
        json.dump(features_list, f, indent=2)
    
    # Prepare features for ML analysis
    df = ml_processor.prepare_features(features_list)
    
    # Analyze features
    results = ml_processor.analyze_features(df)
    
    # Generate visualizations
    ml_processor.visualize_results(df, results, os.path.join(output_dir, 'analysis_visualization.png'))
    
    # Generate report
    report = ml_processor.generate_report(df, results)
    with open(os.path.join(output_dir, 'analysis_report.txt'), 'w') as f:
        f.write(report)
    
    print("\nAnalysis complete!")
    print(f"Results saved to {output_dir}")
    print("\nSummary of findings:")
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process audio files for cognitive pattern analysis.')
    parser.add_argument('--audio_dir', type=str, required=True, help='Directory containing audio files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    
    args = parser.parse_args()
    process_audio_files(args.audio_dir, args.output_dir) 