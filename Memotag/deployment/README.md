# Voice-Based Cognitive Decline Detection

## Project Overview
This project analyzes voice patterns to detect early signs of cognitive decline. It processes audio samples, extracts relevant features, and uses machine learning to identify patterns associated with different cognitive states.

## Features
- Audio processing and feature extraction
- Machine learning analysis
- Visualization of results
- Comprehensive reporting
- Test data generation

## Requirements
See requirements.txt for a complete list of dependencies.

## Installation
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Generate test data:
```bash
python src/generate_test_data.py
```

2. Run analysis:
```bash
python src/main.py --audio_dir data/audio_samples --output_dir results
```

3. View results in the results directory.

## Project Structure
- data/: Contains audio samples
- src/: Source code
- notebooks/: Jupyter notebooks for analysis
- results/: Analysis results and visualizations
- tests/: Test scripts

## License
This project is licensed under the MIT License.

## Contact
For questions or support, please contact [Your Contact Information]
