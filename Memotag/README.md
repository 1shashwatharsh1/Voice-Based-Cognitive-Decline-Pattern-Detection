# Voice-Based Cognitive Decline Pattern Detection

This project implements a proof-of-concept pipeline for detecting cognitive decline indicators through voice analysis.

## Project Structure
```
.
├── data/                  # Audio samples and processed data
├── src/                   # Source code
│   ├── audio_processor.py # Audio processing and feature extraction
│   ├── nlp_processor.py   # NLP processing and text analysis
│   ├── ml_processor.py    # Machine learning models
│   └── utils.py          # Utility functions
├── notebooks/            # Jupyter notebooks for analysis
├── tests/               # Test files
└── requirements.txt     # Project dependencies
```

## Features Extracted
- Pauses per sentence
- Hesitation markers (uh, um, etc.)
- Word recall issues
- Speech rate and pitch variability
- Naming & Word-Association Tasks
- Sentence Completion analysis

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

3. Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

## Usage
1. Place audio samples in the `data/` directory
2. Run the analysis pipeline:
```bash
python src/main.py
```

## License
MIT License 