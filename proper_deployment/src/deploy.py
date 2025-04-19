import os
import sys
import shutil
from pathlib import Path
import json
from datetime import datetime

def create_deployment_package():
    """Create a deployment package with all necessary files."""
    # Create deployment directory
    deploy_dir = Path("deployment")
    deploy_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    subdirs = ["data", "src", "notebooks", "results", "tests"]
    for subdir in subdirs:
        (deploy_dir / subdir).mkdir(exist_ok=True)
    
    # Copy source files
    src_files = [
        "audio_processor.py",
        "ml_processor.py",
        "main.py",
        "generate_test_data.py",
        "deploy.py"
    ]
    for file in src_files:
        shutil.copy2(f"src/{file}", deploy_dir / "src" / file)
    
    # Copy notebooks
    notebook_files = [
        "analysis_notebook.ipynb",
        "presentation.ipynb"
    ]
    for file in notebook_files:
        if os.path.exists(file):
            shutil.copy2(file, deploy_dir / "notebooks" / file)
    
    # Copy results
    results_files = [
        "analysis_report.txt",
        "presentation.md",
        "raw_features.json",
        "analysis_visualization.png"
    ]
    for file in results_files:
        if os.path.exists(f"results/{file}"):
            shutil.copy2(f"results/{file}", deploy_dir / "results" / file)
    
    # Copy requirements
    shutil.copy2("requirements.txt", deploy_dir / "requirements.txt")
    
    # Create README
    create_readme(deploy_dir)
    
    # Create manifest
    create_manifest(deploy_dir)
    
    print(f"Deployment package created in {deploy_dir}")

def create_readme(deploy_dir):
    """Create a README file for the deployment package."""
    readme_content = """# Voice-Based Cognitive Decline Detection

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
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
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
"""
    
    with open(deploy_dir / "README.md", "w") as f:
        f.write(readme_content)

def create_manifest(deploy_dir):
    """Create a manifest file for the deployment package."""
    manifest = {
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "source_files": [
                "audio_processor.py",
                "ml_processor.py",
                "main.py",
                "generate_test_data.py",
                "deploy.py"
            ],
            "notebooks": [
                "analysis_notebook.ipynb",
                "presentation.ipynb"
            ],
            "results": [
                "analysis_report.txt",
                "presentation.md",
                "raw_features.json",
                "analysis_visualization.png"
            ],
            "requirements": "requirements.txt"
        }
    }
    
    with open(deploy_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)

if __name__ == "__main__":
    create_deployment_package() 