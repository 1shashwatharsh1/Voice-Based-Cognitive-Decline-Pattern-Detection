import os
import shutil
from pathlib import Path
import json
import base64
from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt
import seaborn as sns
import io

def generate_static_site():
    # Create output directory
    output_dir = Path('docs')
    output_dir.mkdir(exist_ok=True)
    
    # Copy static files
    static_dir = Path('static')
    if static_dir.exists():
        shutil.copytree(static_dir, output_dir / 'static', dirs_exist_ok=True)
    
    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader('src/templates'))
    
    # Generate index.html
    index_template = env.get_template('index.html')
    with open(output_dir / 'index.html', 'w', encoding='utf-8') as f:
        f.write(index_template.render())
    
    # Generate results.html with sample data
    results_template = env.get_template('results.html')
    
    # Sample features for demonstration
    sample_features = {
        'speech_rate': 2.5,
        'pause_duration': 0.8,
        'pitch_variation': 1.2,
        'energy_level': 0.9
    }
    
    # Generate sample visualization
    plt.style.use('seaborn-v0_8')  # Use the correct seaborn style
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=list(sample_features.keys()), y=list(sample_features.values()))
    for i, v in enumerate(sample_features.values()):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    plt.title('Audio Feature Analysis')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # Sample report
    sample_report = {
        'summary': 'This is a sample analysis report demonstrating the capabilities of the cognitive decline detection system.',
        'recommendations': [
            'Regular monitoring of speech patterns',
            'Consultation with healthcare professional',
            'Further cognitive assessment if needed'
        ]
    }
    
    # Write results page
    with open(output_dir / 'results.html', 'w', encoding='utf-8') as f:
        f.write(results_template.render(
            features=sample_features,
            plot_url=plot_url,
            report=sample_report
        ))
    
    # Create .nojekyll file to disable Jekyll processing
    (output_dir / '.nojekyll').touch()
    
    print(f"Static site generated in {output_dir}")

if __name__ == '__main__':
    generate_static_site()
