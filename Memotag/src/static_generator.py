import os
import shutil
from pathlib import Path
import json
from audio_processor import AudioProcessor
from ml_processor import MLProcessor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns

def generate_plot(features_data):
    """Generate plot in a thread-safe way"""
    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn')
    
    features_names = list(features_data.keys())
    features_values = list(features_data.values())
    
    # Create bar plot
    bars = plt.bar(range(len(features_data)), features_values, color='#4e79a7')
    plt.xticks(range(len(features_data)), features_names, rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.title('Feature Analysis', pad=20)
    plt.ylabel('Normalized Value')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot to bytes
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=300)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def generate_static_site():
    # Create output directory
    output_dir = Path('docs')  # GitHub Pages uses 'docs' folder
    output_dir.mkdir(exist_ok=True)
    
    # Copy static assets
    static_dir = Path('src/static')
    if static_dir.exists():
        shutil.copytree(static_dir, output_dir / 'static', dirs_exist_ok=True)
    
    # Process audio samples
    audio_dir = Path('data/audio_samples')
    results = []
    
    audio_processor = AudioProcessor()
    ml_processor = MLProcessor()
    
    for audio_file in audio_dir.glob('*.wav'):
        try:
            # Process audio file
            features = audio_processor.process_audio_file(str(audio_file))
            
            # Analyze features
            df = ml_processor.prepare_features([features])
            analysis_results = ml_processor.analyze_features(df)
            
            # Generate plot
            plot_url = generate_plot(analysis_results['raw_features']) if 'raw_features' in analysis_results else None
            
            # Generate report
            report = ml_processor.generate_report(df, analysis_results)
            
            # Save results
            result = {
                'filename': audio_file.name,
                'features': features,
                'plot_url': plot_url,
                'report': report
            }
            results.append(result)
            
            # Create individual result page
            with open(output_dir / f'results_{audio_file.stem}.html', 'w') as f:
                f.write(render_results_page(result))
            
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
    
    # Create index page
    with open(output_dir / 'index.html', 'w') as f:
        f.write(render_index_page(results))
    
    # Create 404 page
    with open(output_dir / '404.html', 'w') as f:
        f.write(render_404_page())

def render_index_page(results):
    """Render the index page with sample results"""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Cognitive Decline Detection</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            /* Include all the CSS from the original index.html */
            {open('src/templates/index.html').read().split('<style>')[1].split('</style>')[0]}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="hero-section">
                <div class="upload-container">
                    <div class="header">
                        <h1>Cognitive Decline Detection</h1>
                        <p class="lead">Pre-analyzed audio samples and their cognitive patterns</p>
                    </div>
                    
                    <div class="results-grid">
                        <h3 class="text-center mb-4">Sample Analyses</h3>
                        <div class="row">
                            {''.join([render_sample_card(result) for result in results])}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """

def render_sample_card(result):
    """Render a card for a sample result"""
    return f"""
    <div class="col-md-6 mb-4">
        <div class="card feature-card">
            <div class="card-body">
                <h5 class="card-title">{result['filename']}</h5>
                <p class="feature-value">Speech Rate: {result['features']['speech_rate']:.2f} words/second</p>
                <p class="feature-value">Pause Duration: {result['features']['pause_duration_mean']:.2f} seconds</p>
                <a href="results_{Path(result['filename']).stem}.html" class="btn btn-primary mt-2">View Analysis</a>
            </div>
        </div>
    </div>
    """

def render_results_page(result):
    """Render an individual results page"""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Analysis Results - {result['filename']}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            /* Include all the CSS from the original results.html */
            {open('src/templates/results.html').read().split('<style>')[1].split('</style>')[0]}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="results-container">
                <div class="header">
                    <h1>Analysis Results</h1>
                </div>
                
                <div class="file-info">
                    <h5>File Analyzed: {result['filename']}</h5>
                </div>
                
                <h3 class="mt-4">Key Features</h3>
                <div class="row">
                    <div class="col-md-6">
                        <div class="card feature-card">
                            <div class="card-body">
                                <h5 class="card-title">Speech Rate</h5>
                                <p class="feature-value">{result['features']['speech_rate']:.2f} words/second</p>
                                <p class="interpretation">
                                    {get_interpretation('speech_rate', result['features']['speech_rate'])}
                                </p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card feature-card">
                            <div class="card-body">
                                <h5 class="card-title">Pause Duration</h5>
                                <p class="feature-value">{result['features']['pause_duration_mean']:.2f} seconds</p>
                                <p class="interpretation">
                                    {get_interpretation('pause_duration', result['features']['pause_duration_mean'])}
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
                
                {f'<div class="visualization"><h3>Feature Analysis</h3><img src="data:image/png;base64,{result["plot_url"]}" alt="Feature Analysis" class="img-fluid"></div>' if result['plot_url'] else ''}
                
                <div class="report-section">
                    <h3>Analysis Report</h3>
                    <div class="card">
                        <div class="card-body">
                            <pre class="mb-0">{result['report']}</pre>
                        </div>
                    </div>
                </div>
                
                <div class="text-center mt-4">
                    <a href="index.html" class="btn btn-primary">Back to Samples</a>
                </div>
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """

def get_interpretation(feature, value):
    """Get interpretation text for a feature value"""
    if feature == 'speech_rate':
        if value > 3.0:
            return '<span class="status-indicator status-normal"></span>Normal speech rate'
        elif value > 2.0:
            return '<span class="status-indicator status-mild"></span>Slightly reduced speech rate'
        else:
            return '<span class="status-indicator status-significant"></span>Significantly reduced speech rate'
    elif feature == 'pause_duration':
        if value < 0.5:
            return '<span class="status-indicator status-normal"></span>Normal pause duration'
        elif value < 1.0:
            return '<span class="status-indicator status-mild"></span>Slightly increased pauses'
        else:
            return '<span class="status-indicator status-significant"></span>Significantly increased pauses'
    return ''

def render_404_page():
    """Render a 404 error page"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Page Not Found</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Poppins', sans-serif;
                background-color: #f8f9fa;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .error-container {
                text-align: center;
                padding: 2rem;
            }
            .error-code {
                font-size: 6rem;
                font-weight: 700;
                color: #4e79a7;
            }
            .error-message {
                font-size: 1.5rem;
                color: #666;
                margin-bottom: 2rem;
            }
        </style>
    </head>
    <body>
        <div class="error-container">
            <div class="error-code">404</div>
            <div class="error-message">Page Not Found</div>
            <a href="index.html" class="btn btn-primary">Back to Home</a>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    generate_static_site() 