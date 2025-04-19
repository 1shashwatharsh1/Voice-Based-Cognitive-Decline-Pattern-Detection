from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
from werkzeug.utils import secure_filename
from pathlib import Path
from audio_processor import AudioProcessor
from ml_processor import MLProcessor
import json
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key')
app.config['UPLOAD_FOLDER'] = '/tmp/audio_samples'  # Use /tmp for serverless environment
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3'}

# Ensure upload directory exists
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

# Add CSP headers
@app.after_request
def add_security_headers(response):
    csp = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com 'unsafe-inline'; "
        "style-src 'self' https://cdn.jsdelivr.net https://fonts.googleapis.com https://cdnjs.cloudflare.com 'unsafe-inline'; "
        "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "form-action 'self'; "
        "base-uri 'self'; "
        "object-src 'none'; "
        "media-src 'self'; "
        "worker-src 'self'; "
        "child-src 'self'; "
        "frame-src 'self'; "
        "manifest-src 'self'"
    )
    response.headers['Content-Security-Policy'] = csp
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the audio file
            audio_processor = AudioProcessor()
            features = audio_processor.process_audio_file(filepath)
            
            # Analyze features
            ml_processor = MLProcessor()
            df = ml_processor.prepare_features([features])
            results = ml_processor.analyze_features(df)
            
            # Generate visualization in a thread-safe way
            if 'raw_features' in results:
                plot_url = generate_plot(results['raw_features'])
            else:
                plot_url = None
            
            # Generate report
            report = ml_processor.generate_report(df, results)
            
            # Clean up the temporary file
            os.remove(filepath)
            
            return render_template('results.html', 
                                 filename=filename,
                                 features=features,
                                 plot_url=plot_url,
                                 report=report)
        
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(request.url)
    
    flash('Invalid file type')
    return redirect(request.url)

@app.route('/results/<filename>')
def results(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# This is required for Vercel
if __name__ == '__main__':
    app.run(debug=True)
else:
    # This is required for Vercel
    app = app 