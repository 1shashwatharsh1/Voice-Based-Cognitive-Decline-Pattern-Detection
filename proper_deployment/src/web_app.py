from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
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
import tempfile

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key')

# Configure paths
UPLOAD_FOLDER = tempfile.gettempdir()
DATA_DIR = os.environ.get('DATA_DIR', 'data')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATA_DIR'] = DATA_DIR
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3'}

# Ensure upload directory exists
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

# Initialize processors with data directory
audio_processor = AudioProcessor(data_dir=DATA_DIR)
ml_processor = MLProcessor(data_dir=DATA_DIR)

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

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Create a secure filename and save to temporary directory
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        try:
            # Process audio file
            features = audio_processor.process_audio_file(temp_path)
            
            # Generate visualization
            plt.figure(figsize=(12, 6))
            plt.style.use('seaborn')
            
            # Create bar plot of features
            feature_names = list(features.keys())
            feature_values = list(features.values())
            
            bars = plt.bar(feature_names, feature_values)
            plt.title('Audio Feature Analysis', fontsize=14, pad=20)
            plt.xlabel('Features', fontsize=12)
            plt.ylabel('Values', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')
            
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save plot to bytes
            img_bytes = io.BytesIO()
            plt.savefig(img_bytes, format='png', dpi=300, bbox_inches='tight')
            img_bytes.seek(0)
            plt.close()
            
            # Convert to base64
            img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
            
            # Prepare results
            results = {
                'features': features,
                'visualization': img_base64,
                'analysis': {
                    'speech_rate': f"{features['speech_rate']:.2f} words per second",
                    'pause_duration': f"{features['pause_duration']:.2f} seconds",
                    'pitch_variation': f"{features['pitch_std']:.2f} Hz",
                    'energy_level': f"{features['energy_mean']:.2f} dB"
                }
            }
            
            return render_template('results.html', results=results)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results/<filename>')
def results(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# This is required for Vercel
if __name__ == '__main__':
    app.run(debug=True)
else:
    # This is required for Vercel
    app = app 