import os
import shutil
from pathlib import Path

def package_deployment():
    # Create deployment directory structure
    deployment_dir = Path('deployment_package')
    src_dir = deployment_dir / 'src'
    data_dir = deployment_dir / 'data'
    
    # Create directories
    src_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy source files
    source_files = [
        'web_app.py',
        'audio_processor.py',
        'ml_processor.py',
        'nlp_processor.py',
        'generate_test_data.py',
        'main.py'
    ]
    
    for file in source_files:
        shutil.copy2(f'src/{file}', src_dir / file)
    
    # Copy templates and static files
    shutil.copytree('src/templates', src_dir / 'templates', dirs_exist_ok=True)
    shutil.copytree('src/static', src_dir / 'static', dirs_exist_ok=True)
    
    # Copy data files
    data_files = [
        'models',
        'reference',
        'config'
    ]
    
    for file in data_files:
        if os.path.exists(f'data/{file}'):
            shutil.copytree(f'data/{file}', data_dir / file, dirs_exist_ok=True)
    
    # Copy configuration files
    shutil.copy2('requirements.txt', deployment_dir)
    shutil.copy2('vercel.json', deployment_dir)
    
    print(f"Deployment package created at: {deployment_dir}")
    print("Total size:", sum(f.stat().st_size for f in deployment_dir.rglob('*') if f.is_file()) / (1024 * 1024), "MB")

if __name__ == '__main__':
    package_deployment() 