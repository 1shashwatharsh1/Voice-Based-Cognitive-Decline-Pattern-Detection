import os
import subprocess
import sys

def convert_notebook_to_pdf(notebook_path, output_dir):
    """Convert a Jupyter notebook to PDF format."""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert notebook to PDF
        cmd = [
            'jupyter',
            'nbconvert',
            '--to',
            'pdf',
            '--output-dir',
            output_dir,
            notebook_path
        ]
        
        subprocess.run(cmd, check=True)
        print(f"Successfully converted {notebook_path} to PDF in {output_dir}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error converting notebook: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    notebook_path = "notebooks/presentation.ipynb"
    output_dir = "results"
    convert_notebook_to_pdf(notebook_path, output_dir) 