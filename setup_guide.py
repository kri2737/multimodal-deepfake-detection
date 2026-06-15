"""
Quick Start Guide for Multimodal Deepfake Detection System

This script provides step-by-step instructions to get the system running.
"""

import os
import sys


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text):
    """Print a formatted section."""
    print(f"\n✓ {text}")


def check_requirements():
    """Check if all required packages are installed."""
    print_header("Checking Requirements")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'opencv': 'OpenCV',
        'mtcnn': 'MTCNN',
        'librosa': 'Librosa',
        'moviepy': 'MoviePy',
        'streamlit': 'Streamlit',
        'PIL': 'Pillow',
        'numpy': 'NumPy'
    }
    
    missing = []
    
    for package, name in required_packages.items():
        try:
            __import__(package if package != 'opencv' else 'cv2')
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\nInstall missing packages with:")
        print(f"  pip install -r requirements.txt")
        return False
    
    print_section("All requirements satisfied!")
    return True


def initialize_models():
    """Initialize audio and fusion models."""
    print_header("Initializing Models")
    
    try:
        from initialize_models import initialize_all_models
        import torch
        
        device = torch.device(
            'cuda' if torch.cuda.is_available() 
            else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            else 'cpu'
        )
        
        print(f"  Using device: {device}")
        
        models = initialize_all_models(device=device)
        
        print_section("Models initialized successfully!")
        
        for model_name, model_path in models.items():
            size = os.path.getsize(model_path) / (1024 * 1024)
            print(f"  {model_name}: {size:.2f} MB")
        
        return True
    
    except Exception as e:
        print(f"  ✗ Error initializing models: {e}")
        return False


def run_tests():
    """Run the test suite."""
    print_header("Running Tests")
    
    try:
        import unittest
        
        # Discover and run tests
        loader = unittest.TestLoader()
        suite = loader.discover('tests', pattern='test_*.py')
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        if result.wasSuccessful():
            print_section("All tests passed!")
            return True
        else:
            print(f"  ✗ {len(result.failures)} test(s) failed")
            return False
    
    except Exception as e:
        print(f"  ✗ Error running tests: {e}")
        return False


def verify_structure():
    """Verify directory structure."""
    print_header("Verifying Directory Structure")
    
    required_dirs = [
        'audio_module',
        'fusion_module',
        'video_module',
        'ui',
        'tests',
        'models',
        'data'
    ]
    
    required_files = {
        'initialize_models.py': 'Model initialization',
        'README.md': 'Documentation',
        'requirements.txt': 'Dependencies',
        'audio_module/audio_model.py': 'Audio model',
        'audio_module/preprocess_audio.py': 'Audio preprocessing',
        'fusion_module/fusion_classifier.py': 'Fusion classifier',
        'ui/app.py': 'Streamlit dashboard',
        'tests/test_fusion.py': 'Test suite'
    }
    
    all_ok = True
    
    # Check directories
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"  ✓ {dir_name}/")
        else:
            if dir_name not in ['models', 'data']:  # These might not exist yet
                print(f"  ✗ {dir_name}/ - MISSING")
                all_ok = False
    
    # Check files
    for file_path, description in required_files.items():
        if os.path.isfile(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING")
            all_ok = False
    
    if all_ok:
        print_section("Directory structure verified!")
    else:
        print("  Some files are missing. Please ensure all files were created correctly.")
    
    return all_ok


def print_next_steps():
    """Print next steps to run the system."""
    print_header("Next Steps")
    
    print("""
1. STREAMLIT DASHBOARD (Recommended for users):
   
   streamlit run ui/app.py
   
   Then open: http://localhost:8501

2. PYTHON API (For developers):
   
   from fusion_module.fusion_classifier import MultimodalPipeline
   
   pipeline = MultimodalPipeline()
   results = pipeline.predict('path/to/video.mp4')
   print(results)

3. COMMAND LINE (Quick test):
   
   python -c "
   from fusion_module.fusion_classifier import MultimodalPipeline
   pipeline = MultimodalPipeline()
   results = pipeline.predict('test_video.mp4')
   print(f'Prediction: {results[\"prediction\"]}')
   print(f'Confidence: {results[\"confidence\"]:.2%}')
   "

For more information, see README.md
    """)


def main():
    """Run the setup verification."""
    print("\n" + "=" * 70)
    print("  MULTIMODAL DEEPFAKE DETECTION SYSTEM")
    print("  Setup & Verification Script")
    print("=" * 70)
    
    # Verify directory structure
    if not verify_structure():
        print("\n⚠️  Some files are missing. Please check the setup.")
        return False
    
    # Check requirements
    if not check_requirements():
        print("\n⚠️  Please install missing requirements and try again.")
        return False
    
    # Initialize models
    if not initialize_models():
        print("\n⚠️  Failed to initialize models.")
        return False
    
    # Run tests
    if not run_tests():
        print("\n⚠️  Some tests failed. Please check the implementation.")
        return False
    
    # Print next steps
    print_next_steps()
    
    print("\n" + "=" * 70)
    print("  ✅ SETUP COMPLETE - READY TO USE!")
    print("=" * 70 + "\n")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
