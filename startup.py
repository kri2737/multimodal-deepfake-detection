"""
Startup Script for Multimodal Deepfake Detection System

This script initializes the system and starts the Streamlit dashboard.
"""

import os
import sys
import subprocess
import argparse


def print_banner():
    """Print ASCII banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║     MULTIMODAL DEEPFAKE DETECTION SYSTEM                      ║
    ║     Premium Dashboard with Video & Audio Analysis             ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def ensure_models_exist():
    """Ensure model files exist."""
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    required_models = [
        'audio_model.pth',
        'fusion_model.pth'
    ]
    
    missing_models = []
    for model in required_models:
        model_path = os.path.join(models_dir, model)
        if not os.path.exists(model_path):
            missing_models.append(model)
    
    if missing_models:
        print("\n⚠️  Missing model files:", ", ".join(missing_models))
        print("   Initializing models...")
        
        try:
            from initialize_models import initialize_all_models
            import torch
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            initialize_all_models(device=device)
            
            print("✓ Models initialized successfully!")
        except Exception as e:
            print(f"✗ Error initializing models: {e}")
            print("\n  To fix this, run: python initialize_models.py")
            return False
    else:
        print("✓ All models found")
    
    return True


def start_streamlit():
    """Start the Streamlit dashboard."""
    print("\n🚀 Starting Streamlit Dashboard...")
    print("=" * 70)
    print("\nOpening http://localhost:8501 in your browser...")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70 + "\n")
    
    try:
        subprocess.run(
            [sys.executable, '-m', 'streamlit', 'run', 'ui/app.py'],
            cwd=os.path.dirname(os.path.abspath(__file__)) or '.'
        )
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped")
    except Exception as e:
        print(f"✗ Error starting Streamlit: {e}")
        sys.exit(1)


def run_tests():
    """Run the test suite."""
    print("\n🧪 Running Test Suite...")
    print("=" * 70 + "\n")
    
    import unittest
    
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✓ All tests passed!")
    else:
        print(f"✗ {len(result.failures)} test(s) failed")
    print("=" * 70)
    
    return result.wasSuccessful()


def demo_prediction(video_path):
    """Run a demo prediction on a video file."""
    print(f"\n🎬 Running Demo Prediction on: {video_path}")
    print("=" * 70 + "\n")
    
    try:
        from fusion_module.fusion_classifier import MultimodalPipeline
        import torch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pipeline = MultimodalPipeline(device=device)
        
        print("Analyzing video...")
        results = pipeline.predict(video_path)
        
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Prediction: {results['prediction']}")
        print(f"Confidence: {results['confidence']:.1%}")
        print(f"Has Video: {results['has_video']}")
        print(f"Faces Detected: {results['faces_detected']}")
        print(f"Has Audio: {results['has_audio']}")
        
        if results['has_video']:
            print(f"Video Confidence: {results['video_confidence']:.1%}")
        
        if results['has_audio']:
            print(f"Audio Confidence: {results['audio_confidence']:.1%}")
        
        print("=" * 70)
        
    except Exception as e:
        print(f"✗ Error running prediction: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    print_banner()
    
    parser = argparse.ArgumentParser(
        description='Multimodal Deepfake Detection System'
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        default='dashboard',
        choices=['dashboard', 'test', 'predict'],
        help='Command to run (default: dashboard)'
    )
    
    parser.add_argument(
        'video',
        nargs='?',
        help='Video file path (for predict command)'
    )
    
    args = parser.parse_args()
    
    # Verify directory structure
    required_files = [
        'initialize_models.py',
        'ui/app.py',
        'tests/test_fusion.py',
        'fusion_module/fusion_classifier.py',
        'audio_module/audio_model.py'
    ]
    
    print("Checking installation...")
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"✗ Missing: {file_path}")
            print("\nPlease ensure all files have been created correctly.")
            sys.exit(1)
    
    print("✓ Installation verified")
    
    # Ensure models exist
    if not ensure_models_exist():
        sys.exit(1)
    
    print()
    
    # Execute command
    if args.command == 'dashboard':
        start_streamlit()
    
    elif args.command == 'test':
        success = run_tests()
        sys.exit(0 if success else 1)
    
    elif args.command == 'predict':
        if not args.video:
            print("✗ Please provide a video file path")
            print("  Usage: python startup.py predict <video_path>")
            sys.exit(1)
        
        if not os.path.exists(args.video):
            print(f"✗ Video file not found: {args.video}")
            sys.exit(1)
        
        demo_prediction(args.video)


if __name__ == '__main__':
    main()
