# Multimodal Deepfake Detection System

A comprehensive  AI-powered system for  detecting deepfakes using combined video and audio analysis. This system leverages :

-  **Video Analysis**: ResNet18-based face classification
- ** Audio Analysis**: MFCC feature extraction and classification
- **Multimodal Fusion**: Feature-level and late fusion strategies
- ** User Interface**: Premium Streamlit dashboard

#🎯 Features

## Video Module
- Frame extraction from video files
- Face detection using MTCNN
- Face preprocessing and normalization
- ResNet18-based video classification
- Feature extraction for fusion

## Audio Module
- Audio extraction from video files
- MFCC (Mel-Frequency Cepstral Coefficient) feature extraction
- Audio classifier network
- Graceful handling of mute videos

### Fusion Module
- Feature-level fusion: Concatenates video and audio embeddings
- Late fusion: Combines predictions from individual models
- Edge case handling:
  - If no faces detected: uses audio model
  - If no audio found: uses video model
  - If both present: uses fusion model
- Complete end-to-end prediction pipeline

### Streamlit UI
- Video upload with support for MP4, AVI, MOV, MKV formats
- Real-time processing progress indicators
- Premium dashboard with gradient styling
- Face gallery display
- Audio waveform visualization
- MFCC  spectrogram visualization
- Confidence score display with detailed breakdown
- Individual model confidence scores

## 📦 Installation

## Prerequisites
- Python 3.8 or higher
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

1. **Clone the repository**
```bash
cd multimodal-deepfake-detection
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Initialize models**
```bash
python initialize_models.py
``

This will create the audio and fusion model weights in the `models/` directory.

## 🚀 Quick Start

### Option 1: Streamlit Dashboard

```bash
streamlit run ui/app.py
``

Then open your browser to `http://localhost:8501` and upload a video to analyze.

## Option 2: Python API

```python
from fusion_module.fusion_classifier import MultimodalPipeline
import torch

# Initialize pipeline
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipeline = MultimodalPipeline(device=device)

# Make prediction
results = pipeline.predict('path/to/video.mp4')

print(f"Prediction: {results['prediction']}")
print(f"Confidence: {results['confidence']:.2%}")
print(f"Video Confidence: {results.get('video_confidence', 'N/A')}")
print(f"Audio Confidence: {results.get('audio_confidence', 'N/A')}")
```

# 📁 Project Structure

```
multimodal-deepfake-detection/
├── audio_module/
│   ├── __init__.py
│   ├── audio_model.py           # AudioClassifier network
│   └── preprocess_audio.py      # Audio extraction and MFCC processing
├── fusion_module/
│   ├── __init__.py
│   ├── fusion_classifier.py     # Fusion models and pipeline
├── video_module/
│   ├── extract_frames.py
│   ├── detect_faces.py
│   ├── preprocess_faces.py
│   ├── train_model.py
│   ├── predict_image.py
│   ├── predict_video.py
│   └── evaluate_model.py
├── ui/
│   └── app.py                   # Streamlit dashboard
├── tests/
│   └── test_fusion.py           # Unit tests
├── models/
│   ├── video_model.pth
│   ├── audio_model.pth
│   └── fusion_model.pth
├── initialize_models.py         # Model initialization script
├── requirements.txt
└── README.md
```

## 🧠 Model Architecture

### Audio Classifier
- **Input**: 40-dimensional MFCC features (averaged over time)
- **Architecture**:
  - Embedding: 40 → 128 dimensions
  - Hidden layers: 128 → 64 → 32 dimensions
  - Output: 2 classes (fake/real)
- **Embedding dimension**: 32 (used for fusion)

### Fusion Classifier (Feature-Level)
- **Inputs**:
  - Video features: 512 dimensions (ResNet18 avgpool)
  - Audio features: 32 dimensions (AudioClassifier embedding)
- **Architecture**:
  - Concatenation: 512 + 32 = 544 dimensions
  - MLP: 544 → 256 → 128 → 2
  - Output: 2 classes (fake/real)

### Fusion Classifier (Late Fusion)
- Averages normalized predictions from video and audio models
- Simple lightweight alternative to feature-level fusion

## 📊 Usage Examples

### Example 1: Basic Prediction
```python
from fusion_module.fusion_classifier import MultimodalPipeline

pipeline = MultimodalPipeline()
results = pipeline.predict('test_video.mp4')

if results['prediction'] == 'FAKE':
    print(f"⚠️  DEEPFAKE DETECTED ({results['confidence']:.1%})")
else:
    print(f"✅ AUTHENTIC VIDEO ({results['confidence']:.1%})")
```

### Example 2: Detailed Analysis
```python
results = pipeline.predict('test_video.mp4')

print(f"Overall Prediction: {results['prediction']}")
print(f"Overall Confidence: {results['confidence']:.2%}")
print()

if results['has_video']:
    print(f"Video Model Confidence: {results['video_confidence']:.2%}")
    print(f"Faces Detected: {results['faces_detected']}")

if results['has_audio']:
    print(f"Audio Model Confidence: {results['audio_confidence']:.2%}")
```

### Example 3: Using Different Fusion Strategies
```python
# Feature-level fusion (default)
pipeline_feature = MultimodalPipeline(fusion_type='feature')
results1 = pipeline_feature.predict('video.mp4')

# Late fusion
pipeline_late = MultimodalPipeline(fusion_type='late')
results2 = pipeline_late.predict('video.mp4')
```

## 🧪 Testing

Run the test suite to verify system correctness:

```bash
python -m unittest tests/test_fusion.py -v
```

Tests cover:
- Audio model forward pass and feature extraction
- Fusion classifier feature and late fusion modes
- Model saving and loading
- Edge cases (no faces, no audio)
- Integration tests for complete pipeline

## 📈 Performance Considerations

### Hardware
- **CPU**: Works on CPU but slower
- **GPU**: Recommended for real-time processing
- **Memory**: ~2GB minimum recommended

### Processing Time
- **Frame extraction**: ~1s per 10 frames
- **Face detection**: ~0.5s per frame
- **Audio processing**: ~1s per video
- **Model inference**: ~0.1s per frame
- **Total**: ~3-5s for typical video

### Optimization Tips
1. Reduce max frames with `max_frames` parameter
2. Use GPU acceleration if available
3. Process multiple videos in parallel for batch operations
4. Cache audio extraction results

## 🔧 Configuration

### Streamlit Dashboard Settings
- **Fusion Strategy**: Choose between feature-level and late fusion
- **Max Frames**: Limit frames processed (10-100)

### Audio Processing Parameters
- **Sample Rate**: 22050 Hz (default)
- **MFCC Coefficients**: 40 (default)
- **FFT Window Size**: 2048 (default)
- **Hop Length**: 512 (default)

## 📝 Key Implementation Details

### Feature Extraction Pipeline
1. Extract frames from video (up to 30 frames)
2. Detect faces using MTCNN
3. Preprocess faces (resize to 224×224)
4. Extract ResNet18 features (avgpool layer → 512 dims)
5. Average features across all faces

### Audio Processing Pipeline
1. Extract audio from video using moviepy
2. Load audio with librosa (22050 Hz sample rate)
3. Compute MFCC features (40 coefficients)
4. Average MFCC over time axis
5. Pass to AudioClassifier for feature extraction (32 dims)

### Fusion Strategy
1. Get video features (512 dims) and audio features (32 dims)
2. Concatenate: 512 + 32 = 544 dimensions
3. Pass through MLP: 544 → 256 → 128 → 2
4. Apply softmax to get probabilities
5. Probability of class 0 (FAKE) is the confidence score

## ⚠️ Limitations and Future Work

### Current Limitations
- Models initialized with random weights (requires training on real data)
- Single face detection per frame
- Fixed input resolution (224×224)
- Limited to 30 frames per video

### Future Enhancements
- Multi-face detection and fusion
- Attention mechanisms for important frames
- Transformer-based audio analysis
- Real-time streaming processing
- Model quantization for mobile deployment
- Fine-tuning with real deepfake datasets

## 📄 License

[Add license information here]

## 👥 Authors

Member 3 - Multimodal Deepfake Detection System Implementation

## 🙏 Acknowledgments

- MTCNN for face detection
- librosa for audio processing
- PyTorch for deep learning framework
- Streamlit for UI framework
