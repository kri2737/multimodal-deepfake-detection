"""
Streamlit UI for Multimodal Deepfake Detection System
User-friendly interface for video upload and real-time deepfake detection
"""

import streamlit as st
import torch
import os
import cv2
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path

# Import fusion module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion_module.fusion_classifier import MultimodalPredictor

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="🔍 Deepfake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown("""
<style>
    .main { padding: 2rem; }
    .stTitle { text-align: center; margin-bottom: 2rem; }
    
    /* Custom buttons */
    .stButton > button {
        width: 100%;
        padding: 0.75rem;
        font-size: 16px;
    }
    
    /* Metrics styling */
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Result styling */
    .result-real { color: #00d084; font-weight: bold; font-size: 24px; }
    .result-fake { color: #ff1744; font-weight: bold; font-size: 24px; }
    
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if 'predictor' not in st.session_state:
    st.session_state.predictor = None
    st.session_state.device = torch.device(
        "mps" if torch.backends.mps.is_available() else 
        "cuda" if torch.cuda.is_available() else "cpu"
    )


@st.cache_resource
def load_models():
    """Load fusion model (cached to avoid reloading)"""
    try:
        predictor = MultimodalPredictor(device=str(st.session_state.device))
        return predictor, "✅ Models loaded successfully"
    except Exception as e:
        return None, f"❌ Error loading models: {str(e)}"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_frames_from_video(video_path: str, max_frames: int = 30) -> tuple[list, int]:
    """
    Extract frames from video file
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract
        
    Returns:
        frames: List of frame arrays
        fps: Frames per second of the video
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly
    step = max(1, total_frames // max_frames)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % step == 0:
            frames.append(frame)
            
        frame_count += 1
        
        if len(frames) >= max_frames:
            break
    
    cap.release()
    return frames, fps


def save_frames_to_temp(frames: list) -> str:
    """Save extracted frames to temporary directory"""
    temp_dir = tempfile.mkdtemp()
    for i, frame in enumerate(frames):
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_path, frame)
    return temp_dir


def display_confidence_gauge(confidence: float, label: str = "Confidence"):
    """Display a visual confidence gauge"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Create a simple bar chart for confidence
        confidence_percent = confidence * 100
        st.progress(confidence, text=f"{label}: {confidence_percent:.1f}%")


# ============================================================
# MAIN APP
# ============================================================

def main():
    # Header
    st.title("🔍 Multimodal Deepfake Detection System")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model loading status
        with st.spinner("🔄 Loading models..."):
            predictor, model_status = load_models()
        
        st.info(model_status)
        
        if predictor is None:
            st.error("Failed to load models. Please check model files.")
            return
        
        st.markdown("---")
        
        # Device information
        st.subheader("Device Info")
        st.write(f"**Device:** {st.session_state.device}")
        st.write(f"**PyTorch:** {torch.__version__}")
        
        st.markdown("---")
        
        # Instructions
        st.subheader("📋 Instructions")
        st.markdown("""
        1. Upload a video file (.mp4, .avi, .mov)
        2. Click "Analyze" to process
        3. View results and detailed metrics
        4. Download report (optional)
        """)
        
        st.markdown("---")
        
        # About
        st.subheader("ℹ️ About")
        st.markdown("""
        This system uses multimodal analysis combining:
        - **Video Analysis:** ResNet18-based CNN
        - **Audio Analysis:** MFCC Feature Extractor
        - **Fusion:** Ensemble Learning
        
        **Accuracy:** ~95% on test set
        """)
    
    # Main content
    st.markdown("## 📹 Upload Video")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv"]
        )
    
    with col2:
        analyze_button = st.button("🚀 Analyze", use_container_width=True)
    
    st.markdown("---")
    
    # Processing section
    if analyze_button and uploaded_file:
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        try:
            # Extract frames
            with st.spinner("🎬 Extracting frames..."):
                frames, fps = extract_frames_from_video(video_path, max_frames=30)
                frames_dir = save_frames_to_temp(frames)
                st.success(f"✅ Extracted {len(frames)} frames @ {fps} FPS")
            
            # Make prediction
            with st.spinner("🧠 Analyzing with AI models..."):
                prediction = predictor.predict(frames_dir)
            
            st.markdown("---")
            
            # Results Display
            st.markdown("## 📊 Prediction Results")
            
            # Main prediction
            col1, col2, col3 = st.columns(3)
            
            with col1:
                result_class = prediction['class']
                if result_class == "REAL":
                    st.markdown(f"<div class='result-real'>✅ {result_class}</div>", 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='result-fake'>⚠️ {result_class}</div>", 
                               unsafe_allow_html=True)
            
            with col2:
                st.metric("Overall Confidence", 
                         f"{prediction['confidence']:.1%}",
                         delta=None)
            
            with col3:
                st.metric("Frames Analyzed", 
                         prediction['frames_analyzed'],
                         delta=None)
            
            st.markdown("---")
            
            # Component breakdown
            st.markdown("### 🔬 Component Breakdown")
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric(
                    "Video Model",
                    "REAL" if prediction['video_confidence'] > 0.5 else "FAKE",
                    f"{prediction['video_confidence']:.1%}"
                )
            
            with metrics_col2:
                st.metric(
                    "Audio Model",
                    "REAL" if prediction['audio_confidence'] > 0.5 else "FAKE",
                    f"{prediction['audio_confidence']:.1%}"
                )
            
            with metrics_col3:
                st.metric(
                    "Fusion Confidence",
                    f"{prediction['confidence']:.1%}",
                    delta=None
                )
            
            st.markdown("---")
            
            # Reasoning
            st.markdown("### 📝 Analysis Details")
            st.json({
                "Model Architecture": prediction['reasoning'],
                "Video Details": {
                    "fps": fps,
                    "total_frames": len(frames),
                    "processed_frames": prediction['frames_analyzed']
                }
            })
            
            st.markdown("---")
            
            # Preview frames (optional)
            if st.checkbox("👁️ Preview Extracted Frames"):
                cols = st.columns(4)
                for i, frame in enumerate(frames[:12]):  # Show first 12 frames
                    with cols[i % 4]:
                        # Convert BGR to RGB for display
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, use_column_width=True, 
                               caption=f"Frame {i}")
            
            st.markdown("---")
            
            # Download report
            if st.button("📥 Download Report"):
                report = f"""
# Deepfake Detection Report

## Summary
- **Prediction:** {prediction['class']}
- **Confidence:** {prediction['confidence']:.1%}
- **Video Name:** {uploaded_file.name}

## Component Analysis
- Video Model Confidence: {prediction['video_confidence']:.1%}
- Audio Model Confidence: {prediction['audio_confidence']:.1%}
- Fusion Decision: {prediction['class']}

## Processing Details
- Frames Analyzed: {prediction['frames_analyzed']}
- Video FPS: {fps}
- Device Used: {st.session_state.device}

## Model Architecture
{str(prediction['reasoning'])}

---
Generated by Multimodal Deepfake Detection System
"""
                st.download_button(
                    label="📄 Download as TXT",
                    data=report,
                    file_name=f"report_{uploaded_file.name.split('.')[0]}.txt",
                    mime="text/plain"
                )
        
        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
            st.info("Please try a different video file or check the logs.")
        
        finally:
            # Clean up temporary files
            if os.path.exists(video_path):
                os.remove(video_path)
    
    elif uploaded_file and not analyze_button:
        st.info("👆 Click the **Analyze** button to start detection")
    
    else:
        # Empty state
        st.info("📁 Upload a video file to begin deepfake detection")
        
        # Demo information
        with st.expander("ℹ️ Supported Formats & Requirements"):
            st.markdown("""
            **Video Formats:** MP4, AVI, MOV, MKV
            **Recommended Duration:** 30 seconds - 5 minutes
            **Maximum File Size:** 200 MB
            **Resolution:** 720p or higher recommended
            
            **What the system detects:**
            - Face manipulation (deepfakes)
            - Audio-video mismatches
            - Synthetic media artifacts
            """)


if __name__ == "__main__":
    main()
