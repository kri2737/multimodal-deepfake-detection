"""
Streamlit UI for Multimodal Deepfake Detection System

Premium dashboard for uploading videos, detecting deepfakes, and visualizing
intermediate results including face detections and audio analysis.
"""

import sys
import os

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from PIL import Image
import librosa
import librosa.display

from fusion_module.fusion_classifier import MultimodalPipeline
from audio_module.preprocess_audio import AudioPreprocessor


# Set page configuration
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .title {
            text-align: center;
            color: #1f77b4;
            margin-bottom: 1rem;
        }
        .metric-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 0.5rem;
            color: white;
            margin: 0.5rem 0;
        }
        .fake-box {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 1.5rem;
            border-radius: 0.5rem;
            color: white;
            margin: 0.5rem 0;
        }
        .real-box {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 1.5rem;
            border-radius: 0.5rem;
            color: white;
            margin: 0.5rem 0;
        }
        .progress-bar {
            background-color: #f0f0f0;
            border-radius: 0.5rem;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        .progress-fill {
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 1rem;
            border-radius: 0.5rem;
        }
        .result-container {
            background: #f8f9fa;
            padding: 2rem;
            border-radius: 1rem;
            margin: 1rem 0;
            border-left: 4px solid #1f77b4;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.session_state.pipeline = MultimodalPipeline(device=device)

if 'results' not in st.session_state:
    st.session_state.results = None

if 'faces_detected' not in st.session_state:
    st.session_state.faces_detected = []

if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None


# ============================================================================
# HEADER
# ============================================================================

st.markdown("<h1 class='title'>🎬 Multimodal Deepfake Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Advanced AI-powered detection using video and audio analysis</p>", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    fusion_type = st.radio(
        "Fusion Strategy",
        options=['feature', 'late'],
        help="Feature-level: combines embeddings | Late: averages predictions"
    )
    
    max_frames = st.slider(
        "Max Frames to Process",
        min_value=10,
        max_value=100,
        value=30,
        step=5
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This system detects deepfakes using:
    - **Video Analysis**: ResNet18 face classification
    - **Audio Analysis**: MFCC feature extraction
    - **Fusion**: Combined multimodal prediction
    """)

# ============================================================================
# UPLOAD SECTION
# ============================================================================

st.markdown("## 📹 Upload Video")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV"
    )

with col2:
    st.markdown("")
    st.markdown("")
    process_button = st.button("🚀 Analyze Video", use_container_width=True)

# ============================================================================
# PROCESSING AND RESULTS
# ============================================================================

if process_button and uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    try:
        # Progress indicators
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        stages = [
            ("📊 Frame Extraction", 0.15),
            ("🔍 Face Detection", 0.30),
            ("🎯 Preprocessing", 0.45),
            ("🔊 Audio Processing", 0.60),
            ("🧠 Model Inference", 0.85),
            ("🔗 Fusion Analysis", 1.00)
        ]
        
        for stage_name, progress in stages:
            with progress_placeholder.container():
                st.write(stage_name)
                st.progress(progress)
            
            with status_placeholder.container():
                st.caption(f"Processing... {int(progress * 100)}%")
        
        # Run prediction
        with status_placeholder.container():
            st.caption("Running inference...")
        
        results = st.session_state.pipeline.predict(tmp_path, return_intermediate=True)
        st.session_state.results = results
        
        # Clear progress indicators
        progress_placeholder.empty()
        status_placeholder.empty()
        
        # ====================================================================
        # RESULTS SECTION
        # ====================================================================
        
        st.markdown("---")
        st.markdown("## 🎯 Detection Results")
        
        # Main prediction
        prediction = results.get('prediction', 'ERROR')
        confidence = results.get('confidence', 0)
        
        if prediction == 'FAKE':
            result_html = f"""
            <div class='fake-box'>
                <h2>⚠️ DEEPFAKE DETECTED</h2>
                <p style='font-size: 24px; margin: 10px 0;'>{confidence:.1%} Confidence</p>
            </div>
            """
        elif prediction == 'REAL':
            result_html = f"""
            <div class='real-box'>
                <h2>✅ AUTHENTIC VIDEO</h2>
                <p style='font-size: 24px; margin: 10px 0;'>{confidence:.1%} Confidence</p>
            </div>
            """
        else:
            result_html = f"""
            <div class='metric-box'>
                <h2>⚠️ ERROR</h2>
                <p>{results.get('error', 'Unknown error')}</p>
            </div>
            """
        
        st.markdown(result_html, unsafe_allow_html=True)
        
        # Confidence metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Prediction", prediction)
        
        with col2:
            st.metric("Confidence", f"{confidence:.1%}")
        
        with col3:
            st.metric("Faces Detected", results.get('faces_detected', 0))
        
        # ====================================================================
        # DETAILED BREAKDOWN
        # ====================================================================
        
        st.markdown("## 📊 Detailed Analysis")
        
        breakdown_col1, breakdown_col2 = st.columns(2)
        
        with breakdown_col1:
            st.subheader("Video Analysis")
            if results.get('has_video'):
                video_conf = results.get('video_confidence', 0)
                st.metric("Video Confidence (FAKE)", f"{video_conf:.1%}")
                st.info(f"📹 {results.get('faces_detected', 0)} faces analyzed")
            else:
                st.warning("❌ No faces detected in video")
        
        with breakdown_col2:
            st.subheader("Audio Analysis")
            if results.get('has_audio'):
                audio_conf = results.get('audio_confidence', 0)
                st.metric("Audio Confidence (FAKE)", f"{audio_conf:.1%}")
                st.success("🔊 Audio track found and analyzed")
            else:
                st.warning("❌ No audio track found")
        
        # ====================================================================
        # VISUALIZATION SECTION
        # ====================================================================
        
        st.markdown("## 📈 Visualizations")
        
        # Extract faces for gallery
        try:
            cap = cv2.VideoCapture(tmp_path)
            face_images = []
            frame_count = 0
            
            from mtcnn import MTCNN
            detector = MTCNN()
            
            while frame_count < 5:  # Limit to 5 frames for display
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    detections = detector.detect_faces(frame)
                    if detections:
                        x, y, w, h = detections[0]['box']
                        face = frame[y:y+h, x:x+w]
                        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        face_images.append(face_rgb)
                except:
                    pass
                
                frame_count += 1
            
            cap.release()
            
            if face_images:
                st.subheader("👥 Detected Faces")
                
                # Display face gallery in grid
                cols = st.columns(min(len(face_images), 5))
                for idx, face_img in enumerate(face_images):
                    with cols[idx % 5]:
                        st.image(face_img, use_column_width=True, caption=f"Face {idx+1}")
        
        except Exception as e:
            st.warning(f"Could not extract faces for display: {e}")
        
        # Audio visualization
        if results.get('has_audio'):
            st.subheader("🔊 Audio Analysis")
            
            try:
                audio_preprocessor = AudioPreprocessor()
                
                # Extract audio
                audio_path = audio_preprocessor.extract_audio_from_video(tmp_path)
                
                if audio_path:
                    col1, col2 = st.columns(2)
                    
                    # Waveform
                    with col1:
                        st.markdown("**Waveform**")
                        
                        y, sr = librosa.load(audio_path, sr=22050)
                        
                        fig, ax = plt.subplots(figsize=(10, 3))
                        librosa.display.waveshow(y, sr=sr, ax=ax, color='#1f77b4')
                        ax.set_title('Audio Waveform')
                        ax.set_xlabel('Time (s)')
                        ax.set_ylabel('Amplitude')
                        st.pyplot(fig, use_container_width=True)
                        plt.close()
                    
                    # MFCC Spectrogram
                    with col2:
                        st.markdown("**MFCC Spectrogram**")
                        
                        mfcc = AudioPreprocessor.get_mfcc_spectrogram(
                            audio_path,
                            sr=22050,
                            n_mfcc=40
                        )
                        
                        fig, ax = plt.subplots(figsize=(10, 3))
                        img = librosa.display.specshow(
                            mfcc,
                            sr=22050,
                            hop_length=512,
                            x_axis='time',
                            y_axis='mel',
                            ax=ax,
                            cmap='viridis'
                        )
                        ax.set_title('MFCC Spectrogram')
                        fig.colorbar(img, ax=ax, format='%+2.0f dB')
                        st.pyplot(fig, use_container_width=True)
                        plt.close()
                    
                    # Cleanup
                    try:
                        os.remove(audio_path)
                    except:
                        pass
            
            except Exception as e:
                st.warning(f"Could not visualize audio: {e}")
        
        st.success("✅ Analysis complete!")
    
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
    
    finally:
        # Cleanup temporary file
        try:
            os.remove(tmp_path)
        except:
            pass

else:
    if st.button("📚 Run Demo Analysis"):
        st.info("📝 Demo mode: Upload a video file to see the system in action!")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; margin-top: 2rem;'>
    <p>Multimodal Deepfake Detection System v1.0</p>
    <p>Using ResNet18 Video Classification + MFCC Audio Features + Feature-Level Fusion</p>
</div>
""", unsafe_allow_html=True)
