"""
Fusion Classifier for Multimodal Deepfake Detection

Combines video and audio predictions using feature-level fusion and late fusion
strategies. Implements the complete end-to-end prediction pipeline.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms, models
from mtcnn import MTCNN

from audio_module.audio_model import AudioClassifier
from audio_module.preprocess_audio import AudioPreprocessor
from video_module.predict_image import device as get_device


class FusionClassifier(nn.Module):
    """
    Multimodal fusion model combining video and audio features for deepfake detection.
    
    Supports two fusion strategies:
    1. Feature-level fusion: Concatenates embeddings and passes through MLP
    2. Late fusion: Averages probability predictions from both models
    """
    
    def __init__(self, 
                 video_feature_dim=512, 
                 audio_feature_dim=32,
                 hidden_dim=256,
                 fusion_type='feature'):
        """
        Initialize the FusionClassifier.
        
        Args:
            video_feature_dim (int): Dimension of video features (ResNet18 layer before FC)
            audio_feature_dim (int): Dimension of audio features (AudioClassifier embedding)
            hidden_dim (int): Hidden dimension for fusion MLP
            fusion_type (str): 'feature' for feature-level fusion, 'late' for late fusion
        """
        super(FusionClassifier, self).__init__()
        
        self.video_feature_dim = video_feature_dim
        self.audio_feature_dim = audio_feature_dim
        self.fusion_type = fusion_type
        
        if fusion_type == 'feature':
            # Feature-level fusion: concatenate embeddings
            combined_dim = video_feature_dim + audio_feature_dim
            
            self.fusion_mlp = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 2)  # Binary classification
            )
        elif fusion_type == 'late':
            # Late fusion: average logits
            self.fusion_mlp = None
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
    
    def forward(self, video_features, audio_features):
        """
        Forward pass for fusion.
        
        Args:
            video_features (torch.Tensor): Video embeddings of shape (batch_size, video_feature_dim)
            audio_features (torch.Tensor): Audio embeddings of shape (batch_size, audio_feature_dim)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 2)
        """
        if self.fusion_type == 'feature':
            # Concatenate features
            combined = torch.cat([video_features, audio_features], dim=1)
            return self.fusion_mlp(combined)
        elif self.fusion_type == 'late':
            # Average features (simple late fusion)
            # Normalize to same scale first
            video_norm = video_features / (torch.norm(video_features, dim=1, keepdim=True) + 1e-8)
            audio_norm = audio_features / (torch.norm(audio_features, dim=1, keepdim=True) + 1e-8)
            
            # Average and apply simple classifier
            combined = (video_norm + audio_norm) / 2
            
            # Simple 2-layer MLP for late fusion
            simple_classifier = nn.Sequential(
                nn.Linear(max(self.video_feature_dim, self.audio_feature_dim), 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            )
            
            return simple_classifier(combined)


class MultimodalPipeline:
    """
    End-to-end prediction pipeline combining video and audio models.
    """
    
    def __init__(self, 
                 video_model_path='models/video_model.pth',
                 audio_model_path='models/audio_model.pth',
                 fusion_model_path='models/fusion_model.pth',
                 fusion_type='feature',
                 device=None):
        """
        Initialize the multimodal pipeline.
        
        Args:
            video_model_path (str): Path to saved video model
            audio_model_path (str): Path to saved audio model
            fusion_model_path (str): Path to saved fusion model
            fusion_type (str): 'feature' or 'late' fusion
            device (torch.device): Device to run models on
        """
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.fusion_type = fusion_type
        
        # Initialize models
        self.video_model = self._load_video_model(video_model_path)
        self.audio_model = self._load_audio_model(audio_model_path)
        self.fusion_model = self._load_fusion_model(fusion_model_path)
        
        # Initialize preprocessing
        self.face_detector = MTCNN()
        self.audio_preprocessor = AudioPreprocessor()
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        # Set models to eval mode
        self.video_model.eval()
        self.audio_model.eval()
        self.fusion_model.eval()
    
    def _load_video_model(self, model_path):
        """Load ResNet18 video model."""
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception as e:
                print(f"Warning: Could not load video model: {e}. Using random initialization.")
        
        model = model.to(self.device)
        return model
    
    def _load_audio_model(self, model_path):
        """Load audio model."""
        model = AudioClassifier(input_dim=40, num_classes=2)
        
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception as e:
                print(f"Warning: Could not load audio model: {e}. Using random initialization.")
        
        model = model.to(self.device)
        return model
    
    def _load_fusion_model(self, model_path):
        """Load fusion model."""
        model = FusionClassifier(
            video_feature_dim=512,
            audio_feature_dim=32,
            fusion_type=self.fusion_type
        )
        
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception as e:
                print(f"Warning: Could not load fusion model: {e}. Using random initialization.")
        
        model = model.to(self.device)
        return model
    
    def extract_video_frames(self, video_path, max_frames=30):
        """
        Extract frames from video.
        
        Args:
            video_path (str): Path to video file
            max_frames (int): Maximum frames to extract
        
        Returns:
            list: List of frames (numpy arrays)
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frames.append(frame)
            frame_count += 1
            
            if frame_count >= max_frames:
                break
        
        cap.release()
        return frames
    
    def detect_and_preprocess_faces(self, frames):
        """
        Detect faces in frames and preprocess them.
        
        Args:
            frames (list): List of frames
        
        Returns:
            list: List of preprocessed face tensors
        """
        face_tensors = []
        
        for frame in frames:
            try:
                # Detect faces
                detections = self.face_detector.detect_faces(frame)
                
                if detections:
                    # Use first detected face
                    x, y, w, h = detections[0]['box']
                    face = frame[y:y+h, x:x+w]
                    
                    # Preprocess
                    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    face_tensor = self.image_transform(face_pil)
                    face_tensors.append(face_tensor)
            except Exception as e:
                print(f"Warning: Error processing face: {e}")
                continue
        
        return face_tensors
    
    def get_video_features(self, face_tensors):
        """
        Extract video features using ResNet18.
        
        Args:
            face_tensors (list): List of face tensors
        
        Returns:
            torch.Tensor: Aggregated video features of shape (512,)
        """
        if not face_tensors:
            # No faces detected - return zeros
            return torch.zeros(512).to(self.device)
        
        # Stack tensors
        batch = torch.stack(face_tensors).to(self.device)
        
        # Extract features (avgpool output, before FC layer)
        with torch.no_grad():
            features = self.video_model.avgpool(
                self.video_model.layer4(
                    self.video_model.layer3(
                        self.video_model.layer2(
                            self.video_model.layer1(
                                self.video_model.conv1(batch)
                            )
                        )
                    )
                )
            )
        
        # Flatten and average across faces
        features = features.view(features.size(0), -1)
        video_features = torch.mean(features, dim=0)
        
        return video_features
    
    def get_audio_features(self, video_path):
        """
        Extract audio features from video.
        
        Args:
            video_path (str): Path to video file
        
        Returns:
            torch.Tensor: Audio features of shape (32,) or zeros if no audio
        """
        result = self.audio_preprocessor.process_video(video_path)
        mfcc_features = result['mfcc_features']
        
        # Convert to tensor
        audio_features = torch.from_numpy(mfcc_features).float().to(self.device)
        audio_features = audio_features.unsqueeze(0)  # Add batch dimension
        
        # Get embedding
        with torch.no_grad():
            audio_embedding = self.audio_model.get_embedding(audio_features)
        
        return audio_embedding.squeeze(0)
    
    def predict(self, video_path, return_intermediate=False):
        """
        Complete end-to-end prediction pipeline.
        
        Args:
            video_path (str): Path to video file
            return_intermediate (bool): Whether to return intermediate results
        
        Returns:
            dict: Prediction results containing:
                - 'prediction': 'FAKE' or 'REAL'
                - 'confidence': float (0-1)
                - 'video_confidence': float (only if has_video)
                - 'audio_confidence': float (only if has_audio)
                - 'has_video': bool
                - 'has_audio': bool
                - 'faces_detected': int
                - (optional) intermediate results
        """
        results = {
            'has_video': False,
            'has_audio': False,
            'faces_detected': 0
        }
        
        try:
            # Extract frames
            frames = self.extract_video_frames(video_path)
            if not frames:
                raise ValueError("Could not extract frames from video")
            
            # Detect and preprocess faces
            face_tensors = self.detect_and_preprocess_faces(frames)
            results['faces_detected'] = len(face_tensors)
            
            # Get video features
            has_video = len(face_tensors) > 0
            if has_video:
                video_features = self.get_video_features(face_tensors)
                results['has_video'] = True
                
                # Get video prediction
                with torch.no_grad():
                    video_logits = self.video_model(video_features.unsqueeze(0))
                    video_probs = torch.softmax(video_logits, dim=1)
                    video_confidence = video_probs[0, 0].item()  # Probability of FAKE
                    results['video_confidence'] = video_confidence
            else:
                video_features = None
                video_confidence = None
            
            # Get audio features
            audio_result = self.audio_preprocessor.process_video(video_path)
            has_audio = audio_result['has_audio']
            
            if has_audio:
                audio_features = self.get_audio_features(video_path)
                results['has_audio'] = True
                
                # Get audio prediction
                with torch.no_grad():
                    audio_logits = self.audio_model(audio_features.unsqueeze(0))
                    audio_probs = torch.softmax(audio_logits, dim=1)
                    audio_confidence = audio_probs[0, 0].item()  # Probability of FAKE
                    results['audio_confidence'] = audio_confidence
            else:
                audio_features = None
                audio_confidence = None
            
            # Fusion logic
            if has_video and has_audio:
                # Use fusion model
                with torch.no_grad():
                    fusion_logits = self.fusion_model(
                        video_features.unsqueeze(0),
                        audio_features.unsqueeze(0)
                    )
                    fusion_probs = torch.softmax(fusion_logits, dim=1)
                    confidence = fusion_probs[0, 0].item()  # Probability of FAKE
            elif has_video:
                # Use video model only
                confidence = video_confidence
            elif has_audio:
                # Use audio model only
                confidence = audio_confidence
            else:
                # Neither video nor audio
                confidence = 0.5
            
            # Determine prediction
            prediction = 'FAKE' if confidence > 0.5 else 'REAL'
            results['prediction'] = prediction
            results['confidence'] = confidence
            
            # Return intermediate if requested
            if return_intermediate:
                results['intermediate'] = {
                    'frames_count': len(frames),
                    'audio_has_audio': has_audio
                }
            
            return results
        
        except Exception as e:
            print(f"Error in prediction pipeline: {e}")
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'error': str(e),
                'has_video': False,
                'has_audio': False,
                'faces_detected': 0
            }
