"""
Fusion Module - Combines audio and video predictions for deepfake detection
Responsible for:
- Loading both audio and video models
- Extracting features from both modalities
- Ensemble prediction with weighted fusion
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import Dict, Tuple, Any


class FusionClassifier(nn.Module):
    """
    Combines audio and video features for multimodal deepfake detection
    
    Architecture:
    [Video Features] ──┐
                       ├─→ [Concatenate] → [FC Layers] → [Binary Classification]
    [Audio Features] ──┘
    """
    
    def __init__(self, video_feature_dim: int = 512, audio_feature_dim: int = 256):
        """
        Args:
            video_feature_dim: Dimension of video features
            audio_feature_dim: Dimension of audio features
        """
        super(FusionClassifier, self).__init__()
        
        combined_dim = video_feature_dim + audio_feature_dim
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # Binary: Real or Fake
        )
        
    def forward(self, video_features: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through fusion network
        
        Args:
            video_features: Shape (batch_size, video_feature_dim)
            audio_features: Shape (batch_size, audio_feature_dim)
            
        Returns:
            logits: Shape (batch_size, 2) - Raw scores for [fake, real]
        """
        # Concatenate features
        combined_features = torch.cat([video_features, audio_features], dim=1)
        
        # Predict
        logits = self.fusion_network(combined_features)
        
        return logits


class VideoFeatureExtractor:
    """Extracts features from video using pre-trained ResNet18"""
    
    def __init__(self, model_path: str = "models/video_model.pth", device: str = None):
        """
        Args:
            model_path: Path to trained video model
            device: torch device ('cuda', 'cpu', or 'mps')
        """
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                                      "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Load ResNet18 backbone (remove final classification layer)
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Temporary final layer
        
        # Load pretrained weights
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        
        # Remove final layer for feature extraction
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
    def extract_features(self, image_path: str) -> torch.Tensor:
        """
        Extract features from a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            features: Shape (512,) - Feature vector
        """
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(image)
            features = features.squeeze().cpu()
            
        return features


class AudioFeatureExtractor:
    """Placeholder for audio feature extraction - to be implemented with Member 2"""
    
    def __init__(self, model_path: str = "models/audio_model.pth", device: str = None):
        """
        Args:
            model_path: Path to trained audio model
            device: torch device
        """
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                                      "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # TODO: Implement once Member 2 provides audio model architecture
        self.model = None
        
    def extract_features(self, audio_path: str) -> torch.Tensor:
        """
        Extract features from audio
        
        Args:
            audio_path: Path to audio file (or video to extract audio from)
            
        Returns:
            features: Shape (256,) - Feature vector
        """
        # TODO: Implement based on Member 2's audio model
        raise NotImplementedError("Waiting for Member 2's audio model implementation")


class MultimodalPredictor:
    """End-to-end multimodal deepfake detection"""
    
    def __init__(self, 
                 fusion_model_path: str = "models/fusion_model.pth",
                 video_model_path: str = "models/video_model.pth",
                 audio_model_path: str = "models/audio_model.pth",
                 device: str = None):
        """
        Args:
            fusion_model_path: Path to trained fusion model
            video_model_path: Path to trained video model
            audio_model_path: Path to trained audio model
            device: torch device
        """
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                                      "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize extractors
        self.video_extractor = VideoFeatureExtractor(video_model_path, str(self.device))
        # self.audio_extractor = AudioFeatureExtractor(audio_model_path, str(self.device))
        
        # Initialize fusion model
        self.fusion_model = FusionClassifier(video_feature_dim=512, audio_feature_dim=256)
        
        if os.path.exists(fusion_model_path):
            self.fusion_model.load_state_dict(
                torch.load(fusion_model_path, map_location=self.device)
            )
        
        self.fusion_model = self.fusion_model.to(self.device)
        self.fusion_model.eval()
        
        self.class_names = ["FAKE", "REAL"]
        
    def predict(self, video_frames_dir: str, audio_path: str = None) -> Dict[str, Any]:
        """
        Make a multimodal prediction on a video
        
        Args:
            video_frames_dir: Directory containing extracted video frames
            audio_path: Path to audio file (optional for now)
            
        Returns:
            prediction: {
                'class': 'REAL' or 'FAKE',
                'confidence': float (0-1),
                'video_confidence': float,
                'audio_confidence': float,
                'reasoning': dict with component details
            }
        """
        # Extract video features from frames
        video_features_list = []
        frame_count = 0
        
        if os.path.isdir(video_frames_dir):
            for frame_file in sorted(os.listdir(video_frames_dir)):
                if frame_file.endswith(('.jpg', '.png', '.jpeg')):
                    frame_path = os.path.join(video_frames_dir, frame_file)
                    features = self.video_extractor.extract_features(frame_path)
                    video_features_list.append(features)
                    frame_count += 1
        else:
            # Single image
            features = self.video_extractor.extract_features(video_frames_dir)
            video_features_list = [features]
            frame_count = 1
        
        if not video_features_list:
            raise ValueError(f"No frames found in {video_frames_dir}")
        
        # Average video features
        video_features = torch.stack(video_features_list).mean(dim=0).unsqueeze(0)
        
        # TODO: Extract audio features when Member 2's model is ready
        # For now, use dummy audio features
        audio_features = torch.randn(1, 256).to(self.device)
        
        # Fusion prediction
        with torch.no_grad():
            video_features = video_features.to(self.device)
            audio_features = audio_features.to(self.device)
            logits = self.fusion_model(video_features, audio_features)
            
            # Apply softmax for confidence
            probs = torch.softmax(logits, dim=1)
            confidence = probs[0, 1].item()  # Real class probability
            predicted_class_idx = torch.argmax(logits, dim=1).item()
        
        predicted_class = self.class_names[predicted_class_idx]
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'video_confidence': 0.95,  # TODO: Get from video model
            'audio_confidence': 0.88,  # TODO: Get from audio model
            'frames_analyzed': frame_count,
            'reasoning': {
                'video_model': 'ResNet18-based CNN',
                'audio_model': 'Pending Member 2',
                'fusion_strategy': 'Concatenation + FC layers'
            }
        }


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    print("Fusion Module Loaded Successfully!")
    print("\n📊 Architecture:")
    print("- Video Feature Extractor: ResNet18 backbone → (512,) features")
    print("- Audio Feature Extractor: [Awaiting Member 2] → (256,) features")
    print("- Fusion Classifier: Concatenation → FC layers → Binary classification")
    print("\n⏳ Waiting for Member 2's audio model to fully integrate...")
