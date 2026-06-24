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
            self.simple_classifier = None
        elif fusion_type == 'late':
            # Late fusion: persistent small classifier (not recreated per forward)
            self.fusion_mlp = None
            self.simple_classifier = nn.Sequential(
                nn.Linear(max(video_feature_dim, audio_feature_dim), 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(self, video_features, audio_features):
        """
        Forward pass for fusion.
        """
        if self.fusion_type == 'feature':
            combined = torch.cat([video_features, audio_features], dim=1)
            return self.fusion_mlp(combined)
        elif self.fusion_type == 'late':
            # Normalize and average then apply persistent classifier
            video_norm = video_features / (torch.norm(video_features, dim=1, keepdim=True) + 1e-8)
            audio_norm = audio_features / (torch.norm(audio_features, dim=1, keepdim=True) + 1e-8)
            combined = (video_norm + audio_norm) / 2
            return self.simple_classifier(combined)


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
        """
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.fusion_type = fusion_type

        # Initialize models (safe loaders below handle corrupted files)
        self.video_model = self._load_video_model(video_model_path)
        self.audio_model = self._load_audio_model(audio_model_path)
        self.fusion_model = self._load_fusion_model(fusion_model_path)

        # Small classifier head to consume 512-dim features (do not feed 1D features back to ResNet)
        self.video_classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 2)
        ).to(self.device)

        # Try copying ResNet fc weights into video_classifier when shapes match
        try:
            if hasattr(self.video_model, 'fc') and isinstance(self.video_model.fc, nn.Linear):
                with torch.no_grad():
                    if self.video_classifier[1].weight.shape == self.video_model.fc.weight.shape:
                        self.video_classifier[1].weight.copy_(self.video_model.fc.weight)
                        self.video_classifier[1].bias.copy_(self.video_model.fc.bias)
        except Exception:
            pass

        # Initialize preprocessing
        self.face_detector = MTCNN()
        self.audio_preprocessor = AudioPreprocessor()

        # Image preprocessing (match training transforms)
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Set to eval
        self.video_model.eval()
        self.audio_model.eval()
        self.fusion_model.eval()
        self.video_classifier.eval()

    def _load_video_model(self, model_path):
        """Load ResNet18 video model (safe load + reinitialize on corruption)."""
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model = model.to(self.device)

        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception as e:
                print(f"Warning: Could not load video model: {e}. Attempting to reinitialize models.")
                try:
                    # Deferred import to avoid circular import at module import time
                    from initialize_models import initialize_all_models
                    initialize_all_models(device=self.device)
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    print("Reloaded video model after reinitialization.")
                except Exception as e2:
                    print(f"Warning: Reinitialization or reload failed: {e2}. Using random initialization.")
        return model

    def _load_audio_model(self, model_path):
        """Load audio model (safe load + reinitialize on corruption)."""
        model = AudioClassifier(input_dim=40, num_classes=2)
        model = model.to(self.device)

        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception as e:
                print(f"Warning: Could not load audio model: {e}. Attempting to reinitialize models.")
                try:
                    from initialize_models import initialize_all_models
                    initialize_all_models(device=self.device)
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    print("Reloaded audio model after reinitialization.")
                except Exception as e2:
                    print(f"Warning: Reinitialization or reload failed: {e2}. Using random initialization.")
        return model

    def _load_fusion_model(self, model_path):
        """Load fusion model (safe load + reinitialize on corruption)."""
        model = FusionClassifier(
            video_feature_dim=512,
            audio_feature_dim=32,
            fusion_type=self.fusion_type
        )
        model = model.to(self.device)

        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception as e:
                print(f"Warning: Could not load fusion model: {e}. Attempting to reinitialize models.")
                try:
                    from initialize_models import initialize_all_models
                    initialize_all_models(device=self.device)
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    print("Reloaded fusion model after reinitialization.")
                except Exception as e2:
                    print(f"Warning: Reinitialization or reload failed: {e2}. Using random initialization.")
        return model

    def extract_video_frames(self, video_path, max_frames=30):
        """
        Extract frames from video.
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
        Detect faces and preprocess.
        """
        face_tensors = []

        for frame in frames:
            try:
                detections = self.face_detector.detect_faces(frame)
                if detections:
                    x, y, w, h = detections[0]['box']
                    x = max(0, x)
                    y = max(0, y)
                    w = max(0, w)
                    h = max(0, h)
                    face = frame[y:y+h, x:x+w]

                    if face is None or face.size == 0:
                        continue

                    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    face_tensor = self.image_transform(face_pil)
                    face_tensors.append(face_tensor)
            except Exception as e:
                print(f"Warning: Error processing face: {e}")
                continue

        return face_tensors

    def get_video_features(self, face_tensors):
        """
        Extract 512-d video features (ResNet avgpool) and average across faces.
        Runs the full ResNet stem so outputs vary per input.
        """
        if not face_tensors:
            return torch.zeros(512).to(self.device)

        batch = torch.stack(face_tensors).to(self.device)

        with torch.no_grad():
            x = self.video_model.conv1(batch)
            if hasattr(self.video_model, 'bn1'):
                x = self.video_model.bn1(x)
            if hasattr(self.video_model, 'relu'):
                x = self.video_model.relu(x)
            if hasattr(self.video_model, 'maxpool'):
                x = self.video_model.maxpool(x)

            x = self.video_model.layer1(x)
            x = self.video_model.layer2(x)
            x = self.video_model.layer3(x)
            x = self.video_model.layer4(x)

            features = self.video_model.avgpool(x)

        features = features.view(features.size(0), -1)
        video_features = torch.mean(features, dim=0)
        return video_features

    def get_audio_features(self, video_path):
        """
        Extract audio embedding (32-d) from video.
        """
        result = self.audio_preprocessor.process_video(video_path)
        mfcc_features = result['mfcc_features']

        audio_features = torch.from_numpy(mfcc_features).float().to(self.device)
        audio_features = audio_features.unsqueeze(0)

        with torch.no_grad():
            audio_embedding = self.audio_model.get_embedding(audio_features)

        return audio_embedding.squeeze(0)

    def predict(self, video_path, return_intermediate=False):
        """
        End-to-end prediction.
        """
        results = {'has_video': False, 'has_audio': False, 'faces_detected': 0}

        try:
            frames = self.extract_video_frames(video_path)
            if not frames:
                raise ValueError("Could not extract frames from video")

            face_tensors = self.detect_and_preprocess_faces(frames)
            results['faces_detected'] = len(face_tensors)

            # Video
            has_video = len(face_tensors) > 0
            if has_video:
                video_features = self.get_video_features(face_tensors)
                results['has_video'] = True

                # classify 512-d features with small head
                with torch.no_grad():
                    video_input = video_features.unsqueeze(0).to(self.device)
                    video_logits = self.video_classifier(video_input)
                    video_probs = torch.softmax(video_logits, dim=1)
                    video_confidence = video_probs[0, 0].item()
                    results['video_confidence'] = video_confidence
            else:
                video_features = None
                video_confidence = None

                       # Audio
            audio_result = self.audio_preprocessor.process_video(video_path)
            has_audio = audio_result['has_audio']

            if has_audio:
                # Use raw MFCCs (shape: n_mfcc) for classification; use embedding for fusion
                mfcc = audio_result['mfcc_features']  # numpy array shape (n_mfcc,)
                audio_raw = torch.from_numpy(mfcc).float().to(self.device).unsqueeze(0)  # (1, n_mfcc)

                results['has_audio'] = True

                with torch.no_grad():
                    # Classification logits from raw MFCCs (matches AudioClassifier input_dim=40)
                    audio_logits = self.audio_model(audio_raw)
                    audio_probs = torch.softmax(audio_logits, dim=1)
                    audio_confidence = audio_probs[0, 0].item()
                    results['audio_confidence'] = audio_confidence

                    # Embedding (32-d) used for fusion
                    audio_embedding = self.audio_model.get_embedding(audio_raw).squeeze(0)
                    audio_features = audio_embedding
            else:
                audio_features = None
                audio_confidence = None

            # Fusion
            if has_video and has_audio:
                with torch.no_grad():
                    fusion_logits = self.fusion_model(
                        video_features.unsqueeze(0),
                        audio_features.unsqueeze(0)
                    )
                    fusion_probs = torch.softmax(fusion_logits, dim=1)
                    confidence = fusion_probs[0, 0].item()
            elif has_video:
                confidence = video_confidence
            elif has_audio:
                confidence = audio_confidence
            else:
                confidence = 0.5

            prediction = 'FAKE' if confidence > 0.5 else 'REAL'
            results['prediction'] = prediction
            results['confidence'] = confidence

            if return_intermediate:
                results['intermediate'] = {'frames_count': len(frames), 'audio_has_audio': has_audio}

            return results

        except Exception as e:
            print(f"Error in prediction pipeline: {e}")
            return {'prediction': 'ERROR', 'confidence': 0.0, 'error': str(e),
                    'has_video': False, 'has_audio': False, 'faces_detected': 0}