"""
Fusion Module Package
Combines audio and video features for multimodal deepfake detection
"""

from .fusion_classifier import (
    FusionClassifier,
    VideoFeatureExtractor,
    AudioFeatureExtractor,
    MultimodalPredictor
)

__all__ = [
    'FusionClassifier',
    'VideoFeatureExtractor',
    'AudioFeatureExtractor',
    'MultimodalPredictor'
]

__version__ = '1.0.0'
