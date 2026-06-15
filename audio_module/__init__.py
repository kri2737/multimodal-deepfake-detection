"""
Audio Module for Deepfake Detection

Provides audio extraction, MFCC feature processing, and audio classification.
"""

from .audio_model import AudioClassifier
from .preprocess_audio import AudioPreprocessor

__all__ = ['AudioClassifier', 'AudioPreprocessor']
