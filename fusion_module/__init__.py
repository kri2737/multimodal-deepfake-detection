"""
Fusion Module for Deepfake Detection

Provides multimodal fusion classifiers and end-to-end prediction pipelines.
"""

from .fusion_classifier import FusionClassifier, MultimodalPipeline

__all__ = ['FusionClassifier', 'MultimodalPipeline']
