"""
Model Initialization Script

Initializes and saves audio and fusion model weights with default initialization
so the system can run out-of-the-box without needing a full training run.
"""

import os
import torch
import torch.nn as nn

from audio_module.audio_model import AudioClassifier
from fusion_module.fusion_classifier import FusionClassifier


def initialize_audio_model(output_path='models/audio_model.pth', device=None):
    """
    Initialize and save the audio model with default weights.
    
    Args:
        output_path (str): Path where to save the model
        device (torch.device): Device to initialize on
    """
    device = device or torch.device('cpu')
    
    print(f"Initializing audio model...")
    
    # Create model
    model = AudioClassifier(input_dim=40, num_classes=2)
    model = model.to(device)
    
    # Initialize weights with small random values
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.kaiming_uniform_(param, a=0, mode='fan_in')
        else:
            nn.init.uniform_(param, -0.1, 0.1)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), output_path)
    print(f"✓ Audio model saved to {output_path}")
    
    return model


def initialize_fusion_model(output_path='models/fusion_model.pth', 
                           fusion_type='feature',
                           device=None):
    """
    Initialize and save the fusion model with default weights.
    
    Args:
        output_path (str): Path where to save the model
        fusion_type (str): 'feature' or 'late' fusion
        device (torch.device): Device to initialize on
    """
    device = device or torch.device('cpu')
    
    print(f"Initializing fusion model ({fusion_type} fusion)...")
    
    # Create model
    model = FusionClassifier(
        video_feature_dim=512,
        audio_feature_dim=32,
        hidden_dim=256,
        fusion_type=fusion_type
    )
    model = model.to(device)
    
    # Initialize weights with small random values
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.kaiming_uniform_(param, a=0, mode='fan_in')
        else:
            nn.init.uniform_(param, -0.1, 0.1)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), output_path)
    print(f"✓ Fusion model saved to {output_path}")
    
    return model


def initialize_all_models(models_dir='models', fusion_type='feature', device=None):
    """
    Initialize all required models.
    
    Args:
        models_dir (str): Directory to save models in
        fusion_type (str): 'feature' or 'late' fusion
        device (torch.device): Device to initialize on
    """
    device = device or torch.device('cpu')
    
    os.makedirs(models_dir, exist_ok=True)
    
    print("=" * 60)
    print("INITIALIZING MULTIMODAL DEEPFAKE DETECTION MODELS")
    print("=" * 60)
    
    # Initialize audio model
    audio_model_path = os.path.join(models_dir, 'audio_model.pth')
    initialize_audio_model(audio_model_path, device)
    
    # Initialize fusion model
    fusion_model_path = os.path.join(models_dir, 'fusion_model.pth')
    initialize_fusion_model(fusion_model_path, fusion_type, device)
    
    print("=" * 60)
    print("✓ All models initialized successfully!")
    print("=" * 60)
    
    return {
        'audio_model': audio_model_path,
        'fusion_model': fusion_model_path
    }


if __name__ == '__main__':
    # Determine device
    device = torch.device(
        'cuda' if torch.cuda.is_available() 
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    
    print(f"Using device: {device}")
    
    # Initialize all models
    initialize_all_models(device=device)
