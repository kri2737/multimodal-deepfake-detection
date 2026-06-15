"""
Audio Classifier Model for Deepfake Detection

Processes mean MFCC features (40-dimensional) extracted from audio and
classifies them as real or fake.
"""

import torch
import torch.nn as nn


class AudioClassifier(nn.Module):
    """
    Audio classification model that takes 40-dimensional MFCC features
    as input and outputs binary classification (fake/real).
    
    Architecture:
    - Input: 40-dimensional mean MFCC features
    - Embedding layer: Projects to 128 dimensions
    - Hidden layers: 128 -> 64 -> 32 dimensions
    - Output: 2 classes (fake, real)
    """
    
    def __init__(self, input_dim=40, hidden_dims=None, num_classes=2):
        """
        Initialize the AudioClassifier.
        
        Args:
            input_dim (int): Dimension of input MFCC features (default: 40)
            hidden_dims (list): List of hidden layer dimensions (default: [128, 64, 32])
            num_classes (int): Number of output classes (default: 2)
        """
        super(AudioClassifier, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        
        # Build the network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Store intermediate feature dimension for fusion
        self.embedding_dim = hidden_dims[-1]  # 32 dimensions for fusion
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 40)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 2)
        """
        return self.network(x)
    
    def get_features(self, x):
        """
        Extract intermediate features (before the final classification layer)
        for use in feature-level fusion.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 40)
        
        Returns:
            torch.Tensor: Intermediate features of shape (batch_size, embedding_dim)
        """
        # Pass through all layers except the last one
        for layer in self.network[:-1]:
            x = layer(x)
        
        return x
    
    def get_embedding(self, x):
        """
        Alias for get_features. Returns embedding representation of audio.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 40)
        
        Returns:
            torch.Tensor: Embedding of shape (batch_size, embedding_dim)
        """
        return self.get_features(x)
