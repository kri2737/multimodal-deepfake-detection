"""
Comprehensive testing suite for multimodal deepfake detection system
Tests for fusion module, UI components, and end-to-end integration
"""

import pytest
import torch
import os
import tempfile
import numpy as np
from pathlib import Path

# Import modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion_module.fusion_classifier import (
    FusionClassifier, 
    VideoFeatureExtractor, 
    MultimodalPredictor
)


# ============================================================
# TEST FIXTURES
# ============================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def dummy_device():
    """Get device for testing"""
    return torch.device("cpu")


@pytest.fixture
def fusion_model(dummy_device):
    """Create dummy fusion model"""
    model = FusionClassifier(video_feature_dim=512, audio_feature_dim=256)
    model = model.to(dummy_device)
    return model


# ============================================================
# TEST: FUSION MODULE
# ============================================================

class TestFusionClassifier:
    """Test fusion classifier module"""
    
    def test_fusion_model_creation(self, fusion_model):
        """Test that fusion model can be created"""
        assert fusion_model is not None
        assert isinstance(fusion_model, FusionClassifier)
    
    def test_forward_pass_dimensions(self, fusion_model, dummy_device):
        """Test forward pass with correct dimensions"""
        batch_size = 4
        video_features = torch.randn(batch_size, 512).to(dummy_device)
        audio_features = torch.randn(batch_size, 256).to(dummy_device)
        
        output = fusion_model(video_features, audio_features)
        
        assert output.shape == (batch_size, 2), f"Expected shape (4, 2), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
    
    def test_softmax_confidence_range(self, fusion_model, dummy_device):
        """Test that softmax produces valid probabilities"""
        video_features = torch.randn(1, 512).to(dummy_device)
        audio_features = torch.randn(1, 256).to(dummy_device)
        
        logits = fusion_model(video_features, audio_features)
        probs = torch.softmax(logits, dim=1)
        
        # Check probabilities are in valid range
        assert torch.all(probs >= 0) and torch.all(probs <= 1), "Probabilities out of [0,1] range"
        assert torch.allclose(probs.sum(dim=1), torch.ones(1)), "Probabilities don't sum to 1"
    
    def test_different_batch_sizes(self, fusion_model, dummy_device):
        """Test forward pass with different batch sizes"""
        for batch_size in [1, 2, 4, 8, 16]:
            video_features = torch.randn(batch_size, 512).to(dummy_device)
            audio_features = torch.randn(batch_size, 256).to(dummy_device)
            output = fusion_model(video_features, audio_features)
            assert output.shape == (batch_size, 2)
    
    def test_gradient_flow(self, fusion_model, dummy_device):
        """Test that gradients can flow through fusion model"""
        video_features = torch.randn(1, 512, requires_grad=True, device=dummy_device)
        audio_features = torch.randn(1, 256, requires_grad=True, device=dummy_device)
        
        output = fusion_model(video_features, audio_features)
        loss = output.sum()
        loss.backward()
        
        assert video_features.grad is not None, "Video features gradient is None"
        assert audio_features.grad is not None, "Audio features gradient is None"


class TestVideoFeatureExtractor:
    """Test video feature extraction"""
    
    def test_feature_extractor_creation(self):
        """Test that feature extractor can be created"""
        # Mock video model for testing
        # This would require a real video_model.pth file
        pytest.skip("Requires actual video_model.pth file")
    
    def test_feature_dimensions(self):
        """Test that extracted features have correct dimensions"""
        pytest.skip("Requires actual video_model.pth file")


# ============================================================
# TEST: MULTIMODAL PREDICTOR
# ============================================================

class TestMultimodalPredictor:
    """Test end-to-end multimodal prediction"""
    
    def test_predictor_initialization(self):
        """Test that predictor can be initialized"""
        pytest.skip("Requires actual model files")
    
    def test_prediction_output_format(self):
        """Test that predictions have correct output format"""
        pytest.skip("Requires actual frames")


# ============================================================
# TEST: UTILITY FUNCTIONS
# ============================================================

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_softmax_output(self):
        """Test softmax produces valid probabilities"""
        logits = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
        probs = torch.softmax(logits, dim=1)
        
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2))
    
    def test_argmax_selection(self):
        """Test argmax for class selection"""
        logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        predictions = torch.argmax(logits, dim=1)
        
        assert predictions[0].item() == 1  # Real
        assert predictions[1].item() == 0  # Fake


# ============================================================
# TEST: INTEGRATION TESTS
# ============================================================

class TestIntegration:
    """End-to-end integration tests"""
    
    def test_video_to_prediction_pipeline(self):
        """Test complete pipeline from video to prediction"""
        pytest.skip("Requires actual video and model files")
    
    def test_error_handling_missing_frames(self, temp_dir):
        """Test error handling for missing frames directory"""
        invalid_dir = os.path.join(temp_dir, "nonexistent")
        
        # This should raise an error
        with pytest.raises((ValueError, FileNotFoundError)):
            predictor = MultimodalPredictor()
            predictor.predict(invalid_dir)


# ============================================================
# TEST: PERFORMANCE TESTS
# ============================================================

class TestPerformance:
    """Performance and benchmark tests"""
    
    def test_inference_speed(self, fusion_model, dummy_device):
        """Test inference speed"""
        import time
        
        video_features = torch.randn(1, 512).to(dummy_device)
        audio_features = torch.randn(1, 256).to(dummy_device)
        
        # Warmup
        fusion_model(video_features, audio_features)
        
        # Benchmark
        start = time.time()
        for _ in range(100):
            _ = fusion_model(video_features, audio_features)
        elapsed = time.time() - start
        
        avg_time = elapsed / 100
        print(f"Average inference time: {avg_time*1000:.2f}ms")
        
        # Should complete in reasonable time
        assert avg_time < 0.01, f"Inference too slow: {avg_time}s"
    
    def test_memory_efficiency(self, dummy_device):
        """Test memory usage"""
        model = FusionClassifier()
        model = model.to(dummy_device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        
        # Model should be relatively small
        assert total_params < 1_000_000, "Model too large"


# ============================================================
# PYTEST CONFIGURATION
# ============================================================

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
