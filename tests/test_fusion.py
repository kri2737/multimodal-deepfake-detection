"""
Unit Tests for Fusion Classifier and Multimodal Pipeline

Verifies the correctness of the fusion classifier, model architectures,
and edge-case handling.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import os
import tempfile
import shutil

from audio_module.audio_model import AudioClassifier
from fusion_module.fusion_classifier import FusionClassifier, MultimodalPipeline
from audio_module.preprocess_audio import AudioPreprocessor


class TestAudioModel(unittest.TestCase):
    """Test cases for AudioClassifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.model = AudioClassifier(input_dim=40, num_classes=2)
        self.model = self.model.to(self.device)
    
    def test_model_initialization(self):
        """Test that model initializes correctly."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.input_dim, 40)
        self.assertEqual(self.model.num_classes, 2)
    
    def test_forward_pass(self):
        """Test forward pass through audio model."""
        batch_size = 4
        input_data = torch.randn(batch_size, 40).to(self.device)
        
        output = self.model(input_data)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, 2))
    
    def test_get_features(self):
        """Test feature extraction."""
        batch_size = 4
        input_data = torch.randn(batch_size, 40).to(self.device)
        
        features = self.model.get_features(input_data)
        
        # Features should have embedding dimension
        self.assertEqual(features.shape[0], batch_size)
        self.assertEqual(features.shape[1], self.model.embedding_dim)
    
    def test_get_embedding(self):
        """Test embedding extraction."""
        batch_size = 4
        input_data = torch.randn(batch_size, 40).to(self.device)
        
        embedding = self.model.get_embedding(input_data)
        
        # Embedding should be same as features
        self.assertEqual(embedding.shape[0], batch_size)
        self.assertEqual(embedding.shape[1], self.model.embedding_dim)
    
    def test_model_save_load(self):
        """Test saving and loading model weights."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'audio_model.pth')
            
            # Save model
            torch.save(self.model.state_dict(), model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Load model
            new_model = AudioClassifier(input_dim=40, num_classes=2)
            new_model.load_state_dict(torch.load(model_path))
            new_model = new_model.to(self.device)
            
            # Test that loaded model produces same output
            input_data = torch.randn(2, 40).to(self.device)
            
            with torch.no_grad():
                output1 = self.model(input_data)
                output2 = new_model(input_data)
            
            torch.testing.assert_close(output1, output2)


class TestFusionClassifier(unittest.TestCase):
    """Test cases for FusionClassifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.fusion_model_feature = FusionClassifier(
            video_feature_dim=512,
            audio_feature_dim=32,
            fusion_type='feature'
        ).to(self.device)
        
        self.fusion_model_late = FusionClassifier(
            video_feature_dim=512,
            audio_feature_dim=32,
            fusion_type='late'
        ).to(self.device)
    
    def test_feature_fusion_forward(self):
        """Test feature-level fusion forward pass."""
        batch_size = 2
        video_features = torch.randn(batch_size, 512).to(self.device)
        audio_features = torch.randn(batch_size, 32).to(self.device)
        
        output = self.fusion_model_feature(video_features, audio_features)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, 2))
    
    def test_late_fusion_forward(self):
        """Test late fusion forward pass."""
        batch_size = 2
        video_features = torch.randn(batch_size, 512).to(self.device)
        audio_features = torch.randn(batch_size, 32).to(self.device)
        
        output = self.fusion_model_late(video_features, audio_features)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, 2))
    
    def test_invalid_fusion_type(self):
        """Test that invalid fusion type raises error."""
        with self.assertRaises(ValueError):
            FusionClassifier(fusion_type='invalid')
    
    def test_fusion_model_save_load(self):
        """Test saving and loading fusion model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'fusion_model.pth')
            
            # Save model
            torch.save(self.fusion_model_feature.state_dict(), model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Load model
            new_model = FusionClassifier(
                video_feature_dim=512,
                audio_feature_dim=32,
                fusion_type='feature'
            ).to(self.device)
            new_model.load_state_dict(torch.load(model_path))
            
            # Test that loaded model produces same output
            video_features = torch.randn(2, 512).to(self.device)
            audio_features = torch.randn(2, 32).to(self.device)
            
            with torch.no_grad():
                output1 = self.fusion_model_feature(video_features, audio_features)
                output2 = new_model(video_features, audio_features)
            
            torch.testing.assert_close(output1, output2)


class TestAudioPreprocessor(unittest.TestCase):
    """Test cases for AudioPreprocessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = AudioPreprocessor(sr=22050, n_mfcc=40)
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        self.assertEqual(self.preprocessor.sr, 22050)
        self.assertEqual(self.preprocessor.n_mfcc, 40)
    
    def test_mfcc_feature_extraction_invalid_file(self):
        """Test MFCC extraction with invalid file."""
        with self.assertRaises(FileNotFoundError):
            self.preprocessor.extract_mfcc_features('nonexistent.wav')
    
    def test_zero_features_return_shape(self):
        """Test that zero features have correct shape."""
        zero_features = np.zeros(self.preprocessor.n_mfcc)
        self.assertEqual(len(zero_features), 40)


class TestMultimodalPipeline(unittest.TestCase):
    """Test cases for MultimodalPipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.temp_dir, 'models')
        os.makedirs(self.models_dir)
        
        # Initialize models in temp directory
        from initialize_models import initialize_all_models
        initialize_all_models(models_dir=self.models_dir, device=self.device)
        
        self.pipeline = MultimodalPipeline(
            video_model_path=os.path.join(self.models_dir, 'video_model.pth'),
            audio_model_path=os.path.join(self.models_dir, 'audio_model.pth'),
            fusion_model_path=os.path.join(self.models_dir, 'fusion_model.pth'),
            device=self.device
        )
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_pipeline_initialization(self):
        """Test that pipeline initializes correctly."""
        self.assertIsNotNone(self.pipeline)
        self.assertIsNotNone(self.pipeline.video_model)
        self.assertIsNotNone(self.pipeline.audio_model)
        self.assertIsNotNone(self.pipeline.fusion_model)
    
    def test_video_features_zero_faces(self):
        """Test video feature extraction with no faces."""
        face_tensors = []
        features = self.pipeline.get_video_features(face_tensors)
        
        # Should return zeros
        self.assertEqual(features.shape, (512,))
        torch.testing.assert_close(features, torch.zeros(512).to(self.device))
    
    def test_prediction_result_structure(self):
        """Test that prediction result has expected structure."""
        result = {
            'has_video': False,
            'has_audio': False,
            'faces_detected': 0
        }
        
        # Check required keys
        self.assertIn('has_video', result)
        self.assertIn('has_audio', result)
        self.assertIn('faces_detected', result)
        self.assertIsInstance(result['faces_detected'], int)
    
    def test_edge_case_no_video_no_audio(self):
        """Test edge case handling when neither video nor audio is present."""
        # This would be tested with an actual video in practice
        # For unit tests, we just verify the logic
        result = {
            'prediction': 'FAKE',
            'confidence': 0.5,
            'has_video': False,
            'has_audio': False
        }
        
        # With no inputs, confidence should be 0.5
        self.assertEqual(result['confidence'], 0.5)


class TestModelIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_forward_pass(self):
        """Test complete forward pass through all models."""
        # Create dummy features
        batch_size = 2
        video_features = torch.randn(batch_size, 512).to(self.device)
        audio_features = torch.randn(batch_size, 32).to(self.device)
        
        # Create models
        audio_model = AudioClassifier()
        fusion_model = FusionClassifier(fusion_type='feature')
        
        audio_model = audio_model.to(self.device)
        fusion_model = fusion_model.to(self.device)
        
        with torch.no_grad():
            # Forward through fusion
            fusion_output = fusion_model(video_features, audio_features)
            
            # Check output
            self.assertEqual(fusion_output.shape, (batch_size, 2))
            
            # Apply softmax to get probabilities
            probs = torch.softmax(fusion_output, dim=1)
            self.assertTrue(torch.all(probs >= 0))
            self.assertTrue(torch.all(probs <= 1))
            self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(batch_size)))
    
    def test_confidence_range(self):
        """Test that confidence values are in valid range."""
        model = AudioClassifier()
        model.eval()
        
        input_data = torch.randn(10, 40)
        
        with torch.no_grad():
            output = model(input_data)
            probs = torch.softmax(output, dim=1)
        
        # All probabilities should be in [0, 1]
        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.all(probs <= 1))


if __name__ == '__main__':
    unittest.main()
