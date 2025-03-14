import torch
import unittest
from PIL import Image
import numpy as np
from attacker_network import AttackerNetwork

class TestAttackerNetwork(unittest.TestCase):
    def setUp(self):
        # Skip tests if CUDA is not available and we're not forcing CPU execution
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping GPU tests")
        
        # Create a dummy image for testing
        self.dummy_image = Image.new('RGB', (256, 256), color='white')
        
        # Set shorter inference steps for faster testing
        self.num_inference_steps = 2
    
    def test_initialization(self):
        """Test that the AttackerNetwork initializes correctly."""
        try:
            attacker = AttackerNetwork(
                positive_template="Test prompt: ",
                negative_template="Test negative prompt",
                positive_ctx_len=5,
                negative_ctx_len=3,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            self.assertIsNotNone(attacker)
            print("AttackerNetwork initialized successfully")
        except Exception as e:
            self.fail(f"AttackerNetwork initialization failed with error: {e}")
    
    def test_forward_pass(self):
        """Test the forward pass of the AttackerNetwork."""
        attacker = AttackerNetwork(
            positive_template="Test prompt: ",
            negative_template="Test negative prompt",
            positive_ctx_len=5,
            negative_ctx_len=3,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        try:
            # Run forward pass
            modified_features, original_features, modified_image = attacker(
                self.dummy_image,
                num_inference_steps=self.num_inference_steps
            )
            
            # Check outputs
            self.assertIsNotNone(modified_features)
            self.assertIsNotNone(original_features)
            self.assertIsNotNone(modified_image)
            
            # Check that features have the expected shape
            self.assertEqual(len(modified_features), 1)  # Should be a tuple with one element
            self.assertEqual(len(original_features), 1)  # Should be a tuple with one element
            
            # Check that the modified image is a PIL Image
            self.assertIsInstance(modified_image, Image.Image)
            
            print("Forward pass successful")
            print(f"Modified features shape: {modified_features[0].shape}")
            print(f"Original features shape: {original_features[0].shape}")
            print(f"Modified image size: {modified_image.size}")
            
        except Exception as e:
            self.fail(f"Forward pass failed with error: {e}")
    
    def test_trainable_parameters(self):
        """Test that we can get the trainable parameters."""
        attacker = AttackerNetwork(
            positive_template="Test prompt: ",
            negative_template="Test negative prompt",
            positive_ctx_len=5,
            negative_ctx_len=3,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        try:
            trainable_params = attacker.get_trainable_parameters()
            self.assertIsNotNone(trainable_params)
            self.assertGreater(len(trainable_params), 0)
            
            # Check that all parameters require gradients
            for param in trainable_params:
                self.assertTrue(param.requires_grad)
            
            print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
            
        except Exception as e:
            self.fail(f"Getting trainable parameters failed with error: {e}")

if __name__ == "__main__":
    unittest.main()
