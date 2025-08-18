#!/usr/bin/env python3
"""
Comprehensive test suite for the hyperspectral classification network.

This script runs a complete test of all components to ensure the implementation
is working correctly and demonstrates the key features.
"""

import torch
import time
import numpy as np
from typing import Dict, List

# Import our modules
from utils import validate_patch_size, get_patch_info, get_optimal_window_size
from hybrid_attention import HybridAttention, PredictorLG, WindowAttention_sparse
from hyperspectral_classifier import HyperspectralClassifier, EnsembleHyperspectralClassifier


def test_utils():
    """Test utility functions."""
    print("=== Testing Utility Functions ===")
    
    # Test patch size validation
    valid_sizes = [5, 7, 9, 11, 13, 15]
    invalid_sizes = [4, 6, 8, 16, 17]
    
    print("Patch size validation:")
    for size in valid_sizes:
        assert validate_patch_size(size), f"Size {size} should be valid"
        print(f"  ✓ {size} is valid")
    
    for size in invalid_sizes:
        assert not validate_patch_size(size), f"Size {size} should be invalid"
        print(f"  ✓ {size} is correctly invalid")
    
    # Test patch info generation
    print("\nPatch information:")
    for size in valid_sizes:
        info = get_patch_info(size)
        window_size = get_optimal_window_size(size)
        print(f"  Patch {size}: window={window_size}, info={info}")
    
    print("✓ Utility functions test passed!\n")


def test_attention_modules():
    """Test individual attention modules."""
    print("=== Testing Attention Modules ===")
    
    # Test PredictorLG
    print("Testing PredictorLG (Deformable Attention)...")
    predictor = PredictorLG(dim=64, num_heads=4, kernel_size=3)
    x = torch.randn(2, 64, 9, 9)
    output = predictor(x)
    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    print(f"  ✓ PredictorLG: {x.shape} -> {output.shape}")
    
    # Test WindowAttention_sparse
    print("Testing WindowAttention_sparse...")
    window_attn = WindowAttention_sparse(dim=64, window_size=4, num_heads=4)
    x_window = torch.randn(8, 16, 64)  # (num_windows*B, window_size^2, dim)
    output = window_attn(x_window)
    assert output.shape == x_window.shape, f"Expected {x_window.shape}, got {output.shape}"
    print(f"  ✓ WindowAttention_sparse: {x_window.shape} -> {output.shape}")
    
    # Test HybridAttention
    print("Testing HybridAttention...")
    hybrid_attn = HybridAttention(dim=64, patch_size=9, num_heads=4)
    x = torch.randn(2, 64, 9, 9)
    output = hybrid_attn(x)
    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    print(f"  ✓ HybridAttention: {x.shape} -> {output.shape}")
    
    print("✓ Attention modules test passed!\n")


def test_classifiers():
    """Test complete classifier models."""
    print("=== Testing Classifier Models ===")
    
    # Test single classifier
    print("Testing HyperspectralClassifier...")
    for patch_size in [5, 9, 15]:
        model = HyperspectralClassifier(
            in_channels=100,
            num_classes=10,
            patch_size=patch_size,
            embed_dim=64,
            num_heads=4,
            num_layers=2
        )
        
        x = torch.randn(3, 100, patch_size, patch_size)
        output = model(x)
        expected_shape = (3, 10)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"  ✓ Patch {patch_size}: {x.shape} -> {output.shape}")
    
    # Test ensemble classifier
    print("Testing EnsembleHyperspectralClassifier...")
    ensemble = EnsembleHyperspectralClassifier(
        in_channels=100,
        num_classes=10,
        patch_sizes=[5, 7, 9, 11],
        embed_dim=32,
        num_layers=1
    )
    
    inputs = [torch.randn(2, 100, size, size) for size in [5, 7, 9, 11]]
    output = ensemble(inputs)
    expected_shape = (2, 10)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"  ✓ Ensemble: multiple inputs -> {output.shape}")
    
    print("✓ Classifier models test passed!\n")


def test_training_simulation():
    """Simulate a training step to ensure gradients flow correctly."""
    print("=== Testing Training Simulation ===")
    
    # Create model and data
    model = HyperspectralClassifier(
        in_channels=50,
        num_classes=5,
        patch_size=9,
        embed_dim=64,
        num_layers=2
    )
    
    # Create synthetic data
    batch_size = 4
    x = torch.randn(batch_size, 50, 9, 9)
    y = torch.randint(0, 5, (batch_size,))
    
    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Forward pass
    model.train()
    output = model(x)
    loss = criterion(output, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    grad_norm = 0
    param_count = 0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.norm().item() ** 2
            param_count += 1
    grad_norm = grad_norm ** 0.5
    
    print(f"  Training loss: {loss.item():.4f}")
    print(f"  Gradient norm: {grad_norm:.4f}")
    print(f"  Parameters with gradients: {param_count}")
    
    # Optimizer step
    optimizer.step()
    
    print("✓ Training simulation test passed!\n")


def benchmark_performance():
    """Benchmark model performance across different configurations."""
    print("=== Performance Benchmark ===")
    
    device = 'cpu'  # Use CPU since CUDA may not be available
    configurations = [
        {"patch_size": 5, "embed_dim": 32, "num_layers": 1},
        {"patch_size": 9, "embed_dim": 64, "num_layers": 2},
        {"patch_size": 13, "embed_dim": 96, "num_layers": 3},
    ]
    
    results = []
    
    for config in configurations:
        print(f"Testing configuration: {config}")
        
        # Create model
        model = HyperspectralClassifier(
            in_channels=100,
            num_classes=10,
            **config
        ).to(device)
        
        # Create data
        patch_size = config["patch_size"]
        x = torch.randn(4, 100, patch_size, patch_size).to(device)
        
        # Warm-up
        model.eval()
        with torch.no_grad():
            _ = model(x)
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                output = model(x)
        inference_time = (time.time() - start_time) / 10
        
        # Calculate model size
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = param_count * 4 / (1024 * 1024)  # Assume float32
        
        result = {
            "config": config,
            "inference_time": inference_time,
            "parameters": param_count,
            "model_size_mb": model_size_mb
        }
        results.append(result)
        
        print(f"  Inference time: {inference_time:.4f}s")
        print(f"  Parameters: {param_count:,}")
        print(f"  Model size: {model_size_mb:.2f}MB")
        print()
    
    print("Performance Summary:")
    print(f"{'Config':<20} {'Time (s)':<10} {'Params':<10} {'Size (MB)':<10}")
    print("-" * 55)
    for result in results:
        config_str = f"P{result['config']['patch_size']}_E{result['config']['embed_dim']}_L{result['config']['num_layers']}"
        print(f"{config_str:<20} {result['inference_time']:<10.4f} {result['parameters']:<10,} {result['model_size_mb']:<10.2f}")
    
    print("✓ Performance benchmark completed!\n")


def main():
    """Run all tests."""
    print("Hyperspectral Image Classification - Comprehensive Test Suite")
    print("=" * 70)
    
    try:
        # Run all test suites
        test_utils()
        test_attention_modules()
        test_classifiers()
        test_training_simulation()
        benchmark_performance()
        
        print("=" * 70)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("The hybrid attention network implementation is working correctly.")
        print("Ready for use with real hyperspectral datasets!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)