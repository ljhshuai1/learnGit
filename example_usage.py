"""
Example usage of the hyperspectral image classification network with hybrid attention.

This script demonstrates how to use the HyperspectralClassifier for different
patch sizes and configurations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import time

from hyperspectral_classifier import HyperspectralClassifier, EnsembleHyperspectralClassifier
from utils import get_patch_info, validate_patch_size


def create_synthetic_data(batch_size: int = 8, 
                         num_bands: int = 200, 
                         patch_size: int = 9, 
                         num_classes: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic hyperspectral data for testing.
    
    Args:
        batch_size (int): Number of samples in batch
        num_bands (int): Number of spectral bands
        patch_size (int): Size of spatial patches
        num_classes (int): Number of classes
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Data and labels
    """
    # Generate synthetic hyperspectral data
    data = torch.randn(batch_size, num_bands, patch_size, patch_size)
    
    # Add some spectral structure (simulate real hyperspectral characteristics)
    for i in range(num_bands):
        # Add smooth spectral response
        spectral_weight = np.exp(-((i - num_bands//2) ** 2) / (2 * (num_bands/6) ** 2))
        data[:, i, :, :] *= spectral_weight
        
        # Add spatial correlation
        if i > 0:
            data[:, i, :, :] += 0.3 * data[:, i-1, :, :]
    
    # Generate random labels
    labels = torch.randint(0, num_classes, (batch_size,))
    
    return data, labels


def test_single_model(patch_size: int = 9, 
                     num_bands: int = 200, 
                     num_classes: int = 10,
                     device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Test a single model with specified patch size.
    
    Args:
        patch_size (int): Size of input patches
        num_bands (int): Number of spectral bands
        num_classes (int): Number of classes
        device (str): Device to run on
    """
    print(f"\n=== Testing Single Model (Patch Size: {patch_size}) ===")
    
    # Validate patch size
    if not validate_patch_size(patch_size):
        print(f"Unsupported patch size: {patch_size}")
        return
    
    # Get patch information
    patch_info = get_patch_info(patch_size)
    print(f"Patch Info: {patch_info}")
    
    # Create model
    model = HyperspectralClassifier(
        in_channels=num_bands,
        num_classes=num_classes,
        patch_size=patch_size,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        mlp_ratio=4.0,
        dropout=0.1,
        use_spectral_attention=True,
        use_multi_scale=False
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create synthetic data
    batch_size = 4
    data, labels = create_synthetic_data(batch_size, num_bands, patch_size, num_classes)
    data, labels = data.to(device), labels.to(device)
    
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = model(data)
        inference_time = time.time() - start_time
        
    print(f"Output shape: {outputs.shape}")
    print(f"Inference time: {inference_time:.4f} seconds")
    
    # Test attention maps
    attention_maps = model.get_attention_maps(data)
    print(f"Number of attention maps: {len(attention_maps)}")
    for i, attn_map in enumerate(attention_maps):
        print(f"Attention map {i} shape: {attn_map.shape}")
    
    # Test training step
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    print(f"Training loss: {loss.item():.4f}")
    
    # Predictions
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == labels).float().mean()
    
    print(f"Random accuracy: {accuracy.item():.4f}")
    
    return model


def test_ensemble_model(patch_sizes: List[int] = [5, 7, 9, 11],
                       num_bands: int = 200,
                       num_classes: int = 10,
                       device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Test ensemble model with multiple patch sizes.
    
    Args:
        patch_sizes (List[int]): List of patch sizes
        num_bands (int): Number of spectral bands
        num_classes (int): Number of classes
        device (str): Device to run on
    """
    print(f"\n=== Testing Ensemble Model (Patch Sizes: {patch_sizes}) ===")
    
    # Create ensemble model
    ensemble_model = EnsembleHyperspectralClassifier(
        in_channels=num_bands,
        num_classes=num_classes,
        patch_sizes=patch_sizes,
        embed_dim=128,  # Smaller for ensemble
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.1
    ).to(device)
    
    print(f"Ensemble model created with {sum(p.numel() for p in ensemble_model.parameters())} parameters")
    
    # Create synthetic data for each patch size
    batch_size = 4
    data_list = []
    labels = None
    
    for patch_size in patch_sizes:
        data, label = create_synthetic_data(batch_size, num_bands, patch_size, num_classes)
        data_list.append(data.to(device))
        if labels is None:
            labels = label.to(device)
    
    print(f"Created data for patch sizes: {[data.shape for data in data_list]}")
    
    # Test forward pass
    ensemble_model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = ensemble_model(data_list)
        inference_time = time.time() - start_time
        
    print(f"Ensemble output shape: {outputs.shape}")
    print(f"Ensemble inference time: {inference_time:.4f} seconds")
    
    # Test training step
    ensemble_model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ensemble_model.parameters(), lr=0.001)
    
    optimizer.zero_grad()
    outputs = ensemble_model(data_list)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    print(f"Ensemble training loss: {loss.item():.4f}")
    
    return ensemble_model


def benchmark_patch_sizes(patch_sizes: List[int] = [5, 7, 9, 11, 13, 15],
                         num_bands: int = 200,
                         num_classes: int = 10,
                         device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Benchmark different patch sizes.
    
    Args:
        patch_sizes (List[int]): List of patch sizes to benchmark
        num_bands (int): Number of spectral bands
        num_classes (int): Number of classes
        device (str): Device to run on
    """
    print(f"\n=== Benchmarking Patch Sizes: {patch_sizes} ===")
    
    results = {}
    
    for patch_size in patch_sizes:
        if not validate_patch_size(patch_size):
            print(f"Skipping unsupported patch size: {patch_size}")
            continue
            
        print(f"\nTesting patch size {patch_size}...")
        
        try:
            # Create model
            model = HyperspectralClassifier(
                in_channels=num_bands,
                num_classes=num_classes,
                patch_size=patch_size,
                embed_dim=128,  # Smaller for benchmarking
                num_heads=4,
                num_layers=2,
                mlp_ratio=2.0,
                dropout=0.1
            ).to(device)
            
            # Create data
            batch_size = 4
            data, labels = create_synthetic_data(batch_size, num_bands, patch_size, num_classes)
            data, labels = data.to(device), labels.to(device)
            
            # Benchmark forward pass
            model.eval()
            torch.cuda.synchronize() if device == 'cuda' else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(10):  # Multiple runs for better timing
                    outputs = model(data)
            
            torch.cuda.synchronize() if device == 'cuda' else None
            avg_time = (time.time() - start_time) / 10
            
            # Calculate model size
            param_count = sum(p.numel() for p in model.parameters())
            
            # Calculate memory usage (approximate)
            if device == 'cuda':
                memory_usage = torch.cuda.max_memory_allocated() / 1024**2  # MB
                torch.cuda.reset_peak_memory_stats()
            else:
                memory_usage = 0
            
            results[patch_size] = {
                'inference_time': avg_time,
                'parameters': param_count,
                'memory_mb': memory_usage,
                'output_shape': outputs.shape
            }
            
            print(f"  Inference time: {avg_time:.4f}s")
            print(f"  Parameters: {param_count:,}")
            print(f"  Memory: {memory_usage:.1f}MB")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[patch_size] = {'error': str(e)}
    
    # Print summary
    print(f"\n=== Benchmark Summary ===")
    print(f"{'Patch Size':<10} {'Time (s)':<10} {'Parameters':<12} {'Memory (MB)':<12}")
    print("-" * 50)
    
    for patch_size, result in results.items():
        if 'error' not in result:
            print(f"{patch_size:<10} {result['inference_time']:<10.4f} "
                  f"{result['parameters']:<12,} {result['memory_mb']:<12.1f}")
        else:
            print(f"{patch_size:<10} ERROR: {result['error']}")
    
    return results


def visualize_attention_maps(model, data, save_path: str = None):
    """
    Visualize attention maps from the model.
    
    Args:
        model: Trained model
        data: Input data tensor
        save_path (str): Path to save the visualization
    """
    model.eval()
    with torch.no_grad():
        attention_maps = model.get_attention_maps(data[:1])  # Use first sample
    
    # Create visualization
    num_maps = len(attention_maps)
    fig, axes = plt.subplots(1, num_maps + 1, figsize=(4 * (num_maps + 1), 4))
    
    # Original data (mean across spectral bands)
    original = data[0].mean(dim=0).cpu().numpy()
    axes[0].imshow(original, cmap='viridis')
    axes[0].set_title('Original\n(Mean Spectrum)')
    axes[0].axis('off')
    
    # Attention maps
    for i, attn_map in enumerate(attention_maps):
        attn = attn_map[0, 0].cpu().numpy()  # First sample, first channel
        axes[i + 1].imshow(attn, cmap='hot')
        axes[i + 1].set_title(f'Attention Map\nLayer {i + 1}')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention maps saved to {save_path}")
    
    plt.show()


def main():
    """Main function to run all examples."""
    print("Hyperspectral Image Classification with Hybrid Attention - Example Usage")
    print("=" * 80)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test single models
    for patch_size in [5, 7, 9, 11]:
        try:
            model = test_single_model(patch_size=patch_size, device=device)
            
            # Clean up memory
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error testing patch size {patch_size}: {e}")
    
    # Test ensemble model
    try:
        ensemble_model = test_ensemble_model(device=device)
        del ensemble_model
        if device == 'cuda':
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error testing ensemble model: {e}")
    
    # Benchmark all patch sizes
    try:
        benchmark_results = benchmark_patch_sizes(device=device)
    except Exception as e:
        print(f"Error benchmarking: {e}")
    
    print("\n" + "=" * 80)
    print("Example usage completed!")


if __name__ == "__main__":
    main()