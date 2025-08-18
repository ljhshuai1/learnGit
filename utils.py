"""
Utility functions for hyperspectral image classification with hybrid attention mechanism.

This module provides essential functions for window partitioning, padding strategies,
and other utilities needed for the hybrid attention network.
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple, Optional


def get_optimal_window_size(patch_size: int) -> int:
    """
    Determine optimal window size for sparse window attention based on patch size.
    
    Args:
        patch_size (int): Size of the input patch (P in P×P)
        
    Returns:
        int: Optimal window size
    """
    # Use smaller windows for smaller patches to maintain reasonable attention locality
    if patch_size <= 7:
        return min(4, patch_size)
    elif patch_size <= 11:
        return min(6, patch_size)
    else:
        return min(8, patch_size)


def reflection_pad_to_window_size(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Apply reflection padding to ensure patch dimensions are divisible by window size.
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W)
        window_size (int): Target window size
        
    Returns:
        Tuple[torch.Tensor, Tuple[int, int]]: Padded tensor and original dimensions
    """
    B, C, H, W = x.shape
    
    # Calculate padding needed
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    # Apply reflection padding
    if pad_h > 0 or pad_w > 0:
        # PyTorch padding format: (left, right, top, bottom)
        padding = (0, pad_w, 0, pad_h)
        x_padded = F.pad(x, padding, mode='reflect')
    else:
        x_padded = x
        
    return x_padded, (H, W)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition input tensor into non-overlapping windows.
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W)
        window_size (int): Size of each window
        
    Returns:
        torch.Tensor: Windowed tensor of shape (B * num_windows, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    
    # Ensure dimensions are divisible by window_size
    assert H % window_size == 0 and W % window_size == 0, \
        f"Input dimensions ({H}, {W}) must be divisible by window_size ({window_size})"
    
    # Reshape to create windows
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    windows = windows.view(-1, C, window_size, window_size)
    
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int, B: int) -> torch.Tensor:
    """
    Reverse window partitioning to restore original tensor shape.
    
    Args:
        windows (torch.Tensor): Windowed tensor of shape (B * num_windows, C, window_size, window_size)
        window_size (int): Size of each window
        H (int): Original height
        W (int): Original width
        B (int): Batch size
        
    Returns:
        torch.Tensor: Restored tensor of shape (B, C, H, W)
    """
    C = windows.shape[1]
    
    # Calculate number of windows
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    
    # Reshape windows back to original format
    x = windows.view(B, num_windows_h, num_windows_w, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(B, C, H, W)
    
    return x


def remove_padding(x: torch.Tensor, original_size: Tuple[int, int]) -> torch.Tensor:
    """
    Remove padding to restore original dimensions.
    
    Args:
        x (torch.Tensor): Padded tensor of shape (B, C, H_padded, W_padded)
        original_size (Tuple[int, int]): Original (H, W) dimensions
        
    Returns:
        torch.Tensor: Tensor with original dimensions
    """
    H_orig, W_orig = original_size
    return x[:, :, :H_orig, :W_orig]


def generate_offset_grid(height: int, width: int, kernel_size: int = 3, device: str = 'cuda') -> torch.Tensor:
    """
    Generate offset grid for deformable attention.
    
    Args:
        height (int): Height of the feature map
        width (int): Width of the feature map
        kernel_size (int): Size of the deformable kernel
        device (str): Device to place the tensor on
        
    Returns:
        torch.Tensor: Offset grid of shape (1, 2*kernel_size^2, height, width)
    """
    # Create base offset grid
    offset_y, offset_x = torch.meshgrid(
        torch.arange(-(kernel_size//2), kernel_size//2 + 1, device=device),
        torch.arange(-(kernel_size//2), kernel_size//2 + 1, device=device),
        indexing='ij'
    )
    
    # Stack and reshape
    base_offset = torch.stack([offset_y, offset_x], dim=0).float()  # (2, kernel_size, kernel_size)
    base_offset = base_offset.view(2 * kernel_size * kernel_size, 1, 1)  # (2*K^2, 1, 1)
    
    # Expand to feature map size
    offset_grid = base_offset.expand(-1, height, width)  # (2*K^2, H, W)
    offset_grid = offset_grid.unsqueeze(0)  # (1, 2*K^2, H, W)
    
    return offset_grid


def create_attention_mask(patch_size: int, window_size: int, device: str = 'cuda') -> Optional[torch.Tensor]:
    """
    Create attention mask for sparse window attention if needed.
    
    Args:
        patch_size (int): Size of the input patch
        window_size (int): Size of attention windows
        device (str): Device to place the mask on
        
    Returns:
        Optional[torch.Tensor]: Attention mask or None if not needed
    """
    # For now, return None as we'll implement basic window attention
    # This can be extended for more complex masking strategies
    return None


def validate_patch_size(patch_size: int) -> bool:
    """
    Validate that patch size is supported.
    
    Args:
        patch_size (int): Patch size to validate
        
    Returns:
        bool: True if patch size is supported
    """
    supported_sizes = [5, 7, 9, 11, 13, 15]
    return patch_size in supported_sizes


def get_patch_info(patch_size: int) -> dict:
    """
    Get information about patch configuration.
    
    Args:
        patch_size (int): Size of the patch
        
    Returns:
        dict: Dictionary containing patch configuration info
    """
    if not validate_patch_size(patch_size):
        raise ValueError(f"Unsupported patch size: {patch_size}. Supported sizes: [5, 7, 9, 11, 13, 15]")
    
    window_size = get_optimal_window_size(patch_size)
    
    return {
        'patch_size': patch_size,
        'window_size': window_size,
        'num_windows': (patch_size // window_size) ** 2,
        'needs_padding': patch_size % window_size != 0,
        'padding_size': (window_size - patch_size % window_size) % window_size
    }