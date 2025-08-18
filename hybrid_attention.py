"""
Hybrid attention mechanism combining CAMixer's PredictorLG deformable attention 
and AST's sparse window attention for hyperspectral image classification.

This module implements the core attention mechanisms and their fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from utils import (
    window_partition, window_reverse, reflection_pad_to_window_size,
    remove_padding, generate_offset_grid, get_optimal_window_size
)


class PredictorLG(nn.Module):
    """
    CAMixer-style deformable attention predictor that learns offset predictions.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, kernel_size: int = 3, 
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Offset prediction network
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim * 2, 2 * kernel_size * kernel_size, kernel_size=1),
        )
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # MLP for feature enhancement
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def deformable_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply deformable attention mechanism.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output tensor with deformable attention applied
        """
        B, C, H, W = x.shape
        
        # Predict offsets
        offsets = self.offset_predictor(x)  # (B, 2*K^2, H, W)
        
        # Generate base grid
        base_grid = generate_offset_grid(H, W, self.kernel_size, x.device)
        
        # Add predicted offsets to base grid
        sampling_grid = base_grid + offsets  # (B, 2*K^2, H, W)
        
        # Reshape for grid sampling
        sampling_grid = sampling_grid.view(B, 2, self.kernel_size**2, H, W)
        sampling_grid = sampling_grid.permute(0, 2, 3, 4, 1)  # (B, K^2, H, W, 2)
        
        # Normalize grid coordinates to [-1, 1]
        sampling_grid[..., 0] = 2.0 * sampling_grid[..., 0] / (W - 1) - 1.0
        sampling_grid[..., 1] = 2.0 * sampling_grid[..., 1] / (H - 1) - 1.0
        
        # Sample features using deformed positions
        x_expanded = x.unsqueeze(1).expand(-1, self.kernel_size**2, -1, -1, -1)
        x_expanded = x_expanded.reshape(B * self.kernel_size**2, C, H, W)
        
        sampling_grid = sampling_grid.reshape(B * self.kernel_size**2, H, W, 2)
        sampled_features = F.grid_sample(
            x_expanded, sampling_grid, mode='bilinear', 
            padding_mode='border', align_corners=True
        )
        
        sampled_features = sampled_features.view(B, self.kernel_size**2, C, H, W)
        
        return sampled_features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PredictorLG module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output tensor with enhanced features
        """
        B, C, H, W = x.shape
        
        # Apply deformable attention
        deformed_features = self.deformable_attention(x)  # (B, K^2, C, H, W)
        
        # Aggregate deformed features (mean pooling across kernel positions)
        aggregated = deformed_features.mean(dim=1)  # (B, C, H, W)
        
        # Reshape for attention computation
        x_flat = aggregated.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Apply layer norm and attention
        x_norm = self.norm1(x_flat)
        
        # Compute QKV
        qkv = self.qkv(x_norm).reshape(B, H*W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x_attn = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        x_attn = self.proj(x_attn)
        x_attn = self.dropout(x_attn)
        
        # Residual connection
        x_out = x_flat + x_attn
        
        # MLP
        x_out = x_out + self.mlp(self.norm2(x_out))
        
        # Reshape back to spatial format
        x_out = x_out.transpose(1, 2).reshape(B, C, H, W)
        
        return x_out


class WindowAttention_sparse(nn.Module):
    """
    AST-style sparse window attention mechanism.
    """
    
    def __init__(self, dim: int, window_size: int, num_heads: int = 8, 
                 qkv_bias: bool = True, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of sparse window attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (num_windows*B, window_size*window_size, C)
            mask (Optional[torch.Tensor]): Attention mask
            
        Returns:
            torch.Tensor: Output tensor with window attention applied
        """
        B_, N, C = x.shape
        
        # Generate QKV
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # Apply mask if provided
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class HybridAttention(nn.Module):
    """
    Hybrid attention mechanism combining PredictorLG and WindowAttention_sparse.
    """
    
    def __init__(self, dim: int, patch_size: int, num_heads: int = 8, 
                 mlp_ratio: float = 4.0, dropout: float = 0.1, 
                 deformable_kernel_size: int = 3):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.window_size = get_optimal_window_size(patch_size)
        self.num_heads = num_heads
        
        # PredictorLG deformable attention
        self.deformable_attention = PredictorLG(
            dim=dim, 
            num_heads=num_heads, 
            kernel_size=deformable_kernel_size,
            mlp_ratio=mlp_ratio, 
            dropout=dropout
        )
        
        # Sparse window attention
        self.window_attention = WindowAttention_sparse(
            dim=dim, 
            window_size=self.window_size, 
            num_heads=num_heads, 
            dropout=dropout
        )
        
        # Feature fusion layers
        self.fusion_norm = nn.LayerNorm(dim * 2)  # Input has 2*dim features
        self.fusion_proj = nn.Linear(dim * 2, dim)
        self.fusion_dropout = nn.Dropout(dropout)
        
        # Final MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.final_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of hybrid attention mechanism.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, P, P)
            
        Returns:
            torch.Tensor: Output tensor with hybrid attention applied
        """
        B, C, H, W = x.shape
        
        # Store original input for residual connection
        x_orig = x
        
        # 1. Apply deformable attention
        x_deform = self.deformable_attention(x)  # (B, C, H, W)
        
        # 2. Apply sparse window attention
        # First, pad if necessary
        x_padded, original_size = reflection_pad_to_window_size(x, self.window_size)
        B, C, H_pad, W_pad = x_padded.shape
        
        # Partition into windows
        x_windows = window_partition(x_padded, self.window_size)  # (B*num_windows, C, ws, ws)
        nW = x_windows.shape[0] // B  # number of windows
        
        # Reshape for attention: (B*num_windows, ws*ws, C)
        x_windows = x_windows.view(-1, C, self.window_size * self.window_size)
        x_windows = x_windows.transpose(1, 2)
        
        # Apply window attention
        x_windows_attn = self.window_attention(x_windows)  # (B*num_windows, ws*ws, C)
        
        # Reshape back: (B*num_windows, C, ws, ws)
        x_windows_attn = x_windows_attn.transpose(1, 2)
        x_windows_attn = x_windows_attn.view(-1, C, self.window_size, self.window_size)
        
        # Reverse window partition
        x_window = window_reverse(x_windows_attn, self.window_size, H_pad, W_pad, B)
        
        # Remove padding
        x_window = remove_padding(x_window, original_size)
        
        # 3. Feature fusion
        # Flatten spatial dimensions for fusion
        x_deform_flat = x_deform.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x_window_flat = x_window.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Concatenate and fuse features
        x_fused = torch.cat([x_deform_flat, x_window_flat], dim=-1)  # (B, H*W, 2*C)
        x_fused = self.fusion_norm(x_fused)
        x_fused = self.fusion_proj(x_fused)  # (B, H*W, C)
        x_fused = self.fusion_dropout(x_fused)
        
        # Add residual connection
        x_orig_flat = x_orig.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x_out = x_orig_flat + x_fused
        
        # Final MLP
        x_out = x_out + self.final_mlp(x_out)
        
        # Reshape back to spatial format
        x_out = x_out.transpose(1, 2).reshape(B, C, H, W)
        
        return x_out


class MultiScaleHybridAttention(nn.Module):
    """
    Multi-scale hybrid attention that processes different scales of the input.
    """
    
    def __init__(self, dim: int, patch_size: int, num_heads: int = 8, 
                 num_layers: int = 2, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        
        # Multiple hybrid attention layers
        self.attention_layers = nn.ModuleList([
            HybridAttention(
                dim=dim, 
                patch_size=patch_size, 
                num_heads=num_heads,
                mlp_ratio=mlp_ratio, 
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-scale hybrid attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, P, P)
            
        Returns:
            torch.Tensor: Output tensor with multi-scale attention applied
        """
        # Apply multiple layers of hybrid attention
        for layer in self.attention_layers:
            x = layer(x)
        
        # Final normalization
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        return x