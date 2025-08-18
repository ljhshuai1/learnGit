"""
Complete hyperspectral image classification network using hybrid attention mechanism.

This module implements the full classification network that combines CAMixer's 
deformable attention and AST's sparse window attention for patch-based 
hyperspectral image classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from hybrid_attention import HybridAttention, MultiScaleHybridAttention
from utils import validate_patch_size, get_patch_info


class PatchEmbedding(nn.Module):
    """
    Embedding layer for hyperspectral image patches.
    """
    
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, 
                 norm_layer: Optional[nn.Module] = None):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Convolutional embedding
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        
        # Optional normalization
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, patch_size, patch_size)
        )
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of patch embedding.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, P, P)
            
        Returns:
            torch.Tensor: Embedded patches of shape (B, embed_dim, P, P)
        """
        B, C, H, W = x.shape
        
        # Apply projection
        x = self.proj(x)  # (B, embed_dim, P, P)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply normalization
        if hasattr(self.norm, '__call__'):
            # Reshape for layer norm if needed
            if isinstance(self.norm, nn.LayerNorm):
                x = x.flatten(2).transpose(1, 2)  # (B, P*P, embed_dim)
                x = self.norm(x)
                x = x.transpose(1, 2).reshape(B, self.embed_dim, H, W)
            else:
                x = self.norm(x)
        
        return x


class SpectralAttention(nn.Module):
    """
    Spectral attention mechanism for hyperspectral data.
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral attention to enhance important spectral bands.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Spectrally enhanced tensor
        """
        B, C, H, W = x.shape
        
        # Global average pooling
        y = self.avg_pool(x).view(B, C)
        
        # Channel attention
        y = self.fc(y).view(B, C, 1, 1)
        
        # Apply attention weights
        return x * y.expand_as(x)


class HyperspectralClassifier(nn.Module):
    """
    Complete hyperspectral image classification network with hybrid attention.
    """
    
    def __init__(self, 
                 in_channels: int,
                 num_classes: int,
                 patch_size: int,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 use_spectral_attention: bool = True,
                 use_multi_scale: bool = False):
        """
        Initialize hyperspectral classifier.
        
        Args:
            in_channels (int): Number of input spectral bands
            num_classes (int): Number of classification classes
            patch_size (int): Size of input patches (must be in [5, 7, 9, 11, 13, 15])
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            num_layers (int): Number of hybrid attention layers
            mlp_ratio (float): MLP expansion ratio
            dropout (float): Dropout rate
            use_spectral_attention (bool): Whether to use spectral attention
            use_multi_scale (bool): Whether to use multi-scale attention
        """
        super().__init__()
        
        # Validate patch size
        if not validate_patch_size(patch_size):
            raise ValueError(f"Unsupported patch size: {patch_size}")
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_spectral_attention = use_spectral_attention
        self.use_multi_scale = use_multi_scale
        
        # Get patch configuration
        self.patch_info = get_patch_info(patch_size)
        
        # Spectral attention (optional)
        if use_spectral_attention:
            self.spectral_attention = SpectralAttention(in_channels)
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            norm_layer=nn.LayerNorm
        )
        
        # Hybrid attention layers
        if use_multi_scale:
            self.attention_layers = MultiScaleHybridAttention(
                dim=embed_dim,
                patch_size=patch_size,
                num_heads=num_heads,
                num_layers=num_layers,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
        else:
            self.attention_layers = nn.ModuleList([
                HybridAttention(
                    dim=embed_dim,
                    patch_size=patch_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout
                ) for _ in range(num_layers)
            ])
        
        # Feature aggregation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize model weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using hybrid attention mechanism.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, P, P)
            
        Returns:
            torch.Tensor: Extracted features
        """
        # Apply spectral attention if enabled
        if self.use_spectral_attention:
            x = self.spectral_attention(x)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, P, P)
        
        # Apply hybrid attention layers
        if self.use_multi_scale:
            x = self.attention_layers(x)
        else:
            for layer in self.attention_layers:
                x = layer(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the classifier.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, P, P)
            
        Returns:
            torch.Tensor: Classification logits of shape (B, num_classes)
        """
        # Validate input shape
        B, C, H, W = x.shape
        if H != self.patch_size or W != self.patch_size:
            raise ValueError(f"Expected input size ({self.patch_size}, {self.patch_size}), got ({H}, {W})")
        
        # Extract features
        features = self.extract_features(x)  # (B, embed_dim, P, P)
        
        # Global pooling
        pooled_features = self.global_pool(features)  # (B, embed_dim, 1, 1)
        pooled_features = pooled_features.flatten(1)  # (B, embed_dim)
        
        # Classification
        logits = self.classifier(pooled_features)  # (B, num_classes)
        
        return logits
    
    def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract attention maps for visualization.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, P, P)
            
        Returns:
            List[torch.Tensor]: Attention maps from each layer
        """
        attention_maps = []
        
        # Apply spectral attention if enabled
        if self.use_spectral_attention:
            x = self.spectral_attention(x)
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Extract attention maps from each layer
        if self.use_multi_scale:
            # For multi-scale, we'd need to modify the multi-scale module to return attention maps
            # This is a simplified version
            attention_maps.append(x.mean(dim=1, keepdim=True))  # Placeholder
        else:
            for i, layer in enumerate(self.attention_layers):
                x = layer(x)
                # Extract spatial attention pattern (simplified)
                attn_map = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
                attention_maps.append(attn_map)
        
        return attention_maps


class EnsembleHyperspectralClassifier(nn.Module):
    """
    Ensemble classifier using multiple patch sizes.
    """
    
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 patch_sizes: List[int] = [5, 7, 9, 11],
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        """
        Initialize ensemble classifier.
        
        Args:
            in_channels (int): Number of input spectral bands
            num_classes (int): Number of classification classes
            patch_sizes (List[int]): List of patch sizes to use
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            num_layers (int): Number of hybrid attention layers
            mlp_ratio (float): MLP expansion ratio
            dropout (float): Dropout rate
        """
        super().__init__()
        
        self.patch_sizes = patch_sizes
        self.num_models = len(patch_sizes)
        
        # Create individual classifiers for each patch size
        self.classifiers = nn.ModuleList([
            HyperspectralClassifier(
                in_channels=in_channels,
                num_classes=num_classes,
                patch_size=patch_size,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            ) for patch_size in patch_sizes
        ])
        
        # Ensemble fusion
        self.fusion = nn.Linear(num_classes * self.num_models, num_classes)
        
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of ensemble classifier.
        
        Args:
            inputs (List[torch.Tensor]): List of input tensors, one for each patch size
            
        Returns:
            torch.Tensor: Ensemble classification logits
        """
        if len(inputs) != self.num_models:
            raise ValueError(f"Expected {self.num_models} inputs, got {len(inputs)}")
        
        # Get predictions from each classifier
        predictions = []
        for i, (classifier, x) in enumerate(zip(self.classifiers, inputs)):
            pred = classifier(x)
            predictions.append(pred)
        
        # Concatenate predictions
        concat_preds = torch.cat(predictions, dim=1)
        
        # Fusion
        ensemble_logits = self.fusion(concat_preds)
        
        return ensemble_logits