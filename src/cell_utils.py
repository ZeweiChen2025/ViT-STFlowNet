"""
Positional Embedding and Patch Processing Utilities

This module provides functions for:
- Generating 1D and 2D sinusoidal positional embeddings
- Converting images to/from patch representations
- Various helper functions for tuple processing

Copyright 2022 Huawei Technologies Co., Ltd
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
"""

import numpy as np
import torch
from typing import Union, List, Tuple, Optional

__all__ = ['to_2tuple', 'unpatchify', 'patchify', 'get_2d_sin_cos_pos_embed']


def to_2tuple(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Convert input to a 2-element tuple.
    
    Args:
        value: Input value to convert. If already a tuple, returns unchanged.
               If int, returns (value, value).
               
    Returns:
        Tuple of two elements with the same value.
    """
    return value if isinstance(value, tuple) else (value, value)


def get_2d_sin_cos_pos_embed(
    embed_dim: int, 
    grid_size: Union[int, Tuple[int, int]]
) -> np.ndarray:
    """Generate 2D sine-cosine positional embeddings.
    
    Args:
        embed_dim: Output dimension for each position.
        grid_size: Grid dimensions (height, width) or single int for square grid.
        
    Returns:
        Positional embedding array with shape (1, grid_height*grid_width, embed_dim)
    """
    grid_h, grid_w = to_2tuple(grid_size)
    
    # Create grid coordinates
    grid_h_coords = np.arange(grid_h, dtype=np.float32)
    grid_w_coords = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_w_coords, grid_h_coords)  # Note: width first
    grid = np.stack(grid, axis=0)  # Shape: (2, grid_h, grid_w)
    
    # Reshape and get embeddings
    grid = grid.reshape([2, 1, grid_h, grid_w])
    pos_embed = get_2d_sin_cos_pos_embed_from_grid(embed_dim, grid)
    return np.expand_dims(pos_embed, 0)  # Add batch dimension


def get_2d_sin_cos_pos_embed_from_grid(
    embed_dim: int, 
    grid: np.ndarray
) -> np.ndarray:
    """Generate 2D positional embeddings from grid coordinates.
    
    Uses half of embedding dimensions for height and half for width.
    
    Args:
        embed_dim: Total output dimension for each position.
        grid: Grid coordinates array of shape (2, ...)
        
    Returns:
        Positional embeddings with shape (grid_h*grid_w, embed_dim)
    """
    grid = np.array(grid)
    emb_h = get_1d_sin_cos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sin_cos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    return np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)


def get_1d_sin_cos_pos_embed_from_grid(
    embed_dim: int, 
    positions: np.ndarray
) -> np.ndarray:
    """Generate 1D sine-cosine positional embeddings.
    
    Args:
        embed_dim: Output dimension for each position.
        positions: 1D array of position indices.
        
    Returns:
        Positional embeddings with shape (num_positions, embed_dim)
    """
    assert embed_dim % 2 == 0, "Embed dimension must be even"
    
    # Calculate omega values for frequency scaling
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / (10000 ** omega)  # Shape: (D/2,)
    
    # Reshape positions and compute outer product
    positions = np.array(positions).reshape(-1)  # Shape: (M,)
    out = np.einsum('m,d->md', positions, omega)  # Shape: (M, D/2)
    
    # Concatenate sine and cosine components
    emb_sin = np.sin(out)  # Shape: (M, D/2)
    emb_cos = np.cos(out)  # Shape: (M, D/2)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # Shape: (M, D)


def patchify(
    image: np.ndarray, 
    patch_size: int = 16
) -> np.ndarray:
    """Convert image to sequence of flattened patches.
    
    Args:
        image: Input image array of shape (H, W, C)
        patch_size: Size of square patches
        
    Returns:
        Array of flattened patches with shape (num_patches, patch_size*patch_size*C)
    """
    h, w, c = image.shape
    assert h % patch_size == 0 and w % patch_size == 0, "Image dimensions must be divisible by patch size"
    
    # Reshape into patches and flatten
    patches = image.reshape(
        h // patch_size,
        patch_size,
        w // patch_size,
        patch_size,
        c
    )
    patches = np.transpose(patches, (0, 2, 1, 3, 4))  # Group patch dimensions
    return patches.reshape(-1, patch_size * patch_size * c)


def unpatchify(
    patches: torch.Tensor,
    img_size: Tuple[int, int] = (192, 384),
    patch_size: int = 16,
    nchw: bool = False
) -> torch.Tensor:
    """Convert sequence of patches back to image.
    
    Args:
        patches: Input patches tensor of shape (N, num_patches, patch_dim)
        img_size: Original image dimensions (height, width)
        patch_size: Size of square patches
        nchw: If True, output is in NCHW format instead of NHWC
        
    Returns:
        Reconstructed image tensor of shape (N, H, W, C) or (N, C, H, W)
    """
    h, w = img_size
    n_patches = (h // patch_size) * (w // patch_size)
    c = patches.shape[-1] // (patch_size * patch_size)
    
    # Reshape patches to image
    img = patches.reshape(
        patches.shape[0],
        h // patch_size,
        w // patch_size,
        patch_size,
        patch_size,
        c
    )
    img = img.transpose(1, 3)  # Swap patch and spatial dimensions
    img = img.reshape(patches.shape[0], h, w, c)
    
    if nchw:
        img = img.permute(0, 3, 1, 2)  # NHWC to NCHW
    return img


def test_positional_embedding():
    """Test function for positional embedding generation."""
    embed_dim = 16
    grid_size = (10, 10)
    
    pos_embed = get_2d_sin_cos_pos_embed(embed_dim, grid_size)
    
    print("Positional Embedding Shape:", pos_embed.shape)
    print("Positional Embedding Sample:\n", pos_embed)


if __name__ == "__main__":
    test_positional_embedding()
