"""
Transformer Encoder and Decoder Implementation

This module provides:
- Multi-head self-attention mechanism
- Transformer blocks with residual connections
- Patch embedding layer
- Complete encoder and decoder architectures

The implementation follows standard transformer architecture with:
- Layer normalization
- Residual connections
- Multi-head attention
- Feed-forward networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from .cell_utils import get_2d_sin_cos_pos_embed

__all__ = ['Decoder', 'Encoder']


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism.

    Args:
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads
        dropout_rate: Dropout probability
        compute_dtype: Computation dtype (default: torch.float16)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_rate: float = 1.0,
        compute_dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.compute_dtype = compute_dtype

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Linear projections
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Initialize weights
        for linear in [self.query, self.key, self.value, self.proj]:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

        # Dropout layers
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.proj_dropout = nn.Dropout(dropout_rate)

        # Softmax for attention scores
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape and transpose input for multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len = x.shape[:2]
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Output tensor of same shape as input
        """
        # Apply layer normalization
        x_norm = self.layer_norm(x)

        # Project inputs to query, key, value
        q = self.query(x_norm)
        k = self.key(x_norm)
        v = self.value(x_norm)

        # Reshape for multi-head attention
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))

        # Apply softmax and dropout
        attn_probs = self.softmax(attn_scores)
        attn_probs = self.attention_dropout(attn_probs)

        # Apply attention to values
        context = torch.matmul(attn_probs, v)

        # Reshape back to original dimensions
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(x.shape[0], -1, self.embed_dim)

        # Project and apply final dropout
        output = self.proj(context)
        output = self.proj_dropout(output)

        return output


class FeedForwardNetwork(nn.Module):
    """Feed-forward network for transformer blocks.

    Args:
        embed_dim: Input/output dimension
        mlp_ratio: Ratio of hidden dim to embed_dim
        dropout_rate: Dropout probability
        compute_dtype: Computation dtype (default: torch.float16)
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 1.0,
        compute_dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.compute_dtype = compute_dtype

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for feed-forward network."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x.to(self.compute_dtype)


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward layers.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of hidden dim to embed_dim in FFN
        dropout_rate: Dropout probability
        compute_dtype: Computation dtype (default: torch.float16)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 1.0,
        compute_dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Attention components
        self.attn_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attention = MultiHeadAttention(
            embed_dim, num_heads, dropout_rate, compute_dtype)

        # Feed-forward components
        self.ffn_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(
            embed_dim, mlp_ratio, dropout_rate, compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for transformer block."""
        # Self-attention with residual
        h = x
        x = self.attn_norm(x)
        x = self.attention(x)
        x = x + h

        # Feed-forward with residual
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x


class PatchEmbedding(nn.Module):
    """Patch embedding layer for vision transformers.

    Args:
        in_channels: Input channels
        embed_dim: Embedding dimension
        patch_size: Size of patches (square)
        compute_dtype: Computation dtype (default: torch.float16)
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: Tuple[int, int] = (16, 16),
        compute_dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.patch_size = patch_size

        # Convolutional projection
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for patch embedding.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Tensor of shape (batch_size, num_patches, embed_dim)
        """
        x = self.proj(x)  # (batch_size, embed_dim, h', w')
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x.to(self.compute_dtype)


class Encoder(nn.Module):
    """Transformer encoder for vision tasks.

    Args:
        grid_size: Grid size of patches (height, width)
        in_channels: Input channels
        patch_size: Size of patches (height, width)
        depths: Number of transformer blocks
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of hidden dim to embed_dim in FFN
        dropout_rate: Dropout probability
        compute_dtype: Computation dtype (default: torch.float16)
    """

    def __init__(
        self,
        grid_size: Tuple[int, int],
        in_channels: int,
        patch_size: Tuple[int, int],
        depths: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 1.0,
        compute_dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.depths = depths
        self.compute_dtype = compute_dtype

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            in_channels, embed_dim, patch_size, compute_dtype)

        # Positional embedding
        pos_embed = get_2d_sin_cos_pos_embed(embed_dim, grid_size)
        self.pos_embed = nn.Parameter(
            torch.tensor(pos_embed, dtype=torch.float32),
            requires_grad=False
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads,
                mlp_ratio,
                dropout_rate,
                compute_dtype
            ) for _ in range(depths)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for encoder.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Encoded features of shape (batch_size, num_patches, embed_dim)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)

        return x


class Decoder(nn.Module):
    """Transformer decoder for vision tasks.

    Args:
        grid_size: Grid size of patches (height, width)
        depths: Number of transformer blocks
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of hidden dim to embed_dim in FFN
        dropout_rate: Dropout probability
        compute_dtype: Computation dtype (default: torch.float16)
    """

    def __init__(
        self,
        grid_size: Tuple[int, int],
        depths: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 1.0,
        compute_dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.compute_dtype = compute_dtype

        # Positional embedding
        pos_embed = get_2d_sin_cos_pos_embed(embed_dim, grid_size)
        self.pos_embed = nn.Parameter(
            torch.tensor(pos_embed, dtype=torch.float32),
            requires_grad=False
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads,
                mlp_ratio,
                dropout_rate,
                compute_dtype
            ) for _ in range(depths)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for decoder.

        Args:
            x: Input tensor of shape (batch_size, num_patches, embed_dim)

        Returns:
            Decoded features of same shape as input
        """
        # Add positional embedding
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)

        return x
