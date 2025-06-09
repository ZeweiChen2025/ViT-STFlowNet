"""
Vision Transformer (ViT) Implementation

This module implements a complete Vision Transformer model with:
- Patch embedding encoder
- Transformer encoder blocks
- Transformer decoder blocks
- Patch prediction decoder

Reference:
    Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
"""

import torch
import torch.nn as nn
from einops import rearrange
from torchinfo import summary
from typing import Tuple, Optional

from .cell_utils import to_2tuple
from .layer import Decoder, Encoder


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) model for image processing tasks.

    Args:
        image_size: Input image size (height, width)
        in_channels: Number of input channels
        out_channels: Number of output channels
        patch_size: Size of image patches (square)
        encoder_depths: Number of encoder transformer blocks
        encoder_embed_dim: Encoder embedding dimension
        encoder_num_heads: Number of encoder attention heads
        decoder_depths: Number of decoder transformer blocks
        decoder_embed_dim: Decoder embedding dimension
        decoder_num_heads: Number of decoder attention heads
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
        dropout_rate: Dropout probability
        compute_dtype: Computation dtype (default: torch.float32)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (64, 104),
        in_channels: int = 5,
        out_channels: int = 3,
        patch_size: int = 8,
        encoder_depths: int = 12,
        encoder_embed_dim: int = 768,
        encoder_num_heads: int = 12,
        decoder_depths: int = 8,
        decoder_embed_dim: int = 512,
        decoder_num_heads: int = 16,
        mlp_ratio: int = 4,
        dropout_rate: float = 0.2,
        compute_dtype: torch.dtype = torch.float32
    ):
        super().__init__()

        # Validate and process input dimensions
        image_size = to_2tuple(image_size)
        grid_size = (image_size[0] // patch_size,
                     image_size[1] // patch_size)

        # Store configuration parameters
        self.img_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.compute_dtype = compute_dtype

        # Initialize encoder
        self.encoder = Encoder(
            grid_size=grid_size,
            in_channels=in_channels,
            patch_size=patch_size,
            depths=encoder_depths,
            embed_dim=encoder_embed_dim,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
            compute_dtype=compute_dtype
        )

        # Initialize decoder embedding projection
        self.decoder_embedding = nn.Linear(
            encoder_embed_dim,
            decoder_embed_dim
        )
        self._init_weights(self.decoder_embedding)

        # Initialize decoder
        self.decoder = Decoder(
            grid_size=grid_size,
            depths=decoder_depths,
            embed_dim=decoder_embed_dim,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
            compute_dtype=compute_dtype
        )

        # Initialize prediction head
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size ** 2 * out_channels
        )
        self._init_weights(self.decoder_pred)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for linear layers."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Vision Transformer.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Reconstructed image tensor of shape
            (batch_size, out_channels, height, width)
        """
        # Encoder forward pass
        x = self.encoder(x)

        # Project to decoder dimension
        x = self.decoder_embedding(x)

        # Decoder forward pass
        x = self.decoder(x)

        # Predict output patches
        patches = self.decoder_pred(x)

        # Reshape patches to full image
        images = rearrange(
            patches,
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=self.img_size[0] // self.patch_size,
            w=self.img_size[1] // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.out_channels
        )

        return images.to(torch.float32)


def test_vit_model():
    """Test function for Vision Transformer implementation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration parameters
    H, W = 64, 104
    config = {
        "image_size": (H, W),
        "in_channels": 5,
        "out_channels": 3,
        "patch_size": 8,
        "encoder_depths": 3,
        "encoder_embed_dim": 768,
        "encoder_num_heads": 12,
        "decoder_depths": 2,
        "decoder_embed_dim": 512,
        "decoder_num_heads": 8,
        "mlp_ratio": 4,
        "dropout_rate": 0.1,
        "compute_dtype": torch.float16
    }

    # Initialize model
    model = VisionTransformer(**config).to(device)

    # Create test input
    dummy_input = torch.rand(2, 5, H, W).to(device)

    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Print model summary
    summary(model, (2, 5, H, W), device="cpu")


if __name__ == "__main__":
    test_vit_model()
