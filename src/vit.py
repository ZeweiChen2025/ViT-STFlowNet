"""
Vision Transformer (ViT) Model for Spatiotemporal Flow Prediction

This implementation features:
- Encoder-Decoder architecture with patch-based processing
- Flexible configuration for different input/output dimensions
- Mixed-precision training support
"""

import torch
import torch.nn as nn
from einops import rearrange
from torchinfo import summary

from .cell_utils import to_2tuple
from .layer import Decoder, Encoder


class ViT(nn.Module):
    """Vision Transformer architecture for spatiotemporal flow prediction

    Args:
        image_size: Tuple of (height, width) for input images
        in_channels: Number of input channels
        out_channels: Number of output channels
        patch_size: Size of image patches (square)
        encoder_depths: Number of transformer blocks in encoder
        encoder_embed_dim: Embedding dimension in encoder
        encoder_num_heads: Number of attention heads in encoder
        decoder_depths: Number of transformer blocks in decoder
        decoder_embed_dim: Embedding dimension in decoder
        decoder_num_heads: Number of attention heads in decoder
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
        dropout_rate: Dropout probability
        compute_dtype: Computation dtype (e.g., torch.float16 for mixed precision)
    """

    def __init__(self,
                 image_size=(64, 104),
                 in_channels=5,
                 out_channels=3,
                 patch_size=8,
                 encoder_depths=12,
                 encoder_embed_dim=768,
                 encoder_num_heads=12,
                 decoder_depths=8,
                 decoder_embed_dim=512,
                 decoder_num_heads=16,
                 mlp_ratio=4,
                 dropout_rate=0.2,
                 compute_dtype=torch.float32):
        super(ViT, self).__init__()

        # Convert image_size to tuple and compute grid dimensions
        image_size = to_2tuple(image_size)
        grid_size = (image_size[0] // patch_size,
                     image_size[1] // patch_size)

        # Store architectural parameters
        self.img_size = image_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.in_channels = in_channels

        # Encoder configuration
        self.encoder_depths = encoder_depths
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_num_heads = encoder_num_heads

        # Decoder configuration
        self.decoder_depths = decoder_depths
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_num_heads = decoder_num_heads

        # Initialize encoder module
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

        # Dimensionality reduction between encoder and decoder
        self.decoder_embedding = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        nn.init.xavier_uniform_(self.decoder_embedding.weight)
        if self.decoder_embedding.bias is not None:
            nn.init.zeros_(self.decoder_embedding.bias)

        # Initialize decoder module
        self.decoder = Decoder(
            grid_size=grid_size,
            depths=decoder_depths,
            embed_dim=decoder_embed_dim,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
            compute_dtype=compute_dtype
        )

        # Prediction head - reconstructs patches to output dimensions
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size ** 2 * out_channels
        )
        nn.init.xavier_uniform_(self.decoder_pred.weight)
        if self.decoder_pred.bias is not None:
            nn.init.zeros_(self.decoder_pred.bias)

    def forward(self, x):
        """Forward pass through the ViT model

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Reconstructed flow field of shape (batch_size, out_channels, height, width)
        """
        # Encoder processing
        x = self.encoder(x)

        # Project to decoder dimension
        x = self.decoder_embedding(x)

        # Decoder processing
        x = self.decoder(x)

        # Predict output patches
        images = self.decoder_pred(x)

        # Convert to full precision for final output
        images = images.to(torch.float32)

        # Reconstruct patches to full resolution output
        images_out = rearrange(
            images,
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=self.img_size[0] // self.patch_size,
            w=self.img_size[1] // self.patch_size,
            c=self.out_channels,
            p1=self.patch_size,
            p2=self.patch_size
        )

        return images_out


if __name__ == "__main__":
    """Test script for model verification"""

    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Example configuration for testing
    H, W = 64, 104
    model = ViT(
        image_size=(H, W),
        in_channels=5,
        out_channels=3,
        patch_size=8,
        encoder_depths=3,
        encoder_embed_dim=768,
        encoder_num_heads=12,
        decoder_depths=2,
        decoder_embed_dim=512,
        decoder_num_heads=8,
        mlp_ratio=4,
        dropout_rate=0.1,
        compute_dtype=torch.float16
    ).to(device)

    # Create test input tensor
    dummy_input = torch.rand(2, 5, H, W).to(device)

    # Forward pass test
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Print model summary
    summary(model, (2, 5, H, W), device="cpu")