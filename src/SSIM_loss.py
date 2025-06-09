"""
Structural Similarity Index (SSIM) Implementation

This module provides:
- Gaussian window creation for SSIM computation
- SSIM calculation between two images
- PyTorch Module wrapper for SSIM loss

"""

import torch
import torch.nn.functional as F
from math import exp
from typing import Tuple, Optional


def gaussian_kernel(window_size: int, sigma: float) -> torch.Tensor:
    """Create 1D Gaussian kernel.

    Args:
        window_size: Size of the Gaussian window
        sigma: Standard deviation of Gaussian distribution

    Returns:
        1D tensor containing Gaussian weights
    """
    kernel = torch.Tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return kernel / kernel.sum()


def create_ssim_window(window_size: int, num_channels: int) -> torch.Tensor:
    """Create 2D Gaussian window for SSIM computation.

    Args:
        window_size: Size of the square window
        num_channels: Number of image channels

    Returns:
        4D tensor (num_channels, 1, window_size, window_size) 
        containing Gaussian weights
    """
    # Create 1D kernel
    kernel_1d = gaussian_kernel(window_size, sigma=1.5).unsqueeze(1)

    # Create 2D kernel through outer product
    kernel_2d = kernel_1d.mm(kernel_1d.t()).float()
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

    # Expand to match input channels
    window = kernel_2d.expand(num_channels, 1, window_size, window_size).contiguous()
    return window


def compute_ssim(
        img1: torch.Tensor,
        img2: torch.Tensor,
        window: torch.Tensor,
        window_size: int,
        num_channels: int,
        size_average: bool = True
) -> torch.Tensor:
    """Compute SSIM between two images.

    Args:
        img1: First image tensor (N,C,H,W)
        img2: Second image tensor (N,C,H,W)
        window: Gaussian window tensor
        window_size: Size of the Gaussian window
        num_channels: Number of image channels
        size_average: Whether to average SSIM map

    Returns:
        SSIM score (scalar if size_average=True, otherwise per-image scores)
    """
    # Compute means
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=num_channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=num_channels)

    # Compute mean squares
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Compute variances
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=num_channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=num_channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=num_channels) - mu1_mu2

    # SSIM constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Compute SSIM map
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    ssim_map /= ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(dim=[1, 2, 3])


class SSIM(torch.nn.Module):
    """Structural Similarity Index (SSIM) loss module.

    Args:
        window_size: Size of Gaussian window (default: 11)
        size_average: Whether to average SSIM over batch (default: True)
    """

    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.num_channels = 1
        self.window = create_ssim_window(window_size, self.num_channels)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute SSIM between two images.

        Args:
            img1: First image tensor (N,C,H,W)
            img2: Second image tensor (N,C,H,W)

        Returns:
            SSIM score
        """
        _, num_channels, _, _ = img1.size()

        # Recreate window if input channels changed or device changed
        if (num_channels != self.num_channels or
                self.window.device != img1.device or
                self.window.dtype != img1.dtype):
            self.window = create_ssim_window(self.window_size, num_channels)
            self.window = self.window.to(img1.device).type_as(img1)
            self.num_channels = num_channels

        return compute_ssim(
            img1,
            img2,
            self.window,
            self.window_size,
            self.num_channels,
            self.size_average
        )


def test_ssim():
    """Test SSIM implementation with identical images."""
    # Create test images
    img1 = torch.ones(1, 3, 96, 96)
    img2 = torch.ones(1, 3, 96, 96)

    # Compute SSIM
    ssim_module = SSIM()
    score = ssim_module(img1, img2)

    print(f"SSIM Score: {score.item():.4f} (should be ~1.0 for identical images)")


if __name__ == "__main__":
    test_ssim()
