"""
Structural Similarity Index (SSIM) Loss Implementation

This module implements the SSIM metric for image quality assessment, which considers:
- Luminance comparison
- Contrast comparison
- Structure comparison
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """Create 1D Gaussian distribution tensor

    Args:
        window_size: Size of the Gaussian window
        sigma: Standard deviation of Gaussian distribution

    Returns:
        1D tensor containing Gaussian weights
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                         for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> torch.Tensor:
    """Create 2D Gaussian window for SSIM computation

    Args:
        window_size: Size of the square window
        channel: Number of image channels

    Returns:
        4D tensor (channel, 1, window_size, window_size) containing Gaussian weights
    """
    # Create 1D Gaussian kernel
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)

    # Create 2D window through outer product
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)

    # Expand to match input channels and ensure contiguous memory
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1: torch.Tensor,
          img2: torch.Tensor,
          window: torch.Tensor,
          window_size: int,
          channel: int,
          size_average: bool = True) -> torch.Tensor:
    """Compute SSIM between two images using given window

    Args:
        img1: First image tensor (N,C,H,W)
        img2: Second image tensor (N,C,H,W)
        window: Gaussian window tensor
        window_size: Size of the Gaussian window
        channel: Number of image channels
        size_average: Whether to average SSIM map

    Returns:
        SSIM score (scalar if size_average=True, otherwise per-image scores)
    """
    # Compute local means using Gaussian filtering
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    # Compute mean squares and product
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    # SSIM stability constants
    C1 = 0.01 ** 2  # For luminance stability
    C2 = 0.03 ** 2  # For contrast stability

    # Compute SSIM index map
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()  # Global average
    else:
        return ssim_map.mean(1).mean(1).mean(1)  # Per-image average


class SSIM(torch.nn.Module):
    """Structural Similarity Index Measure (SSIM) loss module

    Args:
        window_size: Size of Gaussian window (default: 11)
        size_average: Whether to average SSIM over batch (default: True)
    """

    def __init__(self, window_size: int = 11, size_average: bool = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1  # Initial channel dimension
        self.window = create_window(window_size, self.channel)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute SSIM between two images

        Args:
            img1: First image tensor (N,C,H,W)
            img2: Second image tensor (N,C,H,W)

        Returns:
            SSIM score
        """
        _, channel, _, _ = img1.size()

        # Recreate window if input channels changed or device changed
        if channel != self.channel or self.window.data.type() != img1.data.type():
            self.window = create_window(self.window_size, channel)

            # Move window to same device as input
            if img1.is_cuda:
                self.window = self.window.cuda(img1.get_device())
            self.window = self.window.type_as(img1)

            self.channel = channel  # Update channel count

        return _ssim(img1, img2, self.window, self.window_size, channel, self.size_average)


if __name__ == "__main__":
    """Test SSIM implementation with identical images"""
    # Create test images (should produce perfect SSIM=1.0)
    img1 = torch.ones(1, 3, 96, 96)  # Batch of 1, 3 channels, 96x96 pixels
    img2 = torch.ones(1, 3, 96, 96)  # Identical to img1

    # Initialize SSIM module
    ssim_model = SSIM()

    # Compute SSIM score
    ssim_score = ssim_model(img1, img2)
    print(f"SSIM Score: {ssim_score.item():.4f} (should be 1.0 for identical images)")