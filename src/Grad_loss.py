"""
Loss Functions Module

This module provides:
- Utility functions for type checking and conversion
- Weighted loss base class
- Multi-task learning weighted loss
- RMSE and gradient RMSE loss functions
- SSIM integration for combined losses

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Optional
from .SSIM_loss import SSIM


def convert_to_tuple(params: Union[None, object, List, Tuple]) -> Optional[Tuple]:
    """Convert input to tuple if not None or already a tuple.

    Args:
        params: Input to convert to tuple

    Returns:
        Tuple version of input or None if input was None
    """
    if params is None:
        return None
    if not isinstance(params, (list, tuple)):
        params = (params,)
    if isinstance(params, list):
        params = tuple(params)
    return params


def validate_param_type(
        param: object,
        param_name: str,
        valid_types: Union[None, type, Tuple[type]] = None,
        invalid_types: Union[None, type, Tuple[type]] = None
) -> None:
    """Validate parameter type against allowed/disallowed types.

    Args:
        param: Parameter to check
        param_name: Name of parameter for error messages
        valid_types: Type or tuple of types that are allowed
        invalid_types: Type or tuple of types that are disallowed

    Raises:
        TypeError: If type validation fails
    """
    valid_types = convert_to_tuple(valid_types)
    invalid_types = convert_to_tuple(invalid_types)

    if valid_types and not isinstance(param, valid_types):
        raise TypeError(
            f"Parameter '{param_name}' should be instance of {valid_types}, "
            f"but got {type(param)}"
        )
    if invalid_types and isinstance(param, invalid_types):
        raise TypeError(
            f"Parameter '{param_name}' should not be instance of {invalid_types}, "
            f"but got {type(param)}"
        )


class WeightedLoss(nn.Module):
    """Base class for weighted loss functions.

    Provides basic structure for loss functions that may need gradient computation.
    """

    def __init__(self):
        super().__init__()
        self.use_grads = False

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """Forward pass for loss computation.

        Args:
            losses: Input loss tensor

        Returns:
            Output loss tensor
        """
        return losses


class MultiTaskLoss(WeightedLoss):
    """Multi-task learning weighted loss function.

    Automatically learns weights for multiple loss terms during training.

    Args:
        num_losses: Number of loss terms to weight
        bound_param: Small positive value to bound weights away from zero
    """

    def __init__(self, num_losses: int, bound_param: float = 0.0):
        super().__init__()

        validate_param_type(num_losses, "num_losses", int, bool)
        if num_losses <= 0:
            raise ValueError(
                f"num_losses must be positive, got {num_losses}"
            )

        validate_param_type(bound_param, "bound_param", float)

        self.num_losses = num_losses
        self.bounded = bound_param > 1e-6
        self.bound_param = bound_param ** 2

        # Learnable weights for each loss term
        self.weights = nn.Parameter(
            torch.ones(num_losses, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """Compute weighted multi-task loss.

        Args:
            losses: List of loss tensors (must match num_losses)

        Returns:
            Combined weighted loss
        """
        if len(losses) != self.num_losses:
            raise ValueError(
                f"Expected {self.num_losses} losses, got {len(losses)}"
            )

        total_loss = 0.0
        squared_weights = torch.square(self.weights)

        for i in range(self.num_losses):
            if self.bounded:
                weight = squared_weights[i] + self.bound_param
                reg_term = squared_weights[i] + self.bound_param
            else:
                weight = squared_weights[i]
                reg_term = squared_weights[i] + 1.0

            weighted_loss = 0.5 * (losses[i] / weight) + torch.log(reg_term)
            total_loss += weighted_loss

        return total_loss


class RMSE(nn.Module):
    """Root Mean Squared Error loss.

    Args:
        reduction: Reduction method ('mean' or 'sum')
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        validate_param_type(reduction, "reduction", str)
        if reduction not in ['mean', 'sum']:
            raise ValueError(f"Reduction must be 'mean' or 'sum', got {reduction}")
        self.reduction = reduction

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute RMSE between predictions and targets.

        Args:
            predictions: Model predictions
            targets: Ground truth values

        Returns:
            RMSE loss
        """
        # Ensure float dtype and flatten spatial dimensions
        predictions = predictions.float().view(predictions.size(0), -1)
        targets = targets.float().view(targets.size(0), -1)

        squared_diff = torch.square(predictions - targets)

        if self.reduction == 'mean':
            return torch.sqrt(torch.mean(squared_diff))
        return torch.sqrt(torch.sum(squared_diff))


def compute_gradients(
        data: torch.Tensor,
        delta: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute finite differences along spatial dimensions.

    Args:
        data: Input tensor (B,C,H,W)
        delta: Step size for finite differences

    Returns:
        Tuple of (gradient_x, gradient_y)
    """
    grad_x = data[:, :, :, delta:] - data[:, :, :, :-delta]
    grad_y = data[:, :, delta:, :] - data[:, :, :-delta, :]
    return grad_x, grad_y


class GradientRMSE(nn.Module):
    """Combined loss with RMSE, gradient RMSE and SSIM terms.

    Args:
        dynamic_weighting: Whether to use learned weights for loss terms
    """

    def __init__(self, dynamic_weighting: bool = True):
        super().__init__()
        self.dynamic_weighting = dynamic_weighting
        self.mtl_2term = MultiTaskLoss(num_losses=2)
        self.mtl_3term = MultiTaskLoss(num_losses=3)
        self.ssim = SSIM()
        self.rmse = RMSE(reduction='sum')

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            predictions: Model predictions
            targets: Ground truth values

        Returns:
            Combined loss value
        """
        # Compute individual loss terms
        intensity_loss = self.rmse(predictions, targets)
        gradient_loss = self._gradient_loss(predictions, targets)
        ssim_loss = -torch.log(self.ssim(predictions, targets) + 1e-8)

        if self.dynamic_weighting:
            return self.mtl_2term([intensity_loss, gradient_loss])
        return self.mtl_3term([intensity_loss, gradient_loss, ssim_loss])

    def _gradient_loss(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient RMSE between predictions and targets."""
        pred_grad_x, pred_grad_y = compute_gradients(predictions)
        target_grad_x, target_grad_y = compute_gradients(targets)

        loss_x = self.rmse(pred_grad_x, target_grad_x)
        loss_y = self.rmse(pred_grad_y, target_grad_y)
        return loss_x + loss_y


def test_gradient_rmse():
    """Test GradientRMSELoss with sample data."""
    # Create test data
    predictions = torch.arange(1, 2 * 3 * 4 * 5 + 1.0).reshape(2, 3, 4, 5)
    predictions.requires_grad_(True)
    targets = torch.full((2, 3, 4, 5), 2.0)

    # Compute loss
    loss_fn = GradientRMSE()
    loss = loss_fn(predictions, targets)
    print(f"Gradient RMSE Loss: {loss.item()}")


if __name__ == "__main__":
    test_gradient_rmse()
