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
from .SSIM_loss import SSIM


def _convert_to_tuple(params):
    """Convert input to tuple if not None or already a tuple

    Args:
        params: Input to convert (scalar, list, tuple or None)

    Returns:
        Tuple version of input or None if input was None
    """
    if params is None:
        return params
    if not isinstance(params, (list, tuple)):
        params = (params,)
    if isinstance(params, list):
        params = tuple(params)
    return params


def check_param_type(param, param_name, data_type=None, exclude_type=None):
    """Validate parameter type against allowed/disallowed types

    Args:
        param: Parameter to check
        param_name: Name of parameter for error messages
        data_type: Type or tuple of types that are allowed
        exclude_type: Type or tuple of types that are disallowed

    Raises:
        TypeError: If type validation fails
    """
    data_type = _convert_to_tuple(data_type)
    exclude_type = _convert_to_tuple(exclude_type)
    if data_type and not isinstance(param, data_type):
        raise TypeError(f"The type of {param_name} should be instance of {data_type}, but got {type(param)}")
    if exclude_type and isinstance(param, exclude_type):
        raise TypeError(f"The type of {param_name} should not be instance of {exclude_type}, but got {type(param)}")


class WeightedLossCell(nn.Module):
    """Base class for weighted loss computation

    Provides framework for loss functions that may need gradient computation.
    Subclasses should override forward() method.
    """

    def __init__(self):
        super(WeightedLossCell, self).__init__()
        self.use_grads = False  # Flag for gradient usage

    def forward(self, losses):
        """Basic forward pass (should be overridden by subclasses)

        Args:
            losses: Input loss tensor

        Returns:
            Output loss tensor
        """
        return losses


class MTLWeightedLoss(WeightedLossCell):
    """Multi-task learning weighted loss function with automatic weight adaptation

    Args:
        num_losses: Number of loss terms to weight
        bound_param: Small positive value to bound weights away from zero
    """

    def __init__(self, num_losses, bound_param=0.0):
        super(MTLWeightedLoss, self).__init__()
        # Validate input types
        check_param_type(num_losses, "num_losses", data_type=int, exclude_type=bool)
        if num_losses <= 0:
            raise ValueError(f"num_losses should be positive, got {num_losses}")

        check_param_type(bound_param, "bound_param", data_type=float)

        self.num_losses = num_losses
        self.bounded = bound_param > 1e-6  # Whether to use lower bound
        self.bound_param = bound_param ** 2  # Squared bound parameter

        # Learnable weights for each loss term
        self.params = nn.Parameter(torch.ones(num_losses, dtype=torch.float32), requires_grad=True)
        self.pow = torch.pow  # Power function for weight adjustment

    def forward(self, losses):
        """Compute weighted multi-task loss with regularization

        Args:
            losses: List of loss tensors (must match num_losses)

        Returns:
            Combined weighted loss with regularization terms
        """
        loss_sum = 0
        params = self.pow(self.params, 2)  # Square parameters to ensure positivity

        for i in range(self.num_losses):
            if self.bounded:
                weight = params[i] + self.bound_param
                reg = params[i] + self.bound_param
            else:
                weight = params[i]
                reg = params[i] + 1.0

            # Weighted loss + regularization term
            weighted_loss = 0.5 * (losses[i] / weight) + torch.log(reg)
            loss_sum += weighted_loss

        return loss_sum


class RMSE(nn.Module):
    """Root Mean Squared Error loss

    Args:
        reduction: Reduction method ('mean' or 'sum')
    """

    def __init__(self, reduction='mean'):
        super(RMSE, self).__init__()
        self.reduction = reduction

    def forward(self, predictions, targets):
        """Compute RMSE between predictions and targets

        Args:
            predictions: Model predictions
            targets: Ground truth values

        Returns:
            RMSE loss
        """
        # Ensure float32 precision
        predictions = predictions.float()
        targets = targets.float()

        # Flatten spatial dimensions while keeping batch dimension
        predictions_flat = predictions.view(predictions.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        # Compute squared differences
        difference = (predictions_flat - targets_flat) ** 2

        if self.reduction == 'mean':
            return difference.mean().sqrt()  # Mean over all elements
        elif self.reduction == 'sum':
            return difference.sum().sqrt()  # Sum over all elements
        return difference  # Return per-element differences


def derivation(data, delta=2):
    """Compute finite differences along spatial dimensions

    Args:
        data: Input tensor (B,C,H,W)
        delta: Step size for finite differences

    Returns:
        Tuple of (gradient_x, gradient_y)
    """
    # Compute x-direction gradients
    grad_x = data[:, :, delta:] - data[:, :, :-delta]
    # Compute y-direction gradients
    grad_y = data[:, delta:, :] - data[:, :-delta, :]
    return grad_x, grad_y


class GradientRMSE(nn.Module):
    """Combined loss with RMSE, gradient RMSE and SSIM terms

    Args:
        loss_weight: Weight for gradient loss term
        ssim_weight: Weight for SSIM loss term
        dynamic_flag: Whether to use learned weights (MTL)
    """

    def __init__(self, loss_weight=750.0, ssim_weight=150.0, dynamic_flag=True):
        super(GradientRMSE, self).__init__()
        self.loss_weight = loss_weight
        self.dynamic_flag = dynamic_flag
        self.ssim_weight = ssim_weight

        # Multi-task learning weight adapters
        self.mtl2 = MTLWeightedLoss(num_losses=2)  # For intensity + gradient
        self.mtl3 = MTLWeightedLoss(num_losses=3)  # For intensity + gradient + SSIM
        self.ssim = SSIM()  # Structural similarity metric

    def forward(self, logits, labels):
        """Compute combined loss

        Args:
            logits: Model predictions
            labels: Ground truth values

        Returns:
            Combined loss value
        """
        # Compute individual loss terms
        int_loss = RMSE(reduction='sum')(logits, labels)  # Intensity RMSE
        grad_loss = self.gradient_loss(logits, labels)  # Gradient RMSE
        ssim_loss = -torch.log(self.ssim(logits, labels) + 1e-8)  # SSIM term

        if self.dynamic_flag:
            # Use learned weighting
            in_loss = [int_loss, grad_loss]
            loss = self.mtl2(in_loss)
            return loss
        else:
            # Use fixed weighting
            in_loss = [int_loss, grad_loss, ssim_loss]
            loss = self.mtl3(in_loss)
            return loss

    @staticmethod
    def gradient_loss(logits, labels):
        """Compute gradient RMSE between predictions and targets"""
        # Compute gradients for predictions and targets
        drec_dx, drec_dy = derivation(logits)
        dimgs_dx, dimgs_dy = derivation(labels)

        # Compute RMSE for each gradient direction
        loss_x = RMSE()(drec_dx, dimgs_dx)
        loss_y = RMSE()(drec_dy, dimgs_dy)

        return loss_x + loss_y  # Combined gradient loss


if __name__ == "__main__":
    """Test case for GradientRMSE"""
    # Create test data
    prediction = torch.arange(1, 2 * 3 * 4 * 5 + 1.0).reshape(2, 3, 4, 5)
    prediction.requires_grad_(True)
    labels = torch.full((2, 3, 4, 5), 2.0)

    # Compute loss
    loss_fn = GradientRMSE()
    loss = loss_fn(prediction, labels)
    print(f"GradientRMSE loss: {loss}")

if __name__ == "__main__":
    """Test case for MTLWeightedLoss"""
    # Create test data
    prediction = torch.arange(1, 2 * 3 * 4 * 5 + 1.0).reshape(2, 3, 4, 5)
    prediction.requires_grad_(True)
    labels = torch.full((2, 3, 4, 5), 2.0)

    # Compute multi-task loss
    loss_fn = MTLWeightedLoss(num_losses=2)
    loss = loss_fn([F.mse_loss(prediction, labels), F.mse_loss(prediction, labels)])
    print(f"MTLWeightedLoss: {loss}")