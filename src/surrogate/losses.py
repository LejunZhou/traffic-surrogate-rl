"""
Loss functions for DeepONet training.

Phase 1: MSE loss on z-score normalized density predictions.

Extension points (Phase 2):
- PDE residual loss (LWR equation)
- Hybrid data + physics loss with tunable weight
Do NOT add physics-informed losses here unless explicitly requested.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# TODO: implement loss functions


def mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean squared error between predicted and target density values.

    Args:
        predictions: shape (batch, N_query) — model output
        targets:     shape (batch, N_query) — ground truth (normalized)

    Returns:
        Scalar MSE loss.
    """
    raise NotImplementedError
