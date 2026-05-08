"""
Baseline DeepONet surrogate model (pure PyTorch).

Architecture: unstacked DeepONet with dot-product output.

Branch net:
- Input: ramp_control(t), shape (T_ctrl,) = (120,) for the constant-inflow MVP
- Encodes the input function pair into a latent vector of width p

Trunk net:
- Input: query coordinate (x, t) normalized to [0, 1]^2, shape (2,)
- Encodes the query location into a latent vector of width p

Output:
- Inner product of branch and trunk outputs → scalar density ρ(x, t)
- Shape: (N_query,) for a batch of query points

Reference: Lu et al., Nature Machine Intelligence 2021.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BranchNet(nn.Module):
    """Encodes the ramp-control input function into a latent vector of width p."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """
        Args:
            input_dim: T_ctrl (= 120 for the constant-inflow MVP).
            hidden_dim: Width of hidden layers.
            output_dim: Latent width p (must match TrunkNet output_dim).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, input_dim)
        Returns:
            shape (batch, output_dim)
        """
        return self.net(x)


class TrunkNet(nn.Module):
    """Encodes query coordinates (x, t) → latent vector of width p."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """
        Args:
            input_dim: 2 (x and t, both normalized to [0, 1]).
            hidden_dim: Width of hidden layers.
            output_dim: Latent width p (must match BranchNet output_dim).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: shape (batch, N_query, input_dim)
        Returns:
            shape (batch, N_query, output_dim)
        """
        batch, n_query, input_dim = y.shape
        flat = y.reshape(batch * n_query, input_dim)
        out = self.net(flat)
        return out.reshape(batch, n_query, -1)


class DeepONet(nn.Module):
    """Unstacked DeepONet: density = sum_k branch_k * trunk_k + bias."""

    def __init__(self, branch_net: BranchNet, trunk_net: TrunkNet) -> None:
        super().__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self, branch_input: torch.Tensor, trunk_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            branch_input: shape (batch, T_ctrl)
            trunk_input:  shape (batch, N_query, 2)
        Returns:
            density predictions, shape (batch, N_query)
        """
        branch = self.branch_net(branch_input)
        trunk = self.trunk_net(trunk_input)
        return torch.einsum("bp,bnp->bn", branch, trunk) + self.bias
