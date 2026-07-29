"""
Simple linear autoregressive baseline (non-transformer).

This model predicts one next timestep from the last T_in timesteps by
flattening (T_in, n_vars) -> (T_in * n_vars) and applying a linear layer.
It supports:
  - teacher-forced multi-step training windows
  - closed-loop autoregressive rollout (uses its own predictions)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearARBaseline(nn.Module):
    def __init__(self, n_vars: int, T_in: int, T_out: int, bias: bool = True):
        super().__init__()
        self.n_vars = int(n_vars)
        self.T_in = int(T_in)
        self.T_out = int(T_out)
        self.linear = nn.Linear(self.T_in * self.n_vars, self.n_vars, bias=bias)

    def _predict_next(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: (B, T_in, n_vars)
        Returns:
            next_step: (B, 1, n_vars)
        """
        if context.ndim != 3:
            raise ValueError(f"context must be 3D (B,T_in,n_vars), got {context.shape}")
        if context.shape[1] != self.T_in or context.shape[2] != self.n_vars:
            raise ValueError(
                f"Expected context shape (*,{self.T_in},{self.n_vars}), got {tuple(context.shape)}"
            )
        bsz = context.shape[0]
        flat = context.reshape(bsz, self.T_in * self.n_vars)
        pred = self.linear(flat)
        return pred.unsqueeze(1)

    def forward_teacher_forcing(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forced rollout:
        for step k, predict y_k from GT window [x, y_<k].

        Args:
            x: (B, T_in, n_vars)
            y: (B, T_out, n_vars)
        Returns:
            preds: (B, T_out, n_vars)
        """
        if y.ndim != 3 or y.shape[1] != self.T_out or y.shape[2] != self.n_vars:
            raise ValueError(
                f"Expected y shape (*,{self.T_out},{self.n_vars}), got {tuple(y.shape)}"
            )

        full_gt = torch.cat([x, y], dim=1)  # (B, T_in + T_out, n_vars)
        preds = []
        for k in range(self.T_out):
            ctx = full_gt[:, k : k + self.T_in, :]
            pred_k = self._predict_next(ctx)
            preds.append(pred_k)
        return torch.cat(preds, dim=1)

    def forward_autoregressive(self, x: torch.Tensor, block_offset: int = 0) -> torch.Tensor:
        """
        Closed-loop rollout:
        each new prediction is appended and used for subsequent steps.
        """
        _ = block_offset  # kept for interface compatibility
        current = x
        preds = []
        for _step in range(self.T_out):
            ctx = current[:, -self.T_in :, :]
            pred = self._predict_next(ctx)
            preds.append(pred)
            current = torch.cat([current, pred], dim=1)
        return torch.cat(preds, dim=1)

    def forward_autoregressive_kvcache(self, x: torch.Tensor, block_offset: int = 0) -> torch.Tensor:
        """
        Alias for compatibility with code paths expecting transformer KV-cache API.
        """
        return self.forward_autoregressive(x, block_offset=block_offset)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_var_baseline(
    n_vars: int = 16,
    T_in: int = 90,
    T_out: int = 90,
    bias: bool = True,
    device: str | torch.device = "cpu",
) -> LinearARBaseline:
    return LinearARBaseline(n_vars=n_vars, T_in=T_in, T_out=T_out, bias=bias).to(device)
