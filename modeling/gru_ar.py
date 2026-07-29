"""Teacher-forced, observation-space GRU autoregressive baseline."""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUAR(nn.Module):
    """Eight-layer unidirectional GRU forecaster for continuous observations."""

    def __init__(
        self,
        n_vars: int = 16,
        hidden_dim: int = 173,
        num_layers: int = 2,
        dropout: float = 0.05,
        context_length: int = 90,
        forecast_length: int = 90,
    ) -> None:
        super().__init__()
        if n_vars != 16:
            raise ValueError("GRU_AR is defined for exactly 16 observation channels")
        if context_length <= 0 or forecast_length <= 0:
            raise ValueError("context_length and forecast_length must be positive")

        self.n_vars = int(n_vars)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout_p = float(dropout)
        self.context_length = int(context_length)
        self.forecast_length = int(forecast_length)

        self.gru = nn.GRU(
            input_size=self.n_vars,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_p,
            bidirectional=False,
        )
        self.top_hidden_norm = nn.LayerNorm(self.hidden_dim)
        self.readout = nn.Linear(self.hidden_dim, self.n_vars)

    def _validate_context(self, context: torch.Tensor) -> None:
        if context.ndim != 3:
            raise ValueError(
                f"context must have shape (B,{self.context_length},{self.n_vars}), "
                f"got {tuple(context.shape)}"
            )
        if context.shape[1:] != (self.context_length, self.n_vars):
            raise ValueError(
                f"Expected context shape (B,{self.context_length},{self.n_vars}), "
                f"got {tuple(context.shape)}"
            )

    def _encode_context(self, context: torch.Tensor) -> torch.Tensor:
        self._validate_context(context)
        _, hidden = self.gru(context)
        return hidden

    def _readout_hidden(self, top_hidden: torch.Tensor) -> torch.Tensor:
        return self.readout(self.top_hidden_norm(top_hidden))

    def forward_teacher_forced(
        self, context: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Predict with a one-step-shifted ground-truth recurrent input.

        Prediction zero is read directly from the encoded context. Prediction
        ``k > 0`` is read after consuming ``target[:, k - 1]``. Consequently,
        no target sample can contribute to its own prediction.
        """
        self._validate_context(context)
        if target.ndim != 3 or target.shape[0] != context.shape[0]:
            raise ValueError("target must be 3D and match the context batch size")
        if target.shape[1:] != (self.forecast_length, self.n_vars):
            raise ValueError(
                f"Expected target shape (B,{self.forecast_length},{self.n_vars}), "
                f"got {tuple(target.shape)}"
            )
        if target.device != context.device:
            raise ValueError("context and target must be on the same device")

        hidden = self._encode_context(context)
        first = self._readout_hidden(hidden[-1]).unsqueeze(1)
        if self.forecast_length == 1:
            return first

        # One batched recurrent call is equivalent to stepping through
        # target[:, 0], ..., target[:, T-2] in order.
        shifted_outputs, _ = self.gru(target[:, :-1, :], hidden)
        remaining = self._readout_hidden(shifted_outputs)
        return torch.cat([first, remaining], dim=1)

    def forecast_autoregressive(
        self, context: torch.Tensor, horizon: int | None = None
    ) -> torch.Tensor:
        """Forecast one sample at a time using only self-feedback."""
        self._validate_context(context)
        steps = self.forecast_length if horizon is None else int(horizon)
        if steps <= 0:
            raise ValueError("horizon must be positive")

        hidden = self._encode_context(context)
        prediction = self._readout_hidden(hidden[-1]).unsqueeze(1)
        predictions = [prediction]

        for _ in range(1, steps):
            recurrent_output, hidden = self.gru(prediction, hidden)
            prediction = self._readout_hidden(recurrent_output)
            predictions.append(prediction)

        return torch.cat(predictions, dim=1)

    def forecast_blockwise(
        self,
        context: torch.Tensor,
        horizon: int = 720,
        block_size: int | None = None,
    ) -> torch.Tensor:
        """Forecast by re-encoding each generated block as the next context."""
        self._validate_context(context)
        total_steps = int(horizon)
        block_steps = self.forecast_length if block_size is None else int(block_size)
        if total_steps <= 0:
            raise ValueError("horizon must be positive")
        if block_steps != self.context_length:
            raise ValueError(
                "The blockwise protocol requires block_size == context_length "
                f"({self.context_length})"
            )

        current_context = context
        blocks = []
        generated = 0
        while generated < total_steps:
            steps = min(block_steps, total_steps - generated)
            block = self.forecast_autoregressive(current_context, horizon=steps)
            blocks.append(block)
            generated += steps
            if generated < total_steps:
                if steps != self.context_length:
                    raise ValueError(
                        "A partial block can only occur at the end of a rollout"
                    )
                current_context = block

        return torch.cat(blocks, dim=1)

    # Compatibility aliases for existing repository entry-point conventions.
    def forward_teacher_forcing(
        self, context: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self.forward_teacher_forced(context, target)

    def forward_autoregressive(
        self, context: torch.Tensor, block_offset: int = 0
    ) -> torch.Tensor:
        _ = block_offset
        return self.forecast_autoregressive(context, self.forecast_length)

    def forward_autoregressive_kvcache(
        self, context: torch.Tensor, block_offset: int = 0
    ) -> torch.Tensor:
        _ = block_offset
        return self.forecast_autoregressive(context, self.forecast_length)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def create_gru_ar(
    n_vars: int = 16,
    hidden_dim: int = 173,
    num_layers: int = 2,
    dropout: float = 0.05,
    context_length: int = 90,
    forecast_length: int = 90,
    device: str | torch.device = "cpu",
) -> GRUAR:
    return GRUAR(
        n_vars=n_vars,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        context_length=context_length,
        forecast_length=forecast_length,
    ).to(device)
