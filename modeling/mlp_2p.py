from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP2P(nn.Module):
    """
    Population-conditioned per-neuron MLP for continuous nonnegative 2p traces.

    Input:  context (B, T_in, N)
    Output: pred_next (B, 1, N)
    """

    def __init__(
        self,
        n_neurons: int,
        T_in: int = 90,
        T_out: int = 10,
        d_local: int = 64,
        d_pop: int = 128,
        dropout: float = 0.05,
        use_slope_summary: bool = True,
        decay_init: float = 0.85,
    ) -> None:
        super().__init__()
        if not 0.0 < decay_init < 1.0:
            raise ValueError("decay_init must be in (0, 1)")
        self.n_neurons = int(n_neurons)
        self.T_in = int(T_in)
        self.T_out = int(T_out)
        self.d_local = int(d_local)
        self.d_pop = int(d_pop)
        self.use_slope_summary = bool(use_slope_summary)

        self.temporal_mlp = nn.Sequential(
            nn.Linear(self.T_in, self.d_local),
            nn.GELU(),
            nn.LayerNorm(self.d_local),
            nn.Dropout(dropout),
            nn.Linear(self.d_local, self.d_local),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.neuron_embedding = nn.Parameter(torch.zeros(self.n_neurons, self.d_local))
        nn.init.normal_(self.neuron_embedding, mean=0.0, std=0.02)

        pop_summary_factor = 4 if self.use_slope_summary else 3
        pop_summary_dim = self.n_neurons * pop_summary_factor
        self.population_encoder = nn.Sequential(
            nn.Linear(pop_summary_dim, self.d_pop),
            nn.GELU(),
            nn.LayerNorm(self.d_pop),
            nn.Dropout(dropout),
            nn.Linear(self.d_pop, self.d_pop),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.film_mlp = nn.Linear(self.d_pop, 2 * self.d_local)
        self.pre_head_norm = nn.LayerNorm(self.d_local)

        self.head = nn.Sequential(
            nn.Linear(self.d_local, self.d_local),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_local, self.T_out), #nn.Linear(self.d_local, 1),
        )

        decay_logit = math.log(decay_init / (1.0 - decay_init))
        self.decay_logit = nn.Parameter(torch.full((self.n_neurons,), float(decay_logit)))

    def _build_population_summary(self, context: torch.Tensor) -> torch.Tensor:
        # context: (B, T_in, N)
        last_t = context[:, -1, :]  # (B, N)
        mean_t = context.mean(dim=1)  # (B, N)
        max_t = context.max(dim=1).values  # (B, N)

        if self.use_slope_summary:
            k = min(5, context.shape[1] - 1)
            slope_t = context[:, -1, :] - context[:, -1 - k, :]
            return torch.cat([last_t, mean_t, max_t, slope_t], dim=-1)
        return torch.cat([last_t, mean_t, max_t], dim=-1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 3:
            raise ValueError(f"Expected context shape (B, T, N), got {tuple(context.shape)}")
        bsz, t_in, n_neurons = context.shape
        if n_neurons != self.n_neurons:
            raise ValueError(f"Expected N={self.n_neurons}, got {n_neurons}")
        if t_in != self.T_in:
            raise ValueError(f"Expected T_in={self.T_in}, got {t_in}")

        # Per-neuron temporal forecaster.
        x = context.transpose(1, 2).reshape(bsz * self.n_neurons, self.T_in)  # (B*N, T_in)
        local = self.temporal_mlp(x).reshape(bsz, self.n_neurons, self.d_local)  # (B, N, d_local)
        local = local + self.neuron_embedding.unsqueeze(0)

        # Population context.
        pop_summary = self._build_population_summary(context)
        pop_context = self.population_encoder(pop_summary)  # (B, d_pop)

        # FiLM conditioning.
        gamma_beta = self.film_mlp(pop_context)  # (B, 2*d_local)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        modulated = self.pre_head_norm(local)
        modulated = modulated * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        # Prediction head + learned persistence/decay skip.
        """head_output = self.head(modulated).squeeze(-1)  # (B, N)
        last_value = context[:, -1, :]  # (B, N)
        decay = torch.sigmoid(self.decay_logit).unsqueeze(0)  # (1, N)
        raw_next = decay * last_value + head_output

        # Nonnegative continuous output.
        pred_next = F.softplus(raw_next) - math.log(2.0)
        pred_next = pred_next.clamp_min(0.0)
        return pred_next.unsqueeze(1)  # (B, 1, N)"""

        head_output = self.head(modulated)  # (B, N, T_out)

        last_value = context[:, -1, :]  # (B, N)
        decay = torch.sigmoid(self.decay_logit)  # (N,)

        # Build horizon-wise decay: decay^1, decay^2, ..., decay^T_out
        horizons = torch.arange(
            1,
            self.T_out + 1,
            device=context.device,
            dtype=context.dtype,
        )

        decay_curve = decay.unsqueeze(0) ** horizons.unsqueeze(1)  # (T_out, N)
        decay_skip = last_value.unsqueeze(1) * decay_curve.unsqueeze(0)  # (B, T_out, N)

        # head_output is (B, N, T_out), transpose to (B, T_out, N)
        head_output = head_output.transpose(1, 2)

        raw = decay_skip + head_output

        pred = F.softplus(raw) - math.log(2.0)
        pred = pred.clamp_min(0.0)

        return pred  # (B, T_out, N)
