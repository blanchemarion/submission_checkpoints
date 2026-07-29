"""Conditional Deep Markov latent state-space model for widefield forecasts."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalDeepMarkovSSM(nn.Module):
    """Context-conditioned DMM with diagonal Gaussian latent dynamics."""

    SUPPORTED_LATENT_DIMS = (4, 8, 12)

    def __init__(
        self,
        n_vars: int = 16,
        latent_dim: int = 8,
        encoder_hidden_dim: int = 64,
        posterior_hidden_dim: int = 64,
        transition_hidden_dim: int = 32,
        decoder_hidden_dims: tuple[int, int] = (32, 32),
        context_length: int = 90,
        forecast_length: int = 90,
        spectral_radius: float = 0.99,
        min_scale: float = 1e-4,
        min_log_variance: float = -12.0,
        max_log_variance: float = 8.0,
    ) -> None:
        super().__init__()
        if n_vars != 16:
            raise ValueError("cDMM_SSM requires exactly 16 observation channels")
        if latent_dim not in self.SUPPORTED_LATENT_DIMS:
            raise ValueError(
                f"latent_dim must be one of {self.SUPPORTED_LATENT_DIMS}, got {latent_dim}"
            )
        if encoder_hidden_dim != posterior_hidden_dim:
            raise ValueError(
                "encoder_hidden_dim and posterior_hidden_dim must match so the "
                "context state can initialize the posterior GRU"
            )
        if tuple(decoder_hidden_dims) != (32, 32):
            raise ValueError("The decoder architecture must be latent->32->32->16")
        if context_length <= 0 or forecast_length <= 0:
            raise ValueError("context_length and forecast_length must be positive")
        if not 0.0 < spectral_radius < 1.0:
            raise ValueError("spectral_radius must lie strictly between zero and one")

        self.n_vars = int(n_vars)
        self.latent_dim = int(latent_dim)
        self.encoder_hidden_dim = int(encoder_hidden_dim)
        self.posterior_hidden_dim = int(posterior_hidden_dim)
        self.transition_hidden_dim = int(transition_hidden_dim)
        self.decoder_hidden_dims = tuple(int(value) for value in decoder_hidden_dims)
        self.context_length = int(context_length)
        self.forecast_length = int(forecast_length)
        self.spectral_radius = float(spectral_radius)
        self.min_scale = float(min_scale)
        self.min_log_variance = float(min_log_variance)
        self.max_log_variance = float(max_log_variance)

        self.context_encoder = nn.GRU(
            input_size=self.n_vars,
            hidden_size=self.encoder_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.initial_latent_head = nn.Linear(
            self.encoder_hidden_dim, 2 * self.latent_dim
        )

        self.posterior_gru = nn.GRU(
            input_size=self.n_vars,
            hidden_size=self.posterior_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.posterior_mlp = nn.Sequential(
            nn.Linear(
                self.posterior_hidden_dim + self.latent_dim,
                self.posterior_hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(self.posterior_hidden_dim, 2 * self.latent_dim),
        )

        self.transition_matrix_raw = nn.Parameter(
            torch.empty(self.latent_dim, self.latent_dim)
        )
        nn.init.orthogonal_(self.transition_matrix_raw)
        self.transition_gate = nn.Linear(self.latent_dim, self.latent_dim)
        self.transition_nonlinear_1 = nn.Linear(
            self.latent_dim, self.transition_hidden_dim
        )
        self.transition_nonlinear_2 = nn.Linear(
            self.transition_hidden_dim, self.latent_dim
        )
        self.initial_transition_scale = 0.1
        raw_scale = math.log(
            math.expm1(
                max(self.initial_transition_scale - self.min_scale, 1e-6)
            )
        )
        self.transition_raw_scale = nn.Parameter(
            torch.full((self.latent_dim,), raw_scale)
        )

        decoder_dim_1, decoder_dim_2 = self.decoder_hidden_dims
        self.decoder_mean = nn.Sequential(
            nn.Linear(self.latent_dim, decoder_dim_1),
            nn.Tanh(),
            nn.Linear(decoder_dim_1, decoder_dim_2),
            nn.Tanh(),
            nn.Linear(decoder_dim_2, self.n_vars),
        )
        self.emission_log_variance = nn.Parameter(torch.zeros(self.n_vars))
        self._initialize_stable_start()

    def _validate_context(self, context: torch.Tensor) -> None:
        if context.ndim != 3:
            raise ValueError(
                f"context must have shape (B,{self.context_length},{self.n_vars})"
            )
        if context.shape[1:] != (self.context_length, self.n_vars):
            raise ValueError(
                f"Expected context shape (B,{self.context_length},{self.n_vars}), "
                f"got {tuple(context.shape)}"
            )

    def _validate_target(self, context: torch.Tensor, target: torch.Tensor) -> None:
        if target.ndim != 3 or target.shape[0] != context.shape[0]:
            raise ValueError("target must be 3D and match the context batch size")
        if target.shape[1:] != (self.forecast_length, self.n_vars):
            raise ValueError(
                f"Expected target shape (B,{self.forecast_length},{self.n_vars}), "
                f"got {tuple(target.shape)}"
            )
        if context.device != target.device:
            raise ValueError("context and target must be on the same device")

    def _clamp_log_variance(self, value: torch.Tensor) -> torch.Tensor:
        return value.clamp(self.min_log_variance, self.max_log_variance)
    
    def _initialize_stable_start(self) -> None:
        """Start with q(z_k) approximately equal to p(z_k | z_{k-1})."""
        matched_log_variance = 2.0 * math.log(
            self.initial_transition_scale
        )

        posterior_head = self.posterior_mlp[-1]
        if not isinstance(posterior_head, nn.Linear):
            raise TypeError("Final posterior layer must be nn.Linear")

        with torch.no_grad():
            # Context-conditioned z0 initially has std ≈ 0.1.
            self.initial_latent_head.weight[
                self.latent_dim :, :
            ].zero_()
            self.initial_latent_head.bias[
                self.latent_dim :
            ].fill_(matched_log_variance)

            # The posterior initially predicts:
            # delta_mean = 0 and std ≈ 0.1.
            posterior_head.weight.zero_()
            posterior_head.bias[: self.latent_dim].zero_()
            posterior_head.bias[
                self.latent_dim :
            ].fill_(matched_log_variance)

            # Initially favour the stable linear transition:
            # sigmoid(-4) ≈ 0.018.
            self.transition_gate.weight.zero_()
            self.transition_gate.bias.fill_(-4.0)
            
    @staticmethod
    def _diagonal_gaussian_kl(
        posterior_mean: torch.Tensor,
        posterior_log_variance: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_log_variance: torch.Tensor,
    ) -> torch.Tensor:
        """Elementwise KL[q || p] for diagonal Gaussian distributions."""
        return 0.5 * (
            prior_log_variance
            - posterior_log_variance
            + (
                torch.exp(posterior_log_variance)
                + (posterior_mean - prior_mean).square()
            )
            * torch.exp(-prior_log_variance)
            - 1.0
        )

    @staticmethod
    def _normal_sample(
        mean: torch.Tensor,
        log_variance: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=generator,
        )
        return mean + torch.exp(0.5 * log_variance) * noise

    @staticmethod
    def _make_generator(
        reference: torch.Tensor, seed: int | None
    ) -> torch.Generator | None:
        if seed is None:
            return None
        generator = torch.Generator(device=reference.device)
        generator.manual_seed(int(seed))
        return generator

    def encode_context(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return context state and conditional initial latent parameters."""
        self._validate_context(context)
        _, hidden = self.context_encoder(context)
        context_state = hidden[-1]
        mean, log_variance = self.initial_latent_head(context_state).chunk(2, dim=-1)
        return context_state, mean, self._clamp_log_variance(log_variance)

    def stable_transition_matrix(self) -> torch.Tensor:
        """Return the effective linear dynamics matrix with norm 0.99."""
        raw_float = self.transition_matrix_raw.float()
        norm = torch.linalg.matrix_norm(raw_float, ord=2).clamp_min(1e-8)
        normalized = raw_float * (self.spectral_radius / norm)
        return normalized.to(dtype=self.transition_matrix_raw.dtype)

    def transition_distribution(
        self, previous_latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        linear_state = F.linear(previous_latent, self.stable_transition_matrix())
        gate = torch.sigmoid(self.transition_gate(previous_latent))
        nonlinear_state = torch.tanh(
            self.transition_nonlinear_2(
                torch.tanh(self.transition_nonlinear_1(previous_latent))
            )
        )
        mean = (1.0 - gate) * linear_state + gate * nonlinear_state
        scale = F.softplus(self.transition_raw_scale) + self.min_scale
        log_variance = 2.0 * torch.log(scale)
        log_variance = self._clamp_log_variance(log_variance)
        return mean, log_variance.expand_as(mean)

    def emission_distribution(
        self, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.decoder_mean(latent)
        log_variance = self._clamp_log_variance(self.emission_log_variance)
        return mean, log_variance.expand_as(mean)

    def posterior_rollout(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        *,
        sample: bool = True,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the causal target-prefix posterior used only during training."""
        self._validate_context(context)
        self._validate_target(context, target)
        context_state, initial_mean, initial_log_variance = self.encode_context(context)
        previous_latent = (
            self._normal_sample(initial_mean, initial_log_variance, generator)
            if sample
            else initial_mean
        )

        # posterior_state[:, k] has observed exactly target[:, :k+1].
        posterior_states, _ = self.posterior_gru(
            target, context_state.unsqueeze(0)
        )

        latents = []
        posterior_means = []
        posterior_log_variances = []
        prior_means = []
        prior_log_variances = []
        emission_means = []
        emission_log_variances = []

        for step in range(target.shape[1]):
            prior_mean, prior_log_variance = self.transition_distribution(
                previous_latent
            )
            posterior_parameters = self.posterior_mlp(
                torch.cat([posterior_states[:, step, :], previous_latent], dim=-1)
            )
            
            posterior_delta_mean, posterior_log_variance = (
                posterior_parameters.chunk(2, dim=-1)
            )

            # Residual posterior: q starts equal to the transition prior,
            # but can learn target-informed corrections.
            posterior_mean = prior_mean + posterior_delta_mean
            
            posterior_log_variance = self._clamp_log_variance(
                posterior_log_variance
            )
            latent = (
                self._normal_sample(
                    posterior_mean, posterior_log_variance, generator
                )
                if sample
                else posterior_mean
            )
            emission_mean, emission_log_variance = self.emission_distribution(latent)

            prior_means.append(prior_mean)
            prior_log_variances.append(prior_log_variance)
            posterior_means.append(posterior_mean)
            posterior_log_variances.append(posterior_log_variance)
            latents.append(latent)
            emission_means.append(emission_mean)
            emission_log_variances.append(emission_log_variance)
            previous_latent = latent

        return {
            "initial_mean": initial_mean,
            "initial_log_variance": initial_log_variance,
            "posterior_states": posterior_states,
            "posterior_means": torch.stack(posterior_means, dim=1),
            "posterior_log_variances": torch.stack(
                posterior_log_variances, dim=1
            ),
            "prior_means": torch.stack(prior_means, dim=1),
            "prior_log_variances": torch.stack(prior_log_variances, dim=1),
            "latents": torch.stack(latents, dim=1),
            "emission_means": torch.stack(emission_means, dim=1),
            "emission_log_variances": torch.stack(
                emission_log_variances, dim=1
            ),
        }
        
    def latent_overshooting_kl(
        self,
        rollout: dict[str, torch.Tensor],
        *,
        horizons: tuple[int, ...],
        free_bits: float,
        num_anchors: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Match posterior latents to priors obtained after several autonomous steps.

        Posterior anchors and targets are detached so that this term specifically
        trains the transition model.
        """
        if num_anchors <= 0:
            raise ValueError("num_anchors must be positive")

        posterior_latents = rollout["latents"]
        posterior_means = rollout["posterior_means"]
        posterior_log_variances = rollout["posterior_log_variances"]

        sequence_length = posterior_latents.shape[1]
        clean_horizons = tuple(sorted(set(int(h) for h in horizons)))

        if not clean_horizons:
            zero = posterior_latents.new_zeros((), dtype=torch.float32)
            return zero, zero

        for horizon in clean_horizons:
            if horizon <= 1:
                raise ValueError(
                    "Overshooting horizons must be greater than 1; "
                    "the ordinary ELBO already contains the one-step KL"
                )
            if horizon >= sequence_length:
                raise ValueError(
                    f"Overshooting horizon {horizon} must be smaller than "
                    f"the target length {sequence_length}"
                )

        raw_terms = []
        constrained_terms = []

        for horizon in clean_horizons:
            valid_start_count = sequence_length - horizon
            anchor_count = min(num_anchors, valid_start_count)

            # Equally spaced anchors avoid adding another random sampling source.
            start_indices = torch.linspace(
                0,
                valid_start_count - 1,
                steps=anchor_count,
                device=posterior_latents.device,
            ).round().long().unique(sorted=True)

            endpoint_indices = start_indices + horizon

            # Shape: batch × anchors × latent dimension.
            predicted_latent = posterior_latents.index_select(
                1, start_indices
            ).detach()

            target_mean = posterior_means.index_select(
                1, endpoint_indices
            ).detach().float()

            target_log_variance = posterior_log_variances.index_select(
                1, endpoint_indices
            ).detach().float()

            # Autonomously propagate the transition for `horizon` steps.
            for rollout_step in range(horizon):
                predicted_mean, predicted_log_variance = (
                    self.transition_distribution(predicted_latent)
                )

                # Sample intermediate states. At the endpoint we retain the
                # predicted Gaussian parameters for the KL calculation.
                if rollout_step < horizon - 1:
                    predicted_latent = self._normal_sample(
                        predicted_mean,
                        predicted_log_variance,
                        generator,
                    )

            kl_per_dimension = self._diagonal_gaussian_kl(
                target_mean,
                target_log_variance,
                predicted_mean.float(),
                predicted_log_variance.float(),
            )

            # Convert the anchor average to a T-step-equivalent total so this has
            # approximately the same scale as the existing one-step KL.
            raw_horizon_kl = (
                kl_per_dimension.sum(dim=-1).mean()
                * float(sequence_length)
            )

            constrained_horizon_kl = (
                kl_per_dimension
                .clamp_min(float(free_bits))
                .sum(dim=-1)
                .mean()
                * float(sequence_length)
            )

            raw_terms.append(raw_horizon_kl)
            constrained_terms.append(constrained_horizon_kl)

        return (
            torch.stack(raw_terms).mean(),
            torch.stack(constrained_terms).mean(),
        )
        
    def conditional_elbo(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        *,
        beta: float,
        free_bits: float = 0.05,
        overshooting_horizons: tuple[int, ...] = (),
        overshooting_weight: float = 0.0,
        overshooting_num_anchors: int = 4,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return negative conditional ELBO and detached diagnostic terms."""
        if overshooting_weight < 0.0:
            raise ValueError("overshooting_weight must be nonnegative")
        if beta < 0.0:
            raise ValueError("beta must be nonnegative")
        if free_bits < 0.0:
            raise ValueError("free_bits must be nonnegative")
        rollout = self.posterior_rollout(
            context, target, sample=True, generator=generator
        )
        emission_mean = rollout["emission_means"].float()
        emission_log_variance = rollout["emission_log_variances"].float()
        target_float = target.float()

        reconstruction_per_value = 0.5 * (
            math.log(2.0 * math.pi)
            + emission_log_variance
            + (target_float - emission_mean).square()
            * torch.exp(-emission_log_variance)
        )
        reconstruction_nll = reconstruction_per_value.sum(dim=(1, 2)).mean()

        posterior_mean = rollout["posterior_means"].float()
        posterior_log_variance = rollout["posterior_log_variances"].float()
        prior_mean = rollout["prior_means"].float()
        prior_log_variance = rollout["prior_log_variances"].float()
        kl_per_dimension = self._diagonal_gaussian_kl(
            posterior_mean,
            posterior_log_variance,
            prior_mean,
            prior_log_variance,
        )

        raw_kl = kl_per_dimension.sum(dim=(1, 2)).mean()

        free_bits_kl = kl_per_dimension.clamp_min(float(free_bits))
        constrained_kl = free_bits_kl.sum(dim=(1, 2)).mean()
        
        if overshooting_weight > 0.0 and overshooting_horizons:
            overshooting_raw_kl, overshooting_constrained_kl = (
                self.latent_overshooting_kl(
                    rollout,
                    horizons=overshooting_horizons,
                    free_bits=free_bits,
                    num_anchors=overshooting_num_anchors,
                    generator=generator,
                )
            )
        else:
            overshooting_raw_kl = reconstruction_nll.new_zeros(())
            overshooting_constrained_kl = reconstruction_nll.new_zeros(())

        kl_regularizer_total = (
            constrained_kl
            + float(overshooting_weight) * overshooting_constrained_kl
        )

        loss = reconstruction_nll + float(beta) * kl_regularizer_total
        latent_steps = target.shape[1] * self.latent_dim

        diagnostics = {
            "loss": loss.detach(),
            "reconstruction_nll": reconstruction_nll.detach(),
            "kl_raw": raw_kl.detach(),
            "kl_free_bits": constrained_kl.detach(),
            "kl_per_latent_step": (
                raw_kl.detach() / float(latent_steps)
            ),
            "posterior_reconstruction_mse": F.mse_loss(
                emission_mean,
                target_float,
            ).detach(),
            "posterior_reconstruction_mae": F.l1_loss(
                emission_mean,
                target_float,
            ).detach(),
            "transition_std": torch.exp(
                0.5 * prior_log_variance
            ).mean().detach(),
            "emission_std": torch.exp(
                0.5 * emission_log_variance
            ).mean().detach(),
            "beta": loss.detach().new_tensor(float(beta)),
            "overshooting_kl_raw": overshooting_raw_kl.detach(),
            "overshooting_kl_free_bits":
                overshooting_constrained_kl.detach(),
            "kl_regularizer_total": kl_regularizer_total.detach(),
            "overshooting_weight": loss.detach().new_tensor(
                float(overshooting_weight)
            ),
        }

        return loss, diagnostics

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        *,
        beta: float,
        free_bits: float = 0.05,
        overshooting_horizons: tuple[int, ...] = (),
        overshooting_weight: float = 0.0,
        overshooting_num_anchors: int = 4,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.conditional_elbo(
            context,
            target,
            beta=beta,
            free_bits=free_bits,
            overshooting_horizons=overshooting_horizons,
            overshooting_weight=overshooting_weight,
            overshooting_num_anchors=overshooting_num_anchors,
        )

    def forecast_mean(
        self, context: torch.Tensor, horizon: int | None = None
    ) -> torch.Tensor:
        """Context-only deterministic generative-mean forecast."""
        self._validate_context(context)
        steps = self.forecast_length if horizon is None else int(horizon)
        if steps <= 0:
            raise ValueError("horizon must be positive")
        _, latent, _ = self.encode_context(context)
        predictions = []
        for _ in range(steps):
            latent, _ = self.transition_distribution(latent)
            emission_mean, _ = self.emission_distribution(latent)
            predictions.append(emission_mean)
        return torch.stack(predictions, dim=1)

    def forecast_sample(
        self,
        context: torch.Tensor,
        horizon: int | None = None,
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        sample_emission: bool = True,
    ) -> torch.Tensor:
        """One context-only stochastic latent and emission trajectory."""
        self._validate_context(context)
        if seed is not None and generator is not None:
            raise ValueError("Provide either seed or generator, not both")
        steps = self.forecast_length if horizon is None else int(horizon)
        if steps <= 0:
            raise ValueError("horizon must be positive")
        active_generator = (
            generator
            if generator is not None
            else self._make_generator(context, seed)
        )
        _, initial_mean, initial_log_variance = self.encode_context(context)
        latent = self._normal_sample(
            initial_mean, initial_log_variance, active_generator
        )
        predictions = []
        for _ in range(steps):
            transition_mean, transition_log_variance = (
                self.transition_distribution(latent)
            )
            latent = self._normal_sample(
                transition_mean, transition_log_variance, active_generator
            )
            emission_mean, emission_log_variance = self.emission_distribution(latent)
            observation = (
                self._normal_sample(
                    emission_mean,
                    emission_log_variance,
                    active_generator,
                )
                if sample_emission
                else emission_mean
            )
            predictions.append(observation)
        return torch.stack(predictions, dim=1)

    def forecast_samples(
        self,
        context: torch.Tensor,
        horizon: int | None = None,
        *,
        num_samples: int = 6,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        sample_emission: bool = True,
    ) -> torch.Tensor:
        """Return trajectories with axes (sample, batch, time, region)."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if seed is not None and generator is not None:
            raise ValueError("Provide either seed or generator, not both")
        active_generator = (
            generator
            if generator is not None
            else self._make_generator(context, seed)
        )
        batch = context.shape[0]
        expanded_context = (
            context.unsqueeze(0)
            .expand(num_samples, -1, -1, -1)
            .reshape(num_samples * batch, self.context_length, self.n_vars)
        )

        trajectories = self.forecast_sample(
            expanded_context,
            horizon=horizon,
            generator=active_generator,
            sample_emission=sample_emission,
        )
        return trajectories.reshape(
            num_samples,
            batch,
            trajectories.shape[1],
            self.n_vars,
        )

    def forecast_mean_blockwise(
        self, context: torch.Tensor, horizon: int = 720
    ) -> torch.Tensor:
        """Re-encode each deterministic 90-step generated block."""
        self._validate_context(context)
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.forecast_length != self.context_length:
            raise ValueError("Blockwise forecasting requires T_out == T_in")
        current_context = context
        blocks = []
        generated = 0
        while generated < horizon:
            block_steps = min(self.forecast_length, horizon - generated)
            block = self.forecast_mean(current_context, block_steps)
            blocks.append(block)
            generated += block_steps
            if generated < horizon:
                if block_steps != self.context_length:
                    raise ValueError("A partial block can only be the final block")
                current_context = block
        return torch.cat(blocks, dim=1)

    def forecast_samples_blockwise(
        self,
        context: torch.Tensor,
        horizon: int = 720,
        *,
        num_samples: int = 6,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        sample_emission: bool = True,
    ) -> torch.Tensor:
        """Propagate each sampled trajectory's own history between blocks."""
        self._validate_context(context)
        if num_samples <= 0 or horizon <= 0:
            raise ValueError("num_samples and horizon must be positive")
        if seed is not None and generator is not None:
            raise ValueError("Provide either seed or generator, not both")
        if self.forecast_length != self.context_length:
            raise ValueError("Blockwise forecasting requires T_out == T_in")
        active_generator = (
            generator
            if generator is not None
            else self._make_generator(context, seed)
        )
        batch = context.shape[0]
        current_context = (
            context.unsqueeze(0)
            .expand(num_samples, -1, -1, -1)
            .reshape(num_samples * batch, self.context_length, self.n_vars)
        )
        blocks = []
        generated = 0
        while generated < horizon:
            block_steps = min(self.forecast_length, horizon - generated)
            block = self.forecast_sample(
                current_context,
                block_steps,
                generator=active_generator,
                sample_emission=sample_emission,
            )
            blocks.append(block)
            generated += block_steps
            if generated < horizon:
                if block_steps != self.context_length:
                    raise ValueError("A partial block can only be the final block")
                current_context = block
        trajectories = torch.cat(blocks, dim=1)
        return trajectories.reshape(
            num_samples, batch, horizon, self.n_vars
        )
        
    def forecast_predictive_mean(
        self,
        context: torch.Tensor,
        horizon: int | None = None,
        *,
        num_samples: int = 32,
        seed: int = 101,
    ) -> torch.Tensor:
        """Monte-Carlo estimate of E[x_future | context]."""
        trajectories = self.forecast_samples(
            context,
            horizon=horizon,
            num_samples=num_samples,
            seed=seed,
            sample_emission=False,
        )
        return trajectories.mean(dim=0)


    def forecast_predictive_mean_blockwise(
        self,
        context: torch.Tensor,
        horizon: int = 720,
        *,
        num_samples: int = 32,
        seed: int = 101,
    ) -> torch.Tensor:
        trajectories = self.forecast_samples_blockwise(
            context,
            horizon=horizon,
            num_samples=num_samples,
            seed=seed,
            sample_emission=False,
        )
        return trajectories.mean(dim=0)

    def transition_spectral_norm(self) -> float:
        with torch.no_grad():
            return float(
                torch.linalg.matrix_norm(
                    self.stable_transition_matrix().float(), ord=2
                ).item()
            )

    def count_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


def create_cdmm_ssm(
    n_vars: int = 16,
    latent_dim: int = 8,
    encoder_hidden_dim: int = 64,
    posterior_hidden_dim: int = 64,
    transition_hidden_dim: int = 32,
    decoder_hidden_dims: tuple[int, int] = (32, 32),
    context_length: int = 90,
    forecast_length: int = 90,
    spectral_radius: float = 0.99,
    min_scale: float = 1e-4,
    min_log_variance: float = -12.0,
    max_log_variance: float = 8.0,
    device: str | torch.device = "cpu",
) -> ConditionalDeepMarkovSSM:
    return ConditionalDeepMarkovSSM(
        n_vars=n_vars,
        latent_dim=latent_dim,
        encoder_hidden_dim=encoder_hidden_dim,
        posterior_hidden_dim=posterior_hidden_dim,
        transition_hidden_dim=transition_hidden_dim,
        decoder_hidden_dims=decoder_hidden_dims,
        context_length=context_length,
        forecast_length=forecast_length,
        spectral_radius=spectral_radius,
        min_scale=min_scale,
        min_log_variance=min_log_variance,
        max_log_variance=max_log_variance,
    ).to(device)

