from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PPOConfig:
    n_steps: int = 128
    n_epochs: int = 4
    minibatch_size: int = 32
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    total_timesteps: int = 100_000
    log_interval: int = 10
    hidden_sizes: tuple[int, ...] = (64, 64)
    tanh_squash: bool = False
    fd_epsilon: float = 1e-4
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if self.n_epochs <= 0:
            raise ValueError("n_epochs must be positive.")
        if self.minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive.")
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive.")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive.")
        if self.fd_epsilon <= 0.0:
            raise ValueError("fd_epsilon must be positive.")
        if not 0.0 <= self.clip_eps:
            raise ValueError("clip_eps must be non-negative.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1.")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be between 0 and 1.")
