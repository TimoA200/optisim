from __future__ import annotations

import numpy as np


class PPOOptimizer:
    def __init__(
        self,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.reset()

    def step(self, params: np.ndarray, grads: np.ndarray, lr: float) -> np.ndarray:
        parameters = np.asarray(params, dtype=float)
        gradients = np.asarray(grads, dtype=float)
        if self.m is None or self.v is None or self.m.shape != parameters.shape:
            self.m = np.zeros_like(parameters)
            self.v = np.zeros_like(parameters)
            self.t = 0

        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (gradients * gradients)
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        return parameters - float(lr) * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self) -> None:
        self.m: np.ndarray | None = None
        self.v: np.ndarray | None = None
        self.t = 0
