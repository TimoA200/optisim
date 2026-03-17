"""Pure NumPy world-model network and trainer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optisim.worldmodel.state import WorldState


@dataclass(slots=True)
class TransitionSample:
    """One supervised transition for next-state prediction."""

    state: WorldState
    action_vec: np.ndarray
    next_state: WorldState
    reward: float = 0.0
    done: bool = False

    def __post_init__(self) -> None:
        self.action_vec = np.asarray(self.action_vec, dtype=float).reshape(-1)
        self.reward = float(self.reward)
        self.done = bool(self.done)


class WorldModelNet:
    """Small MLP that predicts the next encoded scene state."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        if self.state_dim <= 0 or self.action_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("state_dim, action_dim, and hidden_dim must be positive")
        rng = np.random.default_rng(0)
        input_dim = self.state_dim + self.action_dim
        scale = 0.1
        self.weights = {
            "W1": rng.normal(0.0, scale, size=(input_dim, self.hidden_dim)),
            "b1": np.zeros(self.hidden_dim, dtype=float),
            "W2": rng.normal(0.0, scale, size=(self.hidden_dim, self.hidden_dim)),
            "b2": np.zeros(self.hidden_dim, dtype=float),
            "W3": rng.normal(0.0, scale, size=(self.hidden_dim, self.state_dim)),
            "b3": np.zeros(self.state_dim, dtype=float),
        }

    @property
    def n_params(self) -> int:
        return int(sum(value.size for value in self.weights.values()))

    def _forward(self, x: np.ndarray) -> np.ndarray:
        inputs = np.asarray(x, dtype=float)
        squeeze = inputs.ndim == 1
        if squeeze:
            inputs = inputs[None, :]
        hidden1 = np.tanh(inputs @ self.weights["W1"] + self.weights["b1"])
        hidden2 = np.tanh(hidden1 @ self.weights["W2"] + self.weights["b2"])
        output = hidden2 @ self.weights["W3"] + self.weights["b3"]
        return output[0] if squeeze else output

    def predict(self, state_vec: np.ndarray, action_vec: np.ndarray) -> np.ndarray:
        """Predict the next encoded scene state from the current state and action."""

        state_array = np.asarray(state_vec, dtype=float).reshape(-1)
        action_array = np.asarray(action_vec, dtype=float).reshape(-1)
        x = np.concatenate([state_array, action_array])
        return self._forward(x)


class WorldModelTrainer:
    """Backprop trainer for the world-model MLP."""

    def __init__(self, model: WorldModelNet, lr: float = 0.001) -> None:
        self.model = model
        self.lr = float(lr)

    def train_step(self, samples: list[TransitionSample]) -> float:
        if not samples:
            return 0.0

        x_batch = np.stack(
            [np.concatenate([sample.state.as_vector(), sample.action_vec]).astype(float) for sample in samples],
            axis=0,
        )
        y_batch = np.stack([sample.next_state.as_vector().astype(float) for sample in samples], axis=0)

        w1 = self.model.weights["W1"]
        b1 = self.model.weights["b1"]
        w2 = self.model.weights["W2"]
        b2 = self.model.weights["b2"]
        w3 = self.model.weights["W3"]
        b3 = self.model.weights["b3"]

        z1 = x_batch @ w1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ w2 + b2
        a2 = np.tanh(z2)
        y_pred = a2 @ w3 + b3

        error = y_pred - y_batch
        loss = float(np.mean(error**2))
        batch_size = x_batch.shape[0]
        grad_y = (2.0 / (batch_size * y_batch.shape[1])) * error

        grad_w3 = a2.T @ grad_y
        grad_b3 = np.sum(grad_y, axis=0)
        grad_a2 = grad_y @ w3.T
        grad_z2 = grad_a2 * (1.0 - a2**2)
        grad_w2 = a1.T @ grad_z2
        grad_b2 = np.sum(grad_z2, axis=0)
        grad_a1 = grad_z2 @ w2.T
        grad_z1 = grad_a1 * (1.0 - a1**2)
        grad_w1 = x_batch.T @ grad_z1
        grad_b1 = np.sum(grad_z1, axis=0)

        self.model.weights["W3"] = w3 - self.lr * grad_w3
        self.model.weights["b3"] = b3 - self.lr * grad_b3
        self.model.weights["W2"] = w2 - self.lr * grad_w2
        self.model.weights["b2"] = b2 - self.lr * grad_b2
        self.model.weights["W1"] = w1 - self.lr * grad_w1
        self.model.weights["b1"] = b1 - self.lr * grad_b1
        return loss

    def fit(self, samples: list[TransitionSample], epochs: int = 10) -> list[float]:
        history: list[float] = []
        for _ in range(int(epochs)):
            history.append(self.train_step(samples))
        return history


__all__ = ["TransitionSample", "WorldModelNet", "WorldModelTrainer"]
