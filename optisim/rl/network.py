from __future__ import annotations

from pathlib import Path

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(float)


class ActorCritic:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int, ...] | list[int] = (64, 64),
        tanh_squash: bool = False,
        seed: int | None = None,
    ) -> None:
        if obs_dim <= 0 or act_dim <= 0:
            raise ValueError("obs_dim and act_dim must be positive.")
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.tanh_squash = bool(tanh_squash)
        self.rng = np.random.default_rng(seed)

        trunk_dims = [self.obs_dim, *self.hidden_sizes]
        self.trunk: list[dict[str, np.ndarray]] = []
        for fan_in, fan_out in zip(trunk_dims[:-1], trunk_dims[1:]):
            self.trunk.append(
                {
                    "W": self.rng.standard_normal((fan_in, fan_out)).astype(float) * np.sqrt(2.0 / fan_in),
                    "b": np.zeros(fan_out, dtype=float),
                }
            )

        latent_dim = self.hidden_sizes[-1] if self.hidden_sizes else self.obs_dim
        self.actor = {
            "W": self.rng.standard_normal((latent_dim, self.act_dim)).astype(float) * np.sqrt(1.0 / max(latent_dim, 1)),
            "b": np.zeros(self.act_dim, dtype=float),
        }
        self.critic = {
            "W": self.rng.standard_normal((latent_dim, 1)).astype(float) * np.sqrt(1.0 / max(latent_dim, 1)),
            "b": np.zeros(1, dtype=float),
        }
        self.log_std = np.zeros(self.act_dim, dtype=float)

    def _ensure_batch(self, obs: np.ndarray, expected_dim: int) -> tuple[np.ndarray, bool]:
        array = np.asarray(obs, dtype=float)
        single = array.ndim == 1
        if single:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != expected_dim:
            raise ValueError(f"Expected shape (batch, {expected_dim}) or ({expected_dim},).")
        return array, single

    def _forward_with_cache(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        inputs, _ = self._ensure_batch(obs, self.obs_dim)
        current = inputs
        pre_activations: list[np.ndarray] = []
        activations: list[np.ndarray] = [inputs]
        for layer in self.trunk:
            z = current @ layer["W"] + layer["b"]
            current = _relu(z)
            pre_activations.append(z)
            activations.append(current)
        latent = current if self.trunk else inputs
        mean = latent @ self.actor["W"] + self.actor["b"]
        value = latent @ self.critic["W"] + self.critic["b"]
        cache = {
            "pre_activations": pre_activations,
            "activations": activations,
            "latent": latent,
        }
        return mean, np.broadcast_to(self.log_std, mean.shape).copy(), value[:, 0], cache

    def forward(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean, log_std, value, _ = self._forward_with_cache(obs)
        inputs = np.asarray(obs, dtype=float)
        if inputs.ndim == 1:
            return mean[0], log_std[0], value[0]
        return mean, log_std, value

    def _gaussian_log_prob(self, actions: np.ndarray, mean: np.ndarray, log_std: np.ndarray) -> np.ndarray:
        std = np.exp(log_std)
        var = std * std
        return -0.5 * np.sum(((actions - mean) ** 2) / (var + 1e-8) + 2.0 * log_std + np.log(2.0 * np.pi), axis=1)

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        mean, log_std, value = self.forward(obs)
        if mean.ndim != 1:
            raise ValueError("get_action expects a single observation.")
        std = np.exp(log_std)
        raw_action = mean if deterministic else mean + self.rng.normal(size=mean.shape) * std
        action = np.tanh(raw_action) if self.tanh_squash else raw_action
        log_prob = self._gaussian_log_prob(raw_action[None, :], mean[None, :], log_std[None, :])[0]
        if self.tanh_squash:
            log_prob -= float(np.sum(np.log(1.0 - np.clip(action * action, 0.0, 1.0 - 1e-6) + 1e-6)))
        return action.astype(float), float(log_prob), float(value)

    def evaluate_actions(self, obs: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean, log_std, values = self.forward(obs)
        mean_batch, _ = self._ensure_batch(mean, self.act_dim)
        action_batch, _ = self._ensure_batch(actions, self.act_dim)
        log_std_batch = np.broadcast_to(log_std, mean_batch.shape)
        if self.tanh_squash:
            clipped = np.clip(action_batch, -1.0 + 1e-6, 1.0 - 1e-6)
            raw_action = np.arctanh(clipped)
            log_probs = self._gaussian_log_prob(raw_action, mean_batch, log_std_batch)
            log_probs -= np.sum(np.log(1.0 - clipped * clipped + 1e-6), axis=1)
        else:
            log_probs = self._gaussian_log_prob(action_batch, mean_batch, log_std_batch)
        entropy = np.sum(log_std_batch + 0.5 * (1.0 + np.log(2.0 * np.pi)), axis=1)
        return log_probs, np.asarray(values, dtype=float), entropy

    def backward(self, obs: np.ndarray, grad_mean: np.ndarray, grad_value: np.ndarray) -> np.ndarray:
        grad_mean_batch, _ = self._ensure_batch(grad_mean, self.act_dim)
        grad_value_array = np.asarray(grad_value, dtype=float)
        if grad_value_array.ndim == 1:
            grad_value_array = grad_value_array[:, None]
        _, _, _, cache = self._forward_with_cache(obs)
        activations = list(cache["activations"])
        pre_activations = list(cache["pre_activations"])
        latent = np.asarray(cache["latent"], dtype=float)
        batch_size = latent.shape[0]

        grad_actor_W = latent.T @ grad_mean_batch / batch_size
        grad_actor_b = np.sum(grad_mean_batch, axis=0) / batch_size
        grad_critic_W = latent.T @ grad_value_array / batch_size
        grad_critic_b = np.sum(grad_value_array, axis=0) / batch_size
        grad_latent = grad_mean_batch @ self.actor["W"].T + grad_value_array @ self.critic["W"].T

        trunk_grads: list[tuple[np.ndarray, np.ndarray]] = []
        for index in range(len(self.trunk) - 1, -1, -1):
            grad_z = grad_latent * _relu_grad(pre_activations[index])
            grad_W = activations[index].T @ grad_z / batch_size
            grad_b = np.sum(grad_z, axis=0) / batch_size
            trunk_grads.append((grad_W, grad_b))
            grad_latent = grad_z @ self.trunk[index]["W"].T
        trunk_grads.reverse()

        flat_parts: list[np.ndarray] = []
        for grad_W, grad_b in trunk_grads:
            flat_parts.extend([grad_W.ravel(), grad_b.ravel()])
        flat_parts.extend(
            [
                grad_actor_W.ravel(),
                grad_actor_b.ravel(),
                grad_critic_W.ravel(),
                grad_critic_b.ravel(),
                np.zeros_like(self.log_std).ravel(),
            ]
        )
        return np.concatenate(flat_parts).astype(float)

    def get_params(self) -> np.ndarray:
        parts: list[np.ndarray] = []
        for layer in self.trunk:
            parts.extend([layer["W"].ravel(), layer["b"].ravel()])
        parts.extend(
            [
                self.actor["W"].ravel(),
                self.actor["b"].ravel(),
                self.critic["W"].ravel(),
                self.critic["b"].ravel(),
                self.log_std.ravel(),
            ]
        )
        return np.concatenate(parts).astype(float)

    def set_params(self, params: np.ndarray) -> None:
        flat = np.asarray(params, dtype=float).ravel()
        offset = 0

        def take(shape: tuple[int, ...]) -> np.ndarray:
            nonlocal offset
            size = int(np.prod(shape))
            if offset + size > flat.size:
                raise ValueError("Parameter vector is too small.")
            chunk = flat[offset : offset + size].reshape(shape)
            offset += size
            return chunk

        for layer in self.trunk:
            layer["W"] = take(layer["W"].shape).copy()
            layer["b"] = take(layer["b"].shape).copy()
        self.actor["W"] = take(self.actor["W"].shape).copy()
        self.actor["b"] = take(self.actor["b"].shape).copy()
        self.critic["W"] = take(self.critic["W"].shape).copy()
        self.critic["b"] = take(self.critic["b"].shape).copy()
        self.log_std = take(self.log_std.shape).copy()
        if offset != flat.size:
            raise ValueError("Parameter vector has extra values.")

    def save(self, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            destination,
            obs_dim=np.array(self.obs_dim, dtype=int),
            act_dim=np.array(self.act_dim, dtype=int),
            hidden_sizes=np.array(self.hidden_sizes, dtype=int),
            tanh_squash=np.array(int(self.tanh_squash), dtype=int),
            params=self.get_params(),
        )

    @classmethod
    def load(cls, path: str) -> ActorCritic:
        data = np.load(Path(path), allow_pickle=False)
        network = cls(
            obs_dim=int(np.asarray(data["obs_dim"]).item()),
            act_dim=int(np.asarray(data["act_dim"]).item()),
            hidden_sizes=tuple(int(value) for value in np.asarray(data["hidden_sizes"], dtype=int).tolist()),
            tanh_squash=bool(int(np.asarray(data["tanh_squash"]).item())),
        )
        network.set_params(np.asarray(data["params"], dtype=float))
        return network
