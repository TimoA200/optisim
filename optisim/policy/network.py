from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(float)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _sigmoid_grad(x: np.ndarray) -> np.ndarray:
    sigma = _sigmoid(x)
    return sigma * (1.0 - sigma)


def _tanh_grad(x: np.ndarray) -> np.ndarray:
    tanh_x = np.tanh(x)
    return 1.0 - tanh_x * tanh_x


class PolicyNetwork:
    """Pure NumPy multilayer perceptron for behavioral cloning."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        output_dim: int,
        activation: str = "relu",
    ) -> None:
        if int(input_dim) <= 0:
            raise ValueError("input_dim must be positive.")
        if int(output_dim) <= 0:
            raise ValueError("output_dim must be positive.")
        activation_name = str(activation).lower()
        if activation_name not in {"relu", "tanh", "sigmoid"}:
            raise ValueError("activation must be one of: relu, tanh, sigmoid.")

        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.output_dim = int(output_dim)
        self.activation = activation_name
        self.layers: list[dict[str, np.ndarray]] = []

        dims = [self.input_dim, *self.hidden_dims, self.output_dim]
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            scale = np.sqrt(2.0 / fan_in) if self.activation == "relu" else np.sqrt(1.0 / fan_in)
            self.layers.append(
                {
                    "W": np.random.randn(fan_in, fan_out).astype(float) * scale,
                    "b": np.zeros(fan_out, dtype=float),
                }
            )

    def _activation_forward(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return _relu(x)
        if self.activation == "tanh":
            return np.tanh(x)
        return _sigmoid(x)

    def _activation_backward(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return _relu_grad(x)
        if self.activation == "tanh":
            return _tanh_grad(x)
        return _sigmoid_grad(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        output, _ = self._forward_with_cache(x)
        return output

    def _forward_with_cache(self, x: np.ndarray) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
        activations = np.asarray(x, dtype=float)
        single_input = activations.ndim == 1
        if single_input:
            activations = activations[None, :]
        if activations.ndim != 2 or activations.shape[1] != self.input_dim:
            raise ValueError("Input must have shape (batch, input_dim) or (input_dim,).")

        cache: list[dict[str, np.ndarray]] = []
        current = activations
        last_index = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            z = current @ layer["W"] + layer["b"]
            if index == last_index:
                a = z
            else:
                a = self._activation_forward(z)
            cache.append({"input": current, "pre_activation": z, "post_activation": a})
            current = a

        output = current[0] if single_input else current
        return output, cache

    def backward(self, cache: list[dict[str, np.ndarray]], grad_output: np.ndarray) -> list[dict[str, np.ndarray]]:
        gradients: list[dict[str, np.ndarray]] = []
        grad = np.asarray(grad_output, dtype=float)
        if grad.ndim == 1:
            grad = grad[None, :]
        if not cache:
            raise ValueError("Cache must not be empty.")

        for layer_index in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[layer_index]
            layer_cache = cache[layer_index]
            grad_z = grad
            dW = layer_cache["input"].T @ grad_z
            db = np.sum(grad_z, axis=0)
            gradients.append({"dW": dW, "db": db})
            grad = grad_z @ layer["W"].T
            if layer_index > 0:
                prev_pre_activation = cache[layer_index - 1]["pre_activation"]
                grad = grad * self._activation_backward(prev_pre_activation)

        gradients.reverse()
        return gradients

    def parameters(self) -> list[dict[str, np.ndarray]]:
        return [{"W": layer["W"], "b": layer["b"]} for layer in self.layers]

    def set_parameters(self, params: list[dict[str, np.ndarray]]) -> None:
        if len(params) != len(self.layers):
            raise ValueError("Parameter list length does not match network depth.")
        for layer, param in zip(self.layers, params):
            weight = np.asarray(param["W"], dtype=float)
            bias = np.asarray(param["b"], dtype=float)
            if weight.shape != layer["W"].shape or bias.shape != layer["b"].shape:
                raise ValueError("Parameter shapes do not match the network.")
            layer["W"] = weight.copy()
            layer["b"] = bias.copy()

    def predict(self, x: np.ndarray) -> np.ndarray:
        inputs = np.asarray(x, dtype=float)
        if inputs.ndim == 1:
            return self.forward(inputs[None, :])[0]
        return self.forward(inputs)

    def save(self, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "input_dim": np.array(self.input_dim, dtype=int),
            "output_dim": np.array(self.output_dim, dtype=int),
            "hidden_dims": np.array(self.hidden_dims, dtype=int),
            "activation": np.array(self.activation),
            "num_layers": np.array(len(self.layers), dtype=int),
        }
        for index, layer in enumerate(self.layers):
            payload[f"W_{index}"] = layer["W"]
            payload[f"b_{index}"] = layer["b"]
        np.savez(destination, **payload)

    def load(self, path: str) -> None:
        data = np.load(Path(path), allow_pickle=False)
        loaded_input_dim = int(np.asarray(data["input_dim"]).item())
        loaded_output_dim = int(np.asarray(data["output_dim"]).item())
        loaded_hidden_dims = tuple(int(value) for value in np.asarray(data["hidden_dims"], dtype=int).tolist())
        loaded_activation = str(np.asarray(data["activation"]).item())
        if (
            loaded_input_dim != self.input_dim
            or loaded_output_dim != self.output_dim
            or loaded_hidden_dims != self.hidden_dims
            or loaded_activation != self.activation
        ):
            raise ValueError("Serialized network architecture does not match this PolicyNetwork.")

        loaded_params = []
        for index in range(int(np.asarray(data["num_layers"]).item())):
            loaded_params.append(
                {
                    "W": np.asarray(data[f"W_{index}"], dtype=float),
                    "b": np.asarray(data[f"b_{index}"], dtype=float),
                }
            )
        self.set_parameters(loaded_params)

    def copy(self) -> PolicyNetwork:
        clone = PolicyNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            activation=self.activation,
        )
        clone.set_parameters(deepcopy(self.parameters()))
        return clone

    def num_parameters(self) -> int:
        return int(sum(layer["W"].size + layer["b"].size for layer in self.layers))


def build_policy_network(
    input_dim: int,
    hidden_dims: tuple[int, ...] = (256, 256),
    output_dim: int | None = None,
    activation: str = "relu",
) -> PolicyNetwork:
    if output_dim is None:
        raise ValueError("output_dim must be provided.")
    return PolicyNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation=activation,
    )
