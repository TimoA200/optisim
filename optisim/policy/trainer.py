from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optisim.lfd import Demonstration
from optisim.policy.dataset import NormStats, PolicyDataset
from optisim.policy.network import PolicyNetwork, build_policy_network


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(slots=True)
class BCConfig:
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 100
    hidden_dims: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    val_fraction: float = 0.2
    patience: int = 20
    verbose: bool = False


@dataclass(slots=True)
class TrainingResult:
    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int
    final_train_loss: float
    final_val_loss: float
    converged: bool
    epochs_trained: int


class BehavioralCloningTrainer:
    def __init__(self, config: BCConfig | None = None) -> None:
        self.config = config or BCConfig()

    def train(
        self,
        dataset: PolicyDataset,
        network: PolicyNetwork | None = None,
    ) -> tuple[PolicyNetwork, TrainingResult]:
        if len(dataset) == 0:
            raise ValueError("Training requires a non-empty dataset.")

        config = self.config
        train_fraction = 1.0 - float(config.val_fraction)
        train_fraction = min(max(train_fraction, 0.0), 1.0)
        train_data, val_data = dataset.split(train_fraction=train_fraction, seed=config.seed)

        if network is None:
            network = build_policy_network(
                input_dim=dataset.obs_dim,
                hidden_dims=config.hidden_dims,
                output_dim=dataset.act_dim,
                activation=config.activation,
            )

        rng = np.random.default_rng(config.seed)
        params = network.parameters()
        m = [{"W": np.zeros_like(layer["W"]), "b": np.zeros_like(layer["b"])} for layer in params]
        v = [{"W": np.zeros_like(layer["W"]), "b": np.zeros_like(layer["b"])} for layer in params]
        train_losses: list[float] = []
        val_losses: list[float] = []
        best_epoch = 0
        best_val_loss = float("inf")
        best_params = [{"W": layer["W"].copy(), "b": layer["b"].copy()} for layer in params]
        steps_without_improvement = 0
        converged = False

        batch_size = max(1, int(config.batch_size))
        total_steps = 0

        for epoch in range(int(config.epochs)):
            indices = np.arange(len(train_data))
            rng.shuffle(indices)
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start : batch_start + batch_size]
                batch_obs, batch_actions = train_data.get_batch(batch_indices)
                predictions, cache = network._forward_with_cache(batch_obs)
                _, grad_output = self._loss_and_grad(predictions, batch_actions)
                grads = network.backward(cache, grad_output)
                total_steps += 1
                updated_params = self._adam_step(
                    network.parameters(),
                    grads,
                    m,
                    v,
                    total_steps,
                    config.learning_rate,
                    wd=config.weight_decay,
                )
                network.set_parameters(updated_params)

            train_predictions = network.predict(train_data.observations)
            train_loss, _ = self._loss_and_grad(train_predictions, train_data.actions)
            val_predictions = network.predict(val_data.observations)
            val_loss, _ = self._loss_and_grad(val_predictions, val_data.actions)
            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))

            if config.verbose:
                print(
                    f"epoch={epoch + 1} train_loss={train_loss:.6f} val_loss={val_loss:.6f}",
                )

            if val_loss < best_val_loss - 1e-10:
                best_val_loss = float(val_loss)
                best_epoch = epoch
                best_params = [{"W": layer["W"].copy(), "b": layer["b"].copy()} for layer in network.parameters()]
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
                if steps_without_improvement >= int(config.patience):
                    converged = True
                    break

        network.set_parameters(best_params)
        result = TrainingResult(
            train_losses=train_losses,
            val_losses=val_losses,
            best_epoch=best_epoch,
            final_train_loss=float(train_losses[-1]),
            final_val_loss=float(val_losses[-1]),
            converged=converged,
            epochs_trained=len(train_losses),
        )
        return network, result

    def _loss_and_grad(self, predictions: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
        preds = np.asarray(predictions, dtype=float)
        truth = np.asarray(targets, dtype=float)
        if preds.ndim == 1:
            preds = preds[None, :]
        if truth.ndim == 1:
            truth = truth[None, :]
        if preds.shape != truth.shape:
            raise ValueError("Predictions and targets must have identical shape.")

        joint_pred = preds[:, :-1]
        joint_true = truth[:, :-1]
        gripper_logits = preds[:, -1:]
        gripper_true = truth[:, -1:]

        batch_size = preds.shape[0]
        joint_diff = joint_pred - joint_true
        joint_mse = np.mean(joint_diff * joint_diff) if joint_diff.size else 0.0
        joint_grad = (2.0 / max(joint_diff.size, 1)) * joint_diff if joint_diff.size else np.zeros_like(joint_diff)

        gripper_prob = _sigmoid(gripper_logits)
        gripper_bce = -np.mean(
            gripper_true * np.log(gripper_prob + 1e-8)
            + (1.0 - gripper_true) * np.log(1.0 - gripper_prob + 1e-8)
        )
        gripper_grad = (gripper_prob - gripper_true) / batch_size

        total_loss = float(joint_mse + gripper_bce)
        grad_output = np.concatenate([joint_grad, gripper_grad], axis=1)
        return total_loss, grad_output

    def _adam_step(
        self,
        params: list[dict[str, np.ndarray]],
        grads: list[dict[str, np.ndarray]],
        m: list[dict[str, np.ndarray]],
        v: list[dict[str, np.ndarray]],
        t: int,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        wd: float = 0.0,
    ) -> list[dict[str, np.ndarray]]:
        updated: list[dict[str, np.ndarray]] = []
        clip_value = float(self.config.grad_clip)

        for param, grad, m_state, v_state in zip(params, grads, m, v):
            grad_w = np.asarray(grad["dW"], dtype=float)
            grad_b = np.asarray(grad["db"], dtype=float)

            if clip_value > 0.0:
                grad_w = np.clip(grad_w, -clip_value, clip_value)
                grad_b = np.clip(grad_b, -clip_value, clip_value)

            if wd:
                grad_w = grad_w + wd * param["W"]
                grad_b = grad_b + wd * param["b"]

            m_state["W"] = beta1 * m_state["W"] + (1.0 - beta1) * grad_w
            m_state["b"] = beta1 * m_state["b"] + (1.0 - beta1) * grad_b
            v_state["W"] = beta2 * v_state["W"] + (1.0 - beta2) * (grad_w * grad_w)
            v_state["b"] = beta2 * v_state["b"] + (1.0 - beta2) * (grad_b * grad_b)

            m_hat_w = m_state["W"] / (1.0 - beta1**t)
            m_hat_b = m_state["b"] / (1.0 - beta1**t)
            v_hat_w = v_state["W"] / (1.0 - beta2**t)
            v_hat_b = v_state["b"] / (1.0 - beta2**t)

            updated.append(
                {
                    "W": param["W"] - lr * m_hat_w / (np.sqrt(v_hat_w) + eps),
                    "b": param["b"] - lr * m_hat_b / (np.sqrt(v_hat_b) + eps),
                }
            )

        return updated


def train_policy(
    demonstrations: list[Demonstration],
    config: BCConfig | None = None,
) -> tuple[PolicyNetwork, NormStats, TrainingResult]:
    dataset = PolicyDataset.from_demonstrations(demonstrations)
    norm_stats = dataset.normalize()
    trainer = BehavioralCloningTrainer(config=config)
    network, result = trainer.train(dataset)
    return network, norm_stats, result
