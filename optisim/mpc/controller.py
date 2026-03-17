"""Model predictive control utilities for humanoid balance and locomotion."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.linalg import cho_factor, cho_solve
except ModuleNotFoundError:  # pragma: no cover - exercised implicitly in minimal environments
    def cho_factor(matrix: FloatArray, lower: bool = True, check_finite: bool = False) -> tuple[FloatArray, bool]:
        del lower, check_finite
        cholesky = np.linalg.cholesky(matrix)
        return cholesky, True

    def cho_solve(
        factorized: tuple[FloatArray, bool],
        rhs: FloatArray,
        check_finite: bool = False,
    ) -> FloatArray:
        del check_finite
        factor, _ = factorized
        intermediate = np.linalg.solve(factor, rhs)
        return np.linalg.solve(factor.T, intermediate)

from optisim.robot import RobotModel

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class MPCConfig:
    """Configuration for linearized humanoid CoM MPC."""

    horizon_steps: int = 20
    dt: float = 0.05
    com_height_z: float = 0.9
    Q: float = 10.0
    R: float = 1.0
    max_zmp_offset: float = 0.1
    max_com_velocity: float = 1.5


@dataclass(slots=True)
class MPCSolution:
    """Output of an MPC solve."""

    optimal_states: FloatArray
    optimal_inputs: FloatArray
    cost: float
    solve_time_ms: float
    horizon: int


@dataclass(slots=True)
class FootstepPlan:
    """Simple left/right footstep sequence with timing."""

    left_foot_positions: list[FloatArray]
    right_foot_positions: list[FloatArray]
    timing: list[float]


class FootstepPlanner:
    """Generate a simple alternating humanoid walking pattern."""

    def __init__(
        self,
        step_length: float = 0.18,
        step_duration: float = 0.4,
        stance_width: float = 0.22,
        left_foot_start: Iterable[float] = (0.0, 0.11, 0.0),
        right_foot_start: Iterable[float] = (0.0, -0.11, 0.0),
    ) -> None:
        self.step_length = float(step_length)
        self.step_duration = float(step_duration)
        self.stance_width = float(stance_width)
        self.left_foot_start = np.asarray(left_foot_start, dtype=np.float64)
        self.right_foot_start = np.asarray(right_foot_start, dtype=np.float64)

    def plan_walk(self, direction: Iterable[float], steps: int = 4) -> FootstepPlan:
        """Plan alternating footsteps in the requested direction."""

        if steps < 1:
            raise ValueError("steps must be at least 1")
        direction_vec = np.asarray(direction, dtype=np.float64)
        if direction_vec.shape != (3,):
            raise ValueError("direction must be a 3-vector")
        norm = float(np.linalg.norm(direction_vec[:2]))
        if norm < 1e-9:
            direction_unit = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            direction_unit = direction_vec / max(float(np.linalg.norm(direction_vec)), 1e-9)
        lateral = np.asarray([-direction_unit[1], direction_unit[0], 0.0], dtype=np.float64)

        left = self.left_foot_start.copy()
        right = self.right_foot_start.copy()
        left_positions: list[FloatArray] = [left.copy()]
        right_positions: list[FloatArray] = [right.copy()]
        timing = [0.0]

        for step_index in range(steps):
            progress = self.step_length * float(step_index + 1)
            anchor = direction_unit * progress
            left_target = anchor + lateral * (self.stance_width * 0.5)
            right_target = anchor - lateral * (self.stance_width * 0.5)
            left_target[2] = self.left_foot_start[2]
            right_target[2] = self.right_foot_start[2]
            if step_index % 2 == 0:
                left = left_target
            else:
                right = right_target
            left_positions.append(left.copy())
            right_positions.append(right.copy())
            timing.append((step_index + 1) * self.step_duration)

        return FootstepPlan(
            left_foot_positions=left_positions,
            right_foot_positions=right_positions,
            timing=timing,
        )


class LinearMPC:
    """Receding-horizon MPC for a linear inverted pendulum model."""

    def __init__(
        self,
        config: MPCConfig | None = None,
        control_horizon: int | None = None,
        max_iterations: int = 120,
        tolerance: float = 1e-6,
    ) -> None:
        self.config = config or MPCConfig()
        self.horizon = self.config.horizon_steps
        self.control_horizon = int(control_horizon or self.horizon)
        if self.horizon < 1:
            raise ValueError("horizon_steps must be at least 1")
        if self.control_horizon < 1:
            raise ValueError("control_horizon must be at least 1")
        self.control_horizon = min(self.control_horizon, self.horizon)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.state_dim = 5
        self.input_dim = 2
        self.A, self.B = self._build_dynamics(self.config.dt, self.config.com_height_z)
        self.state_weights = np.diag(
            [
                self.config.Q,
                self.config.Q * 0.5,
                self.config.Q,
                self.config.Q * 0.5,
                self.config.Q * 0.25,
            ]
        ).astype(np.float64)
        self.terminal_weights = self.state_weights * 2.0
        self.input_weights = np.eye(self.input_dim, dtype=np.float64) * self.config.R
        self._warm_start = np.zeros((self.control_horizon, self.input_dim), dtype=np.float64)

    @staticmethod
    def _build_dynamics(dt: float, com_height_z: float) -> tuple[FloatArray, FloatArray]:
        gravity = 9.81
        omega_sq = gravity / max(com_height_z, 1e-6)
        omega = float(np.sqrt(omega_sq))
        c = float(np.cosh(omega * dt))
        s = float(np.sinh(omega * dt))
        ad_axis = np.asarray([[c, s / omega], [omega * s, c]], dtype=np.float64)
        bd_axis = np.asarray([[1.0 - c], [-omega * s]], dtype=np.float64)

        a_matrix = np.eye(5, dtype=np.float64)
        b_matrix = np.zeros((5, 2), dtype=np.float64)
        a_matrix[:2, :2] = ad_axis
        a_matrix[2:4, 2:4] = ad_axis
        b_matrix[:2, 0:1] = bd_axis
        b_matrix[2:4, 1:2] = bd_axis
        return a_matrix, b_matrix

    def warm_start(self, inputs: FloatArray | None) -> None:
        """Seed the next optimization with a previous control sequence."""

        if inputs is None:
            self._warm_start = np.zeros((self.control_horizon, self.input_dim), dtype=np.float64)
            return
        array = np.asarray(inputs, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.input_dim:
            raise ValueError("warm start inputs must have shape (steps, 2)")
        truncated = np.zeros((self.control_horizon, self.input_dim), dtype=np.float64)
        usable = min(self.control_horizon, array.shape[0])
        truncated[:usable] = array[:usable]
        if usable > 0 and usable < self.control_horizon:
            truncated[usable:] = truncated[usable - 1]
        self._warm_start = truncated

    def solve(
        self,
        current_state: Iterable[float],
        target_state: Iterable[float] | None = None,
        zmp_reference: FloatArray | None = None,
    ) -> MPCSolution:
        """Solve the condensed finite-horizon QP."""

        start = perf_counter()
        x0 = np.asarray(current_state, dtype=np.float64)
        if x0.shape != (self.state_dim,):
            raise ValueError("current_state must be a 5-vector")
        x0 = x0.copy()
        x0[4] = self.config.com_height_z

        if target_state is None:
            target = x0.copy()
            target[1] = float(np.clip(target[1], -self.config.max_com_velocity, self.config.max_com_velocity))
            target[3] = float(np.clip(target[3], -self.config.max_com_velocity, self.config.max_com_velocity))
        else:
            target = np.asarray(target_state, dtype=np.float64)
            if target.shape != (self.state_dim,):
                raise ValueError("target_state must be a 5-vector")
            target = target.copy()
            target[4] = self.config.com_height_z
            target[1] = float(np.clip(target[1], -self.config.max_com_velocity, self.config.max_com_velocity))
            target[3] = float(np.clip(target[3], -self.config.max_com_velocity, self.config.max_com_velocity))

        zmp_traj = self._normalize_zmp_reference(zmp_reference, x0, target)
        s_x, s_u_full = self._prediction_matrices()
        move_block = self._move_blocking_matrix()
        s_u = s_u_full @ move_block
        reference_states = np.tile(target, self.horizon)
        q_blocks = [self.state_weights.copy() for _ in range(self.horizon)]
        q_blocks[-1] = self.terminal_weights.copy()
        q_bar = _block_diag(q_blocks)
        delta_matrix, delta_offset = self._input_delta_matrix(zmp_traj)
        r_bar = _block_diag([self.input_weights.copy() for _ in range(self.horizon)])

        state_error = s_x @ x0 - reference_states
        hessian = 2.0 * (
            s_u.T @ q_bar @ s_u
            + delta_matrix.T @ r_bar @ delta_matrix
            + np.eye(self.control_horizon * self.input_dim, dtype=np.float64) * 1e-8
        )
        gradient = 2.0 * (
            s_u.T @ q_bar @ state_error
            - delta_matrix.T @ r_bar @ delta_offset
        )

        lower_bounds = (zmp_traj[: self.control_horizon] - self.config.max_zmp_offset).reshape(-1)
        upper_bounds = (zmp_traj[: self.control_horizon] + self.config.max_zmp_offset).reshape(-1)

        decision = self._projected_qp_solve(hessian, gradient, lower_bounds, upper_bounds)
        expanded_inputs = (move_block @ decision).reshape(self.horizon, self.input_dim)
        predicted_states = (s_x @ x0 + s_u @ decision).reshape(self.horizon, self.state_dim)
        predicted_states[:, 1] = np.clip(
            predicted_states[:, 1], -self.config.max_com_velocity, self.config.max_com_velocity
        )
        predicted_states[:, 3] = np.clip(
            predicted_states[:, 3], -self.config.max_com_velocity, self.config.max_com_velocity
        )
        predicted_states[:, 4] = self.config.com_height_z
        cost = float(
            (predicted_states.reshape(-1) - reference_states).T
            @ q_bar
            @ (predicted_states.reshape(-1) - reference_states)
            + (delta_matrix @ decision - delta_offset).T @ r_bar @ (delta_matrix @ decision - delta_offset)
        )
        solve_time_ms = (perf_counter() - start) * 1000.0
        self._warm_start = self._shift_inputs(expanded_inputs)
        return MPCSolution(
            optimal_states=predicted_states,
            optimal_inputs=expanded_inputs,
            cost=cost,
            solve_time_ms=solve_time_ms,
            horizon=self.horizon,
        )

    def _projected_qp_solve(
        self,
        hessian: FloatArray,
        gradient: FloatArray,
        lower_bounds: FloatArray,
        upper_bounds: FloatArray,
    ) -> FloatArray:
        initial = np.clip(self._warm_start.reshape(-1), lower_bounds, upper_bounds)
        factor = cho_factor(hessian, lower=True, check_finite=False)
        unconstrained = cho_solve(factor, -gradient, check_finite=False)
        iterate = np.clip(unconstrained, lower_bounds, upper_bounds)
        if initial.shape == iterate.shape:
            iterate = 0.5 * iterate + 0.5 * initial
        lipschitz = float(np.max(np.linalg.eigvalsh(hessian)))
        step_size = 1.0 / max(lipschitz, 1e-6)
        for _ in range(self.max_iterations):
            grad = hessian @ iterate + gradient
            candidate = np.clip(iterate - step_size * grad, lower_bounds, upper_bounds)
            if np.linalg.norm(candidate - iterate) < self.tolerance:
                iterate = candidate
                break
            iterate = candidate
        return iterate

    def _prediction_matrices(self) -> tuple[FloatArray, FloatArray]:
        powers = [np.linalg.matrix_power(self.A, index + 1) for index in range(self.horizon)]
        s_x = np.zeros((self.horizon * self.state_dim, self.state_dim), dtype=np.float64)
        s_u = np.zeros((self.horizon * self.state_dim, self.horizon * self.input_dim), dtype=np.float64)
        for row in range(self.horizon):
            row_slice = slice(row * self.state_dim, (row + 1) * self.state_dim)
            s_x[row_slice, :] = powers[row]
            for col in range(row + 1):
                col_slice = slice(col * self.input_dim, (col + 1) * self.input_dim)
                transition = np.linalg.matrix_power(self.A, row - col) @ self.B
                s_u[row_slice, col_slice] = transition
        return s_x, s_u

    def _move_blocking_matrix(self) -> FloatArray:
        matrix = np.zeros((self.horizon * self.input_dim, self.control_horizon * self.input_dim), dtype=np.float64)
        for step in range(self.horizon):
            active = min(step, self.control_horizon - 1)
            row_slice = slice(step * self.input_dim, (step + 1) * self.input_dim)
            col_slice = slice(active * self.input_dim, (active + 1) * self.input_dim)
            matrix[row_slice, col_slice] = np.eye(self.input_dim, dtype=np.float64)
        return matrix

    def _input_delta_matrix(self, zmp_reference: FloatArray) -> tuple[FloatArray, FloatArray]:
        move_block = self._move_blocking_matrix()
        full_delta = np.zeros((self.horizon * self.input_dim, self.horizon * self.input_dim), dtype=np.float64)
        delta_offset = np.zeros(self.horizon * self.input_dim, dtype=np.float64)
        for step in range(self.horizon):
            row = slice(step * self.input_dim, (step + 1) * self.input_dim)
            full_delta[row, row] = np.eye(self.input_dim, dtype=np.float64)
            if step > 0:
                prev = slice((step - 1) * self.input_dim, step * self.input_dim)
                full_delta[row, prev] = -np.eye(self.input_dim, dtype=np.float64)
        delta_offset[: self.input_dim] = zmp_reference[0]
        return full_delta @ move_block, delta_offset

    def _normalize_zmp_reference(
        self,
        zmp_reference: FloatArray | None,
        current_state: FloatArray,
        target_state: FloatArray,
    ) -> FloatArray:
        if zmp_reference is None:
            midpoint = np.asarray(
                [
                    0.5 * (current_state[0] + target_state[0]),
                    0.5 * (current_state[2] + target_state[2]),
                ],
                dtype=np.float64,
            )
            return np.tile(midpoint, (self.horizon, 1))
        zmp = np.asarray(zmp_reference, dtype=np.float64)
        if zmp.ndim != 2 or zmp.shape[1] != self.input_dim:
            raise ValueError("zmp_reference must have shape (steps, 2)")
        if zmp.shape[0] >= self.horizon:
            return zmp[: self.horizon].copy()
        padded = np.zeros((self.horizon, self.input_dim), dtype=np.float64)
        padded[: zmp.shape[0]] = zmp
        padded[zmp.shape[0] :] = zmp[-1]
        return padded

    def _shift_inputs(self, inputs: FloatArray) -> FloatArray:
        shifted = np.zeros((self.control_horizon, self.input_dim), dtype=np.float64)
        usable = min(self.control_horizon - 1, inputs.shape[0] - 1)
        if usable > 0:
            shifted[:usable] = inputs[1 : usable + 1]
        shifted[usable:] = inputs[min(inputs.shape[0] - 1, usable)]
        return shifted


class HumanoidMPC:
    """Higher-level wrapper around ``LinearMPC`` for humanoid CoM control."""

    def __init__(
        self,
        config: MPCConfig | None = None,
        robot: RobotModel | None = None,
        footstep_planner: FootstepPlanner | None = None,
        control_horizon: int | None = None,
    ) -> None:
        self.config = config or MPCConfig()
        self.robot = robot
        self.linear_mpc = LinearMPC(self.config, control_horizon=control_horizon)
        self.footstep_planner = footstep_planner or FootstepPlanner()
        self.last_solution: MPCSolution | None = None
        self._last_state = self._infer_state_from_robot(robot)

    def warm_start(self, initial_state: Iterable[float] | None = None) -> None:
        """Initialize or reset the controller state."""

        if initial_state is not None:
            state = np.asarray(initial_state, dtype=np.float64)
            if state.shape != (5,):
                raise ValueError("initial_state must be a 5-vector")
            state = state.copy()
            state[4] = self.config.com_height_z
            self._last_state = state
        elif self.robot is not None:
            self._last_state = self._infer_state_from_robot(self.robot)
        else:
            self._last_state = np.asarray([0.0, 0.0, 0.0, 0.0, self.config.com_height_z], dtype=np.float64)
        self.linear_mpc.warm_start(None)
        self.last_solution = None

    def step(
        self,
        current_state: Iterable[float] | None = None,
        target_position: Iterable[float] | None = None,
        target_velocity: Iterable[float] | None = None,
        footstep_plan: FootstepPlan | None = None,
    ) -> MPCSolution:
        """Advance one MPC step for balance or walking."""

        if current_state is None:
            state = self._last_state.copy()
        else:
            state = np.asarray(current_state, dtype=np.float64)
            if state.shape != (5,):
                raise ValueError("current_state must be a 5-vector")
            state = state.copy()
            state[4] = self.config.com_height_z
        target = state.copy()
        if target_position is not None:
            position = np.asarray(target_position, dtype=np.float64)
            if position.shape != (2,):
                raise ValueError("target_position must be a 2-vector")
            target[0] = position[0]
            target[2] = position[1]
        if target_velocity is not None:
            velocity = np.asarray(target_velocity, dtype=np.float64)
            if velocity.shape != (2,):
                raise ValueError("target_velocity must be a 2-vector")
            target[1] = float(np.clip(velocity[0], -self.config.max_com_velocity, self.config.max_com_velocity))
            target[3] = float(np.clip(velocity[1], -self.config.max_com_velocity, self.config.max_com_velocity))
        zmp_reference = self._zmp_reference_from_plan(footstep_plan)
        solution = self.linear_mpc.solve(state, target_state=target, zmp_reference=zmp_reference)
        self.last_solution = solution
        next_state = solution.optimal_states[0].copy()
        next_state[4] = self.config.com_height_z
        self._last_state = next_state
        return solution

    def _zmp_reference_from_plan(self, footstep_plan: FootstepPlan | None) -> FloatArray | None:
        plan = footstep_plan
        if plan is None and self.robot is not None:
            left = self.robot.end_effector_pose("left_foot").position
            right = self.robot.end_effector_pose("right_foot").position
            midpoint = np.asarray([0.5 * (left[0] + right[0]), 0.5 * (left[1] + right[1])], dtype=np.float64)
            return np.tile(midpoint, (self.config.horizon_steps, 1))
        if plan is None:
            return None
        support_centers: list[FloatArray] = []
        max_index = max(len(plan.left_foot_positions), len(plan.right_foot_positions))
        for index in range(max_index):
            left = plan.left_foot_positions[min(index, len(plan.left_foot_positions) - 1)]
            right = plan.right_foot_positions[min(index, len(plan.right_foot_positions) - 1)]
            support_centers.append(np.asarray([(left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5], dtype=np.float64))
        if not support_centers:
            return None
        return self.linear_mpc._normalize_zmp_reference(np.vstack(support_centers), self._last_state, self._last_state)

    def _infer_state_from_robot(self, robot: RobotModel | None) -> FloatArray:
        if robot is None or "left_foot" not in robot.end_effectors or "right_foot" not in robot.end_effectors:
            return np.asarray([0.0, 0.0, 0.0, 0.0, self.config.com_height_z], dtype=np.float64)
        left = robot.end_effector_pose("left_foot").position
        right = robot.end_effector_pose("right_foot").position
        return np.asarray(
            [(left[0] + right[0]) * 0.5, 0.0, (left[1] + right[1]) * 0.5, 0.0, self.config.com_height_z],
            dtype=np.float64,
        )


def build_humanoid_mpc(config: MPCConfig | None = None) -> HumanoidMPC:
    """Build a humanoid MPC controller with default components."""

    return HumanoidMPC(config=config)


def _block_diag(blocks: list[FloatArray]) -> FloatArray:
    total_rows = sum(block.shape[0] for block in blocks)
    total_cols = sum(block.shape[1] for block in blocks)
    matrix = np.zeros((total_rows, total_cols), dtype=np.float64)
    row_offset = 0
    col_offset = 0
    for block in blocks:
        rows, cols = block.shape
        matrix[row_offset : row_offset + rows, col_offset : col_offset + cols] = block
        row_offset += rows
        col_offset += cols
    return matrix


__all__ = [
    "MPCConfig",
    "MPCSolution",
    "LinearMPC",
    "HumanoidMPC",
    "FootstepPlan",
    "FootstepPlanner",
    "build_humanoid_mpc",
]
