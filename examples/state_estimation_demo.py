"""Humanoid EKF state-estimation demo."""

from __future__ import annotations

import numpy as np

from optisim.estimation import build_estimator


def main() -> None:
    pipeline = build_estimator(n_joints=31)

    print("state estimation rollout:")
    for step in range(20):
        encoder_readings = 0.05 * np.sin(np.linspace(0.0, 1.5, 31) + step * 0.1)
        imu = {
            "accel": np.array([0.08, 0.0, -9.81], dtype=np.float64),
            "gyro": np.array([0.0, 0.0, 0.015], dtype=np.float64),
        }
        visual_odometry = np.array([0.005 * step, 0.0, 0.92], dtype=np.float64)
        contacts = [
            np.array([0.02, 0.10, 0.0], dtype=np.float64),
            np.array([0.02, -0.10, 0.0], dtype=np.float64),
        ]
        state = pipeline.process_sensor_bundle(
            imu=imu,
            encoders=encoder_readings,
            contacts=contacts,
            visual_odometry=visual_odometry,
            dt=0.01,
        )
        print(
            f"step {step:02d}: position={np.round(state.position, 4)} "
            f"uncertainty={pipeline.uncertainty_norm:.6f}"
        )

    history = pipeline.get_history()
    start_error = np.linalg.norm(history[0].position - np.array([0.0, 0.0, 0.92]))
    end_error = np.linalg.norm(history[-1].position - np.array([0.095, 0.0, 0.92]))
    print(f"\nconvergence proxy: initial_error={start_error:.4f} final_error={end_error:.4f}")


if __name__ == "__main__":
    main()
