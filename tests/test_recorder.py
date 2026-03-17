from __future__ import annotations

from dataclasses import is_dataclass

import numpy as np
import pytest

import optisim
import optisim.recorder as recorder
from optisim import EpisodeRecorder as PublicEpisodeRecorder
from optisim import EpisodeReplay as PublicEpisodeReplay
from optisim import TelemetryFrame as PublicTelemetryFrame
from optisim import TelemetryStats as PublicTelemetryStats
from optisim import recorder as public_recorder_module
from optisim.recorder import EpisodeRecorder, EpisodeReplay, TelemetryFrame, TelemetryStats


def _frame(
    timestamp: float,
    *,
    positions: tuple[float, ...] = (0.0, 1.0),
    velocities: tuple[float, ...] = (0.5, -0.5),
    torques: tuple[float, ...] = (1.0, -2.0),
    base_pose: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    contacts: dict[str, bool] | None = None,
    extras: dict[str, float] | None = None,
) -> TelemetryFrame:
    return TelemetryFrame(
        timestamp=timestamp,
        joint_positions=np.array(positions, dtype=float),
        joint_velocities=np.array(velocities, dtype=float),
        joint_torques=np.array(torques, dtype=float),
        base_pose=np.array(base_pose, dtype=float),
        contact_states=contacts or {"left": True, "right": False},
        extras=extras or {"temperature": 25.0},
    )


def _replay() -> EpisodeReplay:
    return EpisodeReplay(
        [
            _frame(
                0.0,
                positions=(0.0, 0.0),
                velocities=(0.0, 2.0),
                torques=(1.0, -4.0),
                base_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                contacts={"left": True, "right": False},
                extras={"temperature": 20.0},
            ),
            _frame(
                1.0,
                positions=(2.0, 4.0),
                velocities=(2.0, 0.0),
                torques=(-3.0, 2.0),
                base_pose=(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 0.0),
                contacts={"left": False, "right": True},
                extras={"temperature": 30.0},
            ),
            _frame(
                2.0,
                positions=(4.0, 8.0),
                velocities=(4.0, -2.0),
                torques=(5.0, 1.0),
                base_pose=(2.0, 4.0, 6.0, 0.0, 0.0, 1.0, 0.0),
                contacts={"left": True, "right": True},
                extras={"temperature": 40.0},
            ),
        ]
    )


def test_telemetryframe_is_dataclass() -> None:
    assert is_dataclass(TelemetryFrame)


def test_telemetryframe_converts_fields_to_numpy_arrays() -> None:
    frame = TelemetryFrame(
        timestamp=1,
        joint_positions=[1, 2],
        joint_velocities=[3, 4],
        joint_torques=[5, 6],
        base_pose=[0, 0, 0, 0, 0, 0, 2],
        contact_states={"left": 1},
    )

    assert frame.timestamp == pytest.approx(1.0)
    assert frame.joint_positions.dtype == float
    np.testing.assert_allclose(frame.base_pose, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert frame.contact_states == {"left": True}


def test_telemetryframe_rejects_non_vector_joint_positions() -> None:
    with pytest.raises(ValueError, match="joint_positions"):
        TelemetryFrame(
            timestamp=0.0,
            joint_positions=np.zeros((1, 2)),
            joint_velocities=np.zeros(2),
            joint_torques=np.zeros(2),
            base_pose=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            contact_states={},
        )


def test_telemetryframe_rejects_mismatched_joint_velocity_shape() -> None:
    with pytest.raises(ValueError, match="joint_positions and joint_velocities"):
        TelemetryFrame(
            timestamp=0.0,
            joint_positions=np.zeros(2),
            joint_velocities=np.zeros(3),
            joint_torques=np.zeros(2),
            base_pose=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            contact_states={},
        )


def test_telemetryframe_rejects_mismatched_joint_torque_shape() -> None:
    with pytest.raises(ValueError, match="joint_positions and joint_torques"):
        TelemetryFrame(
            timestamp=0.0,
            joint_positions=np.zeros(2),
            joint_velocities=np.zeros(2),
            joint_torques=np.zeros(3),
            base_pose=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            contact_states={},
        )


def test_telemetryframe_rejects_invalid_base_pose_shape() -> None:
    with pytest.raises(ValueError, match="base_pose"):
        TelemetryFrame(
            timestamp=0.0,
            joint_positions=np.zeros(2),
            joint_velocities=np.zeros(2),
            joint_torques=np.zeros(2),
            base_pose=np.zeros(6),
            contact_states={},
        )


def test_telemetryframe_rejects_zero_norm_quaternion() -> None:
    with pytest.raises(ValueError, match="quaternion"):
        TelemetryFrame(
            timestamp=0.0,
            joint_positions=np.zeros(2),
            joint_velocities=np.zeros(2),
            joint_torques=np.zeros(2),
            base_pose=np.zeros(7),
            contact_states={},
        )


def test_telemetryframe_to_dict_serializes_numpy_arrays() -> None:
    frame = _frame(1.25)
    payload = frame.to_dict()

    assert payload["timestamp"] == pytest.approx(1.25)
    assert payload["joint_positions"] == [0.0, 1.0]
    assert payload["base_pose"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def test_telemetryframe_from_dict_round_trips_serialized_payload() -> None:
    frame = _frame(2.5, extras={"energy": 12.0})

    restored = TelemetryFrame.from_dict(frame.to_dict())

    assert restored.timestamp == pytest.approx(2.5)
    np.testing.assert_allclose(restored.joint_torques, frame.joint_torques)
    assert restored.extras == {"energy": 12.0}


def test_telemetryframe_from_dict_defaults_missing_extras_to_empty_mapping() -> None:
    payload = _frame(0.0).to_dict()
    payload.pop("extras")

    restored = TelemetryFrame.from_dict(payload)

    assert restored.extras == {}


def test_episoderecorder_rejects_negative_capacity() -> None:
    with pytest.raises(ValueError, match="max_frames"):
        EpisodeRecorder(max_frames=-1)


def test_episoderecorder_records_frames_until_capacity() -> None:
    recorder_instance = EpisodeRecorder(max_frames=2)
    first = _frame(0.0)
    second = _frame(1.0)

    recorder_instance.record(first)
    recorder_instance.record(second)

    assert recorder_instance.frames == [first, second]


def test_episoderecorder_silently_drops_frames_after_capacity() -> None:
    recorder_instance = EpisodeRecorder(max_frames=1)
    first = _frame(0.0)
    second = _frame(1.0)

    recorder_instance.record(first)
    recorder_instance.record(second)

    assert recorder_instance.frames == [first]


def test_episoderecorder_clear_resets_frames() -> None:
    recorder_instance = EpisodeRecorder(max_frames=2)
    recorder_instance.record(_frame(0.0))

    recorder_instance.clear()

    assert recorder_instance.frames == []


def test_episoderecorder_duration_is_zero_for_fewer_than_two_frames() -> None:
    recorder_instance = EpisodeRecorder()
    recorder_instance.record(_frame(0.5))

    assert recorder_instance.duration() == pytest.approx(0.0)


def test_episoderecorder_duration_uses_first_and_last_timestamp() -> None:
    recorder_instance = EpisodeRecorder()
    recorder_instance.record(_frame(1.5))
    recorder_instance.record(_frame(4.0))

    assert recorder_instance.duration() == pytest.approx(2.5)


def test_episoderecorder_from_dict_list_restores_frames() -> None:
    payload = [_frame(0.0).to_dict(), _frame(1.0).to_dict()]

    recorder_instance = EpisodeRecorder.from_dict_list(payload)

    assert len(recorder_instance.frames) == 2
    assert recorder_instance.max_frames == 10000


def test_episoderecorder_from_dict_list_preserves_large_loaded_count() -> None:
    payload = [_frame(float(index)).to_dict() for index in range(10001)]

    recorder_instance = EpisodeRecorder.from_dict_list(payload)

    assert len(recorder_instance.frames) == 10001
    assert recorder_instance.max_frames == 10001


def test_episodereplay_len_matches_frame_count() -> None:
    replay = _replay()

    assert len(replay) == 3


def test_episodereplay_time_range_uses_first_and_last_frame() -> None:
    replay = _replay()

    assert replay.time_range() == (0.0, 2.0)


def test_episodereplay_get_at_time_rejects_empty_episode() -> None:
    with pytest.raises(ValueError, match="empty episode"):
        EpisodeReplay([]).get_at_time(0.0)


def test_episodereplay_get_at_time_clamps_before_first_frame() -> None:
    replay = _replay()

    assert replay.get_at_time(-1.0) is replay[0]


def test_episodereplay_get_at_time_clamps_after_last_frame() -> None:
    replay = _replay()

    assert replay.get_at_time(10.0) is replay[-1]


def test_episodereplay_get_at_time_returns_exact_timestamp_frame() -> None:
    replay = _replay()

    assert replay.get_at_time(1.0) is replay[1]


def test_episodereplay_get_at_time_interpolates_joint_positions() -> None:
    frame = _replay().get_at_time(0.5)

    np.testing.assert_allclose(frame.joint_positions, [1.0, 2.0])


def test_episodereplay_get_at_time_interpolates_joint_torques() -> None:
    frame = _replay().get_at_time(0.5)

    np.testing.assert_allclose(frame.joint_torques, [-1.0, -1.0])


def test_episodereplay_get_at_time_interpolates_base_position() -> None:
    frame = _replay().get_at_time(0.5)

    np.testing.assert_allclose(frame.base_pose[:3], [0.5, 1.0, 1.5])


def test_episodereplay_get_at_time_normalizes_interpolated_quaternion() -> None:
    frame = _replay().get_at_time(0.5)

    assert np.linalg.norm(frame.base_pose[3:]) == pytest.approx(1.0)
    np.testing.assert_allclose(frame.base_pose[3:], [0.0, 0.0, 0.70710678, 0.70710678], atol=1e-7)


def test_episodereplay_get_at_time_uses_nearest_contact_state() -> None:
    frame = _replay().get_at_time(0.75)

    assert frame.contact_states == {"left": False, "right": True}


def test_episodereplay_get_at_time_uses_nearest_extras() -> None:
    frame = _replay().get_at_time(0.25)

    assert frame.extras == {"temperature": 20.0}


def test_episodereplay_get_at_time_uses_lower_frame_on_equal_distance_tie() -> None:
    frame = _replay().get_at_time(0.5)

    assert frame.contact_states == {"left": True, "right": False}
    assert frame.extras == {"temperature": 20.0}


def test_telemetrystats_mean_joint_positions_computes_per_joint_average() -> None:
    stats = TelemetryStats(_replay())

    np.testing.assert_allclose(stats.mean_joint_positions(), [2.0, 4.0])


def test_telemetrystats_max_joint_torques_uses_absolute_value() -> None:
    stats = TelemetryStats(_replay())

    np.testing.assert_allclose(stats.max_joint_torques(), [5.0, 4.0])


def test_telemetrystats_contact_ratio_returns_fraction_in_contact() -> None:
    stats = TelemetryStats(_replay())

    assert stats.contact_ratio("left") == pytest.approx(2.0 / 3.0)


def test_telemetrystats_total_duration_matches_replay_range() -> None:
    stats = TelemetryStats(_replay())

    assert stats.total_duration() == pytest.approx(2.0)


def test_telemetrystats_mean_joint_positions_requires_frames() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        TelemetryStats(EpisodeReplay([])).mean_joint_positions()


def test_telemetrystats_max_joint_torques_requires_frames() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        TelemetryStats(EpisodeReplay([])).max_joint_torques()


def test_telemetrystats_contact_ratio_requires_frames() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        TelemetryStats(EpisodeReplay([])).contact_ratio("left")


def test_recorder_module_exports_public_surface() -> None:
    assert recorder.__all__ == ["TelemetryFrame", "EpisodeRecorder", "EpisodeReplay", "TelemetryStats"]
    assert recorder.TelemetryFrame is TelemetryFrame
    assert recorder.EpisodeRecorder is EpisodeRecorder
    assert recorder.EpisodeReplay is EpisodeReplay
    assert recorder.TelemetryStats is TelemetryStats


def test_root_exports_recorder_module_and_classes() -> None:
    assert public_recorder_module is recorder
    assert PublicTelemetryFrame is TelemetryFrame
    assert PublicEpisodeRecorder is EpisodeRecorder
    assert PublicEpisodeReplay is EpisodeReplay
    assert PublicTelemetryStats is TelemetryStats
    assert optisim.recorder is recorder
    assert optisim.__version__ == "0.27.0"
