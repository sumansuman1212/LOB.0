import math

import numpy as np
import pytest
from hypothesis import given, strategies as st

from src.python.trajectory_buffer import ACTION_DIM, OBS_DIM, TrajectoryBuffer, Transition


def transition(*, reward=0.0, value=0.0, done=False, log_prob=0.0):
    return Transition(
        obs=np.zeros(OBS_DIM, dtype=np.float32),
        action=np.zeros(ACTION_DIM, dtype=np.float32),
        log_prob=log_prob,
        value=value,
        reward=reward,
        done=done,
    )


@given(
    st.lists(
        st.tuples(
            st.floats(min_value=-1_000.0, max_value=1_000.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-1_000.0, max_value=1_000.0, allow_nan=False, allow_infinity=False),
            st.booleans(),
        ),
        min_size=2,
        max_size=32,
    )
)
def test_gae_is_finite_and_normalized_for_nonconstant_rollouts(samples):
    buffer = TrajectoryBuffer(64)
    for reward, value, done in samples:
        assert buffer.push(transition(reward=reward, value=value, done=done))

    buffer.compute_advantages(0.99, 0.95, last_value=0.25)
    advantages = buffer.advantage_tensor().numpy()
    assert np.all(np.isfinite(advantages))
    # In float32 arithmetic, machine epsilon allows zero-mean up to float32 numerical rounding tolerance
    assert abs(float(advantages.mean())) < 1e-3
    # A degenerate rollout has zero variance by definition; otherwise the
    # requested normalization produces unit population standard deviation.
    if np.ptp(advantages) > 1e-4:
        assert abs(float(advantages.std()) - 1.0) < 1e-3


@pytest.mark.parametrize("bad_log_prob", [math.nan, math.inf, -math.inf])
def test_nonfinite_log_prob_is_rejected_without_mutating_buffer(bad_log_prob):
    buffer = TrajectoryBuffer(8)
    assert buffer.push(transition(log_prob=0.5))
    size_before = buffer.size
    observations_before = buffer.obs_tensor().clone()

    assert not buffer.push(transition(log_prob=bad_log_prob))
    assert buffer.size == size_before
    assert np.array_equal(buffer.obs_tensor().numpy(), observations_before.numpy())


def test_buffer_capacity_gae_and_clear_lifecycle():
    buffer = TrajectoryBuffer(4)
    for index in range(4):
        assert buffer.push(transition(reward=float(index), value=0.5, done=index == 3))
    assert buffer.full()
    with pytest.raises(RuntimeError, match="full"):
        buffer.push(transition())

    buffer.compute_advantages(0.99, 0.95)
    assert buffer.obs_tensor().shape == (4, OBS_DIM)
    assert buffer.action_tensor().shape == (4, ACTION_DIM)
    assert buffer.return_tensor().shape == (4,)

    capacity = buffer.capacity
    buffer.clear()
    assert buffer.capacity == capacity
    assert buffer.size == 0
    with pytest.raises(RuntimeError, match="not yet available"):
        buffer.advantage_tensor()
