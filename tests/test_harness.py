"""Harness shapes and contracts. Budgets are tiny so the suite stays fast."""

from __future__ import annotations

import numpy as np
import pytest

from repro.harness import (
    ForwardModel,
    Transitions,
    action_dim,
    collect,
    make_env,
    mpc_control,
    perturb,
    sample_efficiency_curve,
)


# --- collect ---------------------------------------------------------------

def test_collect_shapes_continuous() -> None:
    data = collect("Pendulum-v1", 64, seed=0)
    assert len(data) == 64
    assert data.states.shape == (64, 3)
    assert data.actions.shape == (64, 1)
    assert data.next_states.shape == (64, 3)
    assert data.rewards.shape == (64,)
    assert data.states.dtype == np.float32


def test_collect_shapes_discrete_action_is_one_wide() -> None:
    data = collect("CartPole-v1", 64, seed=0)
    assert data.states.shape == (64, 4)
    assert data.actions.shape == (64, 1)
    assert data.next_states.shape == (64, 4)


def test_collect_is_deterministic_for_a_seed() -> None:
    first = collect("Pendulum-v1", 32, seed=7)
    second = collect("Pendulum-v1", 32, seed=7)
    assert np.allclose(first.states, second.states)
    assert np.allclose(first.actions, second.actions)


def test_collect_differs_across_seeds() -> None:
    assert not np.allclose(
        collect("Pendulum-v1", 32, seed=0).states,
        collect("Pendulum-v1", 32, seed=1).states,
    )


def test_collect_accepts_a_callable_policy() -> None:
    calls: list[np.ndarray] = []

    def zero_policy(observation: np.ndarray) -> np.ndarray:
        calls.append(observation)
        return np.zeros(1, dtype=np.float32)

    data = collect("Pendulum-v1", 16, policy=zero_policy, seed=0)
    assert len(calls) == 16
    assert np.allclose(data.actions, 0.0)


def test_collect_rejects_a_non_positive_budget() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        collect("Pendulum-v1", 0, seed=0)


def test_collect_rejects_an_unknown_policy() -> None:
    with pytest.raises(ValueError, match="Unsupported policy"):
        collect("Pendulum-v1", 8, policy="expert", seed=0)  # type: ignore[arg-type]


def test_make_env_reports_the_id_it_failed_on() -> None:
    with pytest.raises(RuntimeError, match="NotAnEnv-v9"):
        make_env("NotAnEnv-v9")


# --- Transitions -----------------------------------------------------------

def test_transitions_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="disagree on length"):
        Transitions(np.zeros((4, 3)), np.zeros((3, 1)), np.zeros((4, 3)))


def test_transitions_split_partitions_the_buffer() -> None:
    data = collect("Pendulum-v1", 100, seed=0)
    train, holdout = data.split(0.8)
    assert len(train) == 80
    assert len(holdout) == 20
    assert np.allclose(train.states[0], data.states[0])
    assert np.allclose(holdout.states[0], data.states[80])


def test_transitions_split_keeps_at_least_one_training_row() -> None:
    train, _ = collect("Pendulum-v1", 4, seed=0).split(0.01)
    assert len(train) >= 1


# --- ForwardModel ----------------------------------------------------------

def test_forward_model_for_env_matches_the_spaces() -> None:
    model = ForwardModel.for_env(make_env("Pendulum-v1"))
    assert model.state_dim == 3
    assert model.act_dim == 1
    assert action_dim(make_env("CartPole-v1")) == 1


def test_predict_single_and_batch_shapes() -> None:
    data = collect("Pendulum-v1", 64, seed=0)
    model = ForwardModel(3, 1, hidden=(16,), seed=0)
    model.fit(data, epochs=2, seed=0)
    assert model.predict(data.states[0], data.actions[0]).shape == (3,)
    assert model.predict(data.states[:5], data.actions[:5]).shape == (5, 3)


def test_fit_reports_a_finite_loss_and_marks_the_model_fitted() -> None:
    data = collect("Pendulum-v1", 128, seed=0)
    model = ForwardModel(3, 1, hidden=(32, 32), seed=0)
    assert not model.fitted
    stats = model.fit(data, epochs=3, seed=0)
    assert model.fitted
    assert np.isfinite(stats["train_loss"])
    assert stats["epochs"] == 3.0


def test_training_reduces_one_step_error() -> None:
    data = collect("Pendulum-v1", 512, seed=0)
    model = ForwardModel(3, 1, hidden=(64, 64), seed=0)
    before = model.one_step_error(data)
    model.fit(data, epochs=30, seed=0)
    assert model.one_step_error(data) < before


def test_fit_is_deterministic_for_a_seed() -> None:
    data = collect("Pendulum-v1", 128, seed=0)
    first = ForwardModel(3, 1, hidden=(16,), seed=3)
    second = ForwardModel(3, 1, hidden=(16,), seed=3)
    first.fit(data, epochs=3, seed=3)
    second.fit(data, epochs=3, seed=3)
    assert first.one_step_error(data) == pytest.approx(second.one_step_error(data))


def test_fit_on_an_empty_buffer_raises() -> None:
    empty = Transitions(np.zeros((0, 3)), np.zeros((0, 1)), np.zeros((0, 3)))
    with pytest.raises(ValueError, match="empty buffer"):
        ForwardModel(3, 1).fit(empty)


def test_one_step_error_on_an_empty_buffer_raises() -> None:
    empty = Transitions(np.zeros((0, 3)), np.zeros((0, 1)), np.zeros((0, 3)))
    with pytest.raises(ValueError, match="empty buffer"):
        ForwardModel(3, 1).one_step_error(empty)


# --- mpc_control -----------------------------------------------------------

def test_mpc_control_returns_a_finite_scalar() -> None:
    data = collect("Pendulum-v1", 256, seed=0)
    model = ForwardModel(3, 1, hidden=(32, 32), seed=0)
    model.fit(data, epochs=5, seed=0)
    value = mpc_control(
        "Pendulum-v1", model, horizon=3, n_candidates=8, episodes=1, seed=0, max_steps=5
    )
    assert isinstance(value, float)
    assert np.isfinite(value)


def test_mpc_control_handles_a_discrete_action_space() -> None:
    data = collect("CartPole-v1", 256, seed=0)
    model = ForwardModel(4, 1, hidden=(32, 32), seed=0)
    model.fit(data, epochs=5, seed=0)
    value = mpc_control(
        "CartPole-v1", model, horizon=3, n_candidates=8, episodes=1, seed=0, max_steps=5
    )
    assert np.isfinite(value)


def test_mpc_control_refuses_an_env_with_no_analytic_reward() -> None:
    model = ForwardModel(2, 1, hidden=(8,), seed=0)
    with pytest.raises(ValueError, match="No analytic reward"):
        mpc_control("Reacher-v5", model, horizon=2, n_candidates=2, episodes=1)


def test_mpc_control_accepts_a_custom_reward_fn() -> None:
    data = collect("Pendulum-v1", 128, seed=0)
    model = ForwardModel(3, 1, hidden=(16,), seed=0)
    model.fit(data, epochs=2, seed=0)
    value = mpc_control(
        "Pendulum-v1",
        model,
        horizon=2,
        n_candidates=4,
        episodes=1,
        seed=0,
        max_steps=3,
        reward_fn=lambda states, actions: np.zeros(states.shape[0], dtype=np.float32),
    )
    assert np.isfinite(value)


# --- perturb ---------------------------------------------------------------

def test_perturb_scales_pendulum_mass() -> None:
    env = make_env("Pendulum-v1")
    env.reset(seed=0)
    before = env.unwrapped.m
    changed = perturb(env, "mass", 1.5)
    assert changed == {"m": pytest.approx(before * 1.5)}
    assert env.unwrapped.m == pytest.approx(before * 1.5)


def test_perturb_keeps_cartpole_derived_quantities_consistent() -> None:
    env = make_env("CartPole-v1")
    env.reset(seed=0)
    perturb(env, "mass", 2.0)
    unwrapped = env.unwrapped
    assert unwrapped.total_mass == pytest.approx(unwrapped.masspole + unwrapped.masscart)
    assert unwrapped.polemass_length == pytest.approx(unwrapped.masspole * unwrapped.length)


def test_perturb_refuses_to_silently_do_nothing() -> None:
    env = make_env("Pendulum-v1")
    env.reset(seed=0)
    with pytest.raises(ValueError, match="Refusing to pretend"):
        perturb(env, "friction", 1.5)


def test_perturb_rejects_unknown_kinds_and_bad_magnitudes() -> None:
    env = make_env("Pendulum-v1")
    env.reset(seed=0)
    with pytest.raises(ValueError, match="Unknown perturbation"):
        perturb(env, "gravity", 1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be positive"):
        perturb(env, "mass", 0.0)


# --- sample_efficiency_curve ----------------------------------------------

def test_sample_efficiency_curve_shape_and_ordering() -> None:
    curve = sample_efficiency_curve("Pendulum-v1", [128, 64], epochs=2, mpc=False)
    assert curve["sample_sizes"] == [64.0, 128.0]  # sorted ascending
    assert len(curve["one_step_error"]) == 2
    assert len(curve["return"]) == 2
    assert all(np.isfinite(value) for value in curve["one_step_error"])
    assert all(np.isnan(value) for value in curve["return"])  # mpc=False


def test_sample_efficiency_curve_with_mpc_returns_finite_values() -> None:
    curve = sample_efficiency_curve(
        "Pendulum-v1", [128], epochs=2, horizon=2, n_candidates=4, episodes=1
    )
    assert np.isfinite(curve["return"][0])


def test_sample_efficiency_curve_rejects_an_empty_request() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        sample_efficiency_curve("Pendulum-v1", [])
