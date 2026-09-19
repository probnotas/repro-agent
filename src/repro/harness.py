"""Reusable model-based-control primitives that generated repro scripts import.

Deliberately small and dependency-light (gymnasium, torch, numpy) and restricted
to classic-control environments, so a repro runs on a laptop CPU with no MuJoCo
licence and no GPU. See LIMITATIONS in the README for what that substitution
does and does not buy you.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import gymnasium as gym
import numpy as np
import torch
from torch import nn

PolicyKind = Literal["random"]
PerturbKind = Literal["mass", "damping", "friction"]

#: Physics attributes we know how to scale, per perturbation kind, in priority
#: order. Classic-control envs name these inconsistently, hence the candidates.
_PERTURB_ATTRS: dict[str, tuple[str, ...]] = {
    "mass": ("m", "masspole", "masscart", "total_mass", "link_mass"),
    "damping": ("b", "damping", "tau"),
    "friction": ("friction", "mu", "force_mag", "power"),
}


@dataclass
class Transitions:
    """A flat buffer of ``(state, action, next_state)`` with matching leading dim."""

    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    rewards: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __post_init__(self) -> None:
        n = self.states.shape[0]
        if not (self.actions.shape[0] == n == self.next_states.shape[0]):
            raise ValueError(
                "Transitions arrays disagree on length: "
                f"states={self.states.shape[0]} actions={self.actions.shape[0]} "
                f"next_states={self.next_states.shape[0]}"
            )

    def split(self, frac: float = 0.8) -> tuple["Transitions", "Transitions"]:
        """Chronological train/holdout split; ``frac`` is the training share."""
        cut = max(1, int(len(self) * frac))
        return (
            Transitions(
                self.states[:cut],
                self.actions[:cut],
                self.next_states[:cut],
                self.rewards[:cut] if len(self.rewards) else self.rewards,
            ),
            Transitions(
                self.states[cut:],
                self.actions[cut:],
                self.next_states[cut:],
                self.rewards[cut:] if len(self.rewards) else self.rewards,
            ),
        )


def _as_action_array(action: Any, env: gym.Env) -> np.ndarray:
    """Represent an action as a float vector regardless of the action space kind."""
    if isinstance(env.action_space, gym.spaces.Discrete):
        return np.asarray([float(action)], dtype=np.float32)
    return np.asarray(action, dtype=np.float32).reshape(-1)


def action_dim(env: gym.Env) -> int:
    """Width of the action vector this harness feeds the forward model."""
    if isinstance(env.action_space, gym.spaces.Discrete):
        return 1
    return int(np.prod(env.action_space.shape))


def make_env(env_id: str, **kwargs: Any) -> gym.Env:
    """Construct a gymnasium env, failing loudly with the id that was attempted."""
    try:
        return gym.make(env_id, **kwargs)
    except Exception as exc:  # gymnasium raises several unrelated types here
        raise RuntimeError(f"Could not create environment {env_id!r}: {exc}") from exc


def collect(
    env_id: str | gym.Env,
    n_transitions: int,
    policy: PolicyKind | Callable[[np.ndarray], Any] = "random",
    seed: int = 0,
) -> Transitions:
    """Roll out ``n_transitions`` steps and return the (s, a, s') buffer.

    ``policy`` is either ``"random"`` (uniform over the action space) or a
    callable mapping an observation to an action.
    """
    if n_transitions <= 0:
        raise ValueError(f"n_transitions must be positive, got {n_transitions}")

    env = make_env(env_id) if isinstance(env_id, str) else env_id
    rng = np.random.default_rng(seed)
    env.action_space.seed(seed)

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    rewards: list[float] = []

    obs, _ = env.reset(seed=seed)
    for step in range(n_transitions):
        if policy == "random":
            action = env.action_space.sample()
        elif callable(policy):
            action = policy(np.asarray(obs, dtype=np.float32))
        else:
            raise ValueError(f"Unsupported policy {policy!r}; use 'random' or a callable")

        next_obs, reward, terminated, truncated, _ = env.step(action)
        states.append(np.asarray(obs, dtype=np.float32))
        actions.append(_as_action_array(action, env))
        next_states.append(np.asarray(next_obs, dtype=np.float32))
        rewards.append(float(reward))

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))

    if isinstance(env_id, str):
        env.close()

    return Transitions(
        states=np.stack(states),
        actions=np.stack(actions),
        next_states=np.stack(next_states),
        rewards=np.asarray(rewards, dtype=np.float32),
    )


class ForwardModel(nn.Module):
    """A small MLP dynamics model predicting the state *delta*.

    Predicting ``s' - s`` rather than ``s'`` is the standard parameterisation in
    the model-based control literature and is what makes short-horizon MPC work
    at all on classic control.
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden: tuple[int, ...] = (200, 200),
        seed: int = 0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.state_dim = state_dim
        self.act_dim = act_dim
        layers: list[nn.Module] = []
        in_dim = state_dim + act_dim
        for width in hidden:
            layers += [nn.Linear(in_dim, width), nn.ReLU()]
            in_dim = width
        layers.append(nn.Linear(in_dim, state_dim))
        self.net = nn.Sequential(*layers)
        self._state_mean = torch.zeros(state_dim)
        self._state_std = torch.ones(state_dim)
        self._act_mean = torch.zeros(act_dim)
        self._act_std = torch.ones(act_dim)
        self.fitted = False

    @classmethod
    def for_env(cls, env: gym.Env, **kwargs: Any) -> "ForwardModel":
        """Build a model whose input/output widths match ``env``."""
        obs_dim = int(np.prod(env.observation_space.shape))
        return cls(obs_dim, action_dim(env), **kwargs)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        norm_s = (states - self._state_mean) / self._state_std
        norm_a = (actions - self._act_mean) / self._act_std
        delta = self.net(torch.cat([norm_s, norm_a], dim=-1))
        return states + delta

    def fit(
        self,
        data: Transitions,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        seed: int = 0,
        verbose: bool = False,
    ) -> dict[str, float]:
        """Train on ``data`` and return ``{"train_loss": ..., "epochs": ...}``."""
        if len(data) == 0:
            raise ValueError("Cannot fit a ForwardModel on an empty buffer")
        torch.manual_seed(seed)

        states = torch.as_tensor(data.states, dtype=torch.float32)
        actions = torch.as_tensor(data.actions, dtype=torch.float32)
        targets = torch.as_tensor(data.next_states, dtype=torch.float32) - states

        self._state_mean = states.mean(0)
        self._state_std = states.std(0).clamp_min(1e-6)
        self._act_mean = actions.mean(0)
        self._act_std = actions.std(0).clamp_min(1e-6)

        optimiser = torch.optim.Adam(self.net.parameters(), lr=lr)
        n = states.shape[0]
        generator = torch.Generator().manual_seed(seed)
        last_loss = float("nan")

        for epoch in range(epochs):
            order = torch.randperm(n, generator=generator)
            epoch_loss = 0.0
            batches = 0
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                norm_s = (states[idx] - self._state_mean) / self._state_std
                norm_a = (actions[idx] - self._act_mean) / self._act_std
                predicted = self.net(torch.cat([norm_s, norm_a], dim=-1))
                loss = nn.functional.mse_loss(predicted, targets[idx])
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                epoch_loss += float(loss.item())
                batches += 1
            last_loss = epoch_loss / max(batches, 1)
            if verbose and epoch % 20 == 0:
                print(f"  epoch {epoch:3d}  train_mse={last_loss:.6f}", flush=True)

        self.fitted = True
        return {"train_loss": last_loss, "epochs": float(epochs)}

    @torch.no_grad()
    def predict(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Predict ``s'`` for one or a batch of ``(s, a)`` pairs."""
        states = torch.as_tensor(np.atleast_2d(s), dtype=torch.float32)
        actions = torch.as_tensor(np.atleast_2d(a), dtype=torch.float32)
        if actions.shape[-1] != self.act_dim:
            actions = actions.reshape(states.shape[0], self.act_dim)
        out = self.forward(states, actions).numpy()
        return out[0] if np.asarray(s).ndim == 1 else out

    @torch.no_grad()
    def one_step_error(self, data: Transitions) -> float:
        """Mean squared one-step prediction error over ``data``."""
        if len(data) == 0:
            raise ValueError("Cannot score a ForwardModel on an empty buffer")
        states = torch.as_tensor(data.states, dtype=torch.float32)
        actions = torch.as_tensor(data.actions, dtype=torch.float32)
        truth = torch.as_tensor(data.next_states, dtype=torch.float32)
        return float(nn.functional.mse_loss(self.forward(states, actions), truth).item())


def _reward_fn(env_id: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Analytic reward for the supported classic-control envs.

    Random-shooting MPC needs to score imagined trajectories, which means a
    reward it can evaluate without stepping the real environment.
    """
    name = env_id.split("-")[0].lower()

    if name == "pendulum":

        def pendulum(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
            cos_t, sin_t, dtheta = states[..., 0], states[..., 1], states[..., 2]
            theta = np.arctan2(sin_t, cos_t)
            torque = actions[..., 0]
            return -(theta**2 + 0.1 * dtheta**2 + 0.001 * torque**2)

        return pendulum

    if name in {"cartpole", "cartpoleswingup"}:

        def cartpole(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
            x, theta = states[..., 0], states[..., 2]
            alive = (np.abs(x) < 2.4) & (np.abs(theta) < 0.2095)
            return alive.astype(np.float32)

        return cartpole

    if name in {"mountaincarcontinuous", "mountaincar"}:

        def mountaincar(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
            return states[..., 0]

        return mountaincar

    if name == "acrobot":

        def acrobot(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
            # -cos(t1) - cos(t2 + t1) > 1.0 is the gymnasium success condition.
            height = -states[..., 0] - (
                states[..., 0] * states[..., 2] - states[..., 1] * states[..., 3]
            )
            return height

        return acrobot

    raise ValueError(
        f"No analytic reward for {env_id!r}. Supported: Pendulum-v1, CartPole-v1, "
        "MountainCarContinuous-v0, Acrobot-v1. Pass reward_fn= explicitly to use "
        "another environment."
    )


def mpc_control(
    env: gym.Env | str,
    model: ForwardModel,
    horizon: int = 15,
    n_candidates: int = 500,
    episodes: int = 5,
    seed: int = 0,
    max_steps: int | None = None,
    reward_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> float:
    """Random-shooting receding-horizon MPC; returns the mean episode return.

    Each step samples ``n_candidates`` action sequences of length ``horizon``,
    rolls them through ``model``, scores them with the analytic reward, and
    executes the first action of the best sequence.
    """
    env_id = env if isinstance(env, str) else getattr(env.spec, "id", "")
    # Resolve the reward before building anything: an unsupported env should say
    # so immediately rather than after paying for env construction.
    score = reward_fn or _reward_fn(env_id)
    environment = make_env(env) if isinstance(env, str) else env

    rng = np.random.default_rng(seed)
    discrete = isinstance(environment.action_space, gym.spaces.Discrete)
    if discrete:
        n_actions = int(environment.action_space.n)
    else:
        low = np.asarray(environment.action_space.low, dtype=np.float32)
        high = np.asarray(environment.action_space.high, dtype=np.float32)

    returns: list[float] = []
    for episode in range(episodes):
        obs, _ = environment.reset(seed=seed + episode)
        total = 0.0
        steps = 0
        while True:
            if discrete:
                candidates = rng.integers(
                    0, n_actions, size=(n_candidates, horizon, 1)
                ).astype(np.float32)
            else:
                candidates = rng.uniform(
                    low, high, size=(n_candidates, horizon, low.shape[0])
                ).astype(np.float32)

            states = np.repeat(
                np.asarray(obs, dtype=np.float32)[None, :], n_candidates, axis=0
            )
            totals = np.zeros(n_candidates, dtype=np.float32)
            for t in range(horizon):
                actions = candidates[:, t, :]
                states = model.predict(states, actions)
                totals += score(states, actions)

            best = candidates[int(np.argmax(totals)), 0]
            action = int(best[0]) if discrete else np.clip(best, low, high)

            obs, reward, terminated, truncated, _ = environment.step(action)
            total += float(reward)
            steps += 1
            if terminated or truncated or (max_steps is not None and steps >= max_steps):
                break
        returns.append(total)

    if isinstance(env, str):
        environment.close()
    return float(np.mean(returns))


def perturb(env: gym.Env, kind: PerturbKind, magnitude: float) -> dict[str, float]:
    """Scale a physics parameter in place, e.g. ``perturb(env, "mass", 1.5)``.

    Returns ``{attribute: new_value}`` for every attribute that was changed, so a
    repro script can report exactly what it perturbed. Raises if nothing on this
    environment matches ``kind`` -- a silent no-op would fake an experiment.
    """
    if kind not in _PERTURB_ATTRS:
        raise ValueError(f"Unknown perturbation {kind!r}; use one of {sorted(_PERTURB_ATTRS)}")
    if magnitude <= 0:
        raise ValueError(f"magnitude must be positive, got {magnitude}")

    target = env.unwrapped
    changed: dict[str, float] = {}
    for attribute in _PERTURB_ATTRS[kind]:
        if hasattr(target, attribute):
            current = getattr(target, attribute)
            if isinstance(current, (int, float)):
                new_value = float(current) * magnitude
                setattr(target, attribute, new_value)
                changed[attribute] = new_value

    if not changed:
        raise ValueError(
            f"Environment {type(target).__name__} exposes no {kind!r} parameter "
            f"(looked for {_PERTURB_ATTRS[kind]}). Refusing to pretend it was perturbed."
        )

    # CartPole derives total_mass/polemass_length from its components.
    if hasattr(target, "total_mass") and hasattr(target, "masscart"):
        target.total_mass = target.masspole + target.masscart
        if hasattr(target, "polemass_length") and hasattr(target, "length"):
            target.polemass_length = target.masspole * target.length
    return changed


def sample_efficiency_curve(
    env_id: str,
    sample_sizes: list[int] | tuple[int, ...],
    seed: int = 0,
    epochs: int = 60,
    horizon: int = 15,
    n_candidates: int = 300,
    episodes: int = 3,
    mpc: bool = True,
) -> dict[str, list[float]]:
    """Train on increasing amounts of data and measure what each buys you.

    Returns ``{"sample_sizes": [...], "one_step_error": [...], "return": [...]}``
    with one entry per requested sample size. Set ``mpc=False`` to skip control
    evaluation (much faster) -- the return list is then filled with NaN.
    """
    sizes = sorted(int(s) for s in sample_sizes)
    if not sizes:
        raise ValueError("sample_sizes must not be empty")

    errors: list[float] = []
    returns: list[float] = []
    for size in sizes:
        data = collect(env_id, size, policy="random", seed=seed)
        train, holdout = data.split(0.8)
        model = ForwardModel.for_env(make_env(env_id), seed=seed)
        model.fit(train, epochs=epochs, seed=seed)
        errors.append(model.one_step_error(holdout if len(holdout) else train))
        if mpc:
            returns.append(
                mpc_control(
                    env_id,
                    model,
                    horizon=horizon,
                    n_candidates=n_candidates,
                    episodes=episodes,
                    seed=seed,
                )
            )
        else:
            returns.append(float("nan"))

    return {
        "sample_sizes": [float(s) for s in sizes],
        "one_step_error": errors,
        "return": returns,
    }
