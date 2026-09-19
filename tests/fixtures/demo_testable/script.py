"""Canned repro script for the demo_testable fixture.

This stands in for what `repro run` would ask the model to generate, so that
`repro demo` exercises stages 5-7 (execute / compare / report) for real with no
API key and no network. Everything it prints is measured, not canned.
"""

from __future__ import annotations

import argparse

import numpy as np

from repro.harness import ForwardModel, collect, make_env, mpc_control

N_TRANSITIONS = 2000
HIDDEN = (200, 200)
EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
HORIZON = 15
N_CANDIDATES = 500
EPISODES = 5
ENV_ID = "Pendulum-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pendulum-v1 MPC repro")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    seed = args.seed

    print("ASSUMPTION: evaluation episodes run the Gymnasium default 200-step "
          "limit for Pendulum-v1 (the paper says '200 steps' but not whether "
          "that is the env limit or an override)")
    print("ASSUMPTION: the training buffer is collected fresh per seed rather "
          "than shared across seeds (the paper does not say which)")
    print("ASSUMPTION: 80/20 chronological train/holdout split for the reported "
          "one-step error (the paper reports no holdout protocol)")

    print(f"[seed {seed}] collecting {N_TRANSITIONS} random transitions from {ENV_ID}...")
    data = collect(ENV_ID, N_TRANSITIONS, policy="random", seed=seed)
    train, holdout = data.split(0.8)
    print(f"[seed {seed}] collected {len(data)} transitions "
          f"(train={len(train)}, holdout={len(holdout)})")

    env = make_env(ENV_ID)
    model = ForwardModel.for_env(env, hidden=HIDDEN, seed=seed)
    print(f"[seed {seed}] fitting forward model for {EPOCHS} epochs...")
    stats = model.fit(
        train, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, seed=seed
    )
    print(f"[seed {seed}] train_mse={stats['train_loss']:.6f}")
    print(f"[seed {seed}] holdout one-step error={model.one_step_error(holdout):.6f}")

    print(f"[seed {seed}] running MPC: horizon={HORIZON}, "
          f"candidates={N_CANDIDATES}, episodes={EPISODES}...")
    mean_return = mpc_control(
        ENV_ID,
        model,
        horizon=HORIZON,
        n_candidates=N_CANDIDATES,
        episodes=EPISODES,
        seed=seed,
    )
    env.close()

    print(f"[seed {seed}] mean episode return over {EPISODES} episodes: {mean_return:.2f}")
    print(f"RESULT: mean_episode_return={float(np.round(mean_return, 4))}")


if __name__ == "__main__":
    main()
