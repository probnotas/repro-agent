"""Ask the model for a self-contained repro script targeting our harness.

The script is written to disk and run in a subprocess -- never ``exec()``-ed in
this process -- so a bad generation cannot corrupt the agent's own state.
"""

from __future__ import annotations

import re

from .config import Settings
from .extract import Claim
from .llm import complete

HARNESS_API = """\
from repro.harness import (
    collect,                    # (env_id, n_transitions, policy="random", seed=0) -> Transitions
    ForwardModel,               # MLP dynamics model; .fit(data, epochs=, seed=), .predict(s, a),
                                #   .one_step_error(data), ForwardModel.for_env(env, seed=)
    mpc_control,                # (env_or_id, model, horizon=, n_candidates=, episodes=, seed=) -> mean return
    perturb,                    # (env, kind, magnitude) kind in {"mass","damping","friction"}
    sample_efficiency_curve,    # (env_id, sample_sizes, seed=, mpc=) -> dict of lists
    make_env,                   # (env_id) -> gymnasium env
)

sample_efficiency_curve returns {"sample_sizes": [...], "one_step_error": [...],
"return": [...]}. With mpc=False the "return" list is filled with NaN -- it skips
control evaluation entirely. Never compute a reported number from a NaN list.

Transitions has .states, .actions, .next_states, .rewards (numpy arrays, matching
leading dim), len(), and .split(frac) -> (train, holdout).
"""

SYSTEM_PROMPT = f"""\
You write ONE self-contained Python script that tests a single empirical claim
from a paper, using a pre-built harness. The script will be run with
`python script.py --seed N`, on CPU, with no GPU and no MuJoCo.

The harness is already installed and importable:
{HARNESS_API}

Hard requirements:
1. The script reads its seed from `--seed` (argparse, default 0) and uses that
   seed for EVERY stochastic step: data collection, model init, MPC.
2. The LAST line the script prints must be exactly:
       RESULT: <metric_name>=<float>
   where <metric_name> is a short snake_case name and <float> is a plain number.
   Print it with: print(f"RESULT: {{name}}={{value}}")
3. Use ONLY: the repro.harness imports above, numpy, torch, gymnasium, and the
   standard library. Do not pip install anything. Do not read from the network.
4. Classic control only: Pendulum-v1, CartPole-v1, MountainCarContinuous-v0,
   Acrobot-v1. If the paper used MuJoCo, substitute the closest classic-control
   env and print a line starting with "ASSUMPTION:" saying so.
5. It must finish in well under 10 minutes on one CPU core. Keep budgets modest
   (e.g. <= 20000 transitions, <= 200 training epochs, <= 500 MPC candidates,
   <= 10 evaluation episodes). Favour finishing over faithfulness to scale.
6. For EVERY value you had to invent because the paper does not state it, print
   one line: "ASSUMPTION: <what you chose> (<why>)". These lines are collected
   into the report's mandatory assumptions section, so do not omit any.
7. Print progress to stdout as you go so a long run is legible.
8. No try/except that swallows an error into a fake number. If something fails,
   let it raise -- a crash is an honest RUN_FAILED, a fabricated number is not.

Return ONLY the Python source. No markdown fences, no commentary.
"""

USER_TEMPLATE = """\
Paper: {title}

Claim to test: {claim_text}

method: {method}
environment: {environment}
metric: {metric}
reported_value: {reported_value}
conditions:
{conditions}
the paper does NOT specify:
{missing}

Write the script that measures the metric above so it can be compared against
reported_value. Choose sensible values for everything the paper leaves out, and
declare each one with an ASSUMPTION: line.
"""

_FENCE_START = re.compile(r"^\s*```(?:python|py)?\s*\n", re.IGNORECASE)
_FENCE_END = re.compile(r"\n```\s*$")


def strip_code_fences(text: str) -> str:
    """Remove markdown fences a model may have added despite instructions."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = _FENCE_START.sub("", cleaned)
        cleaned = _FENCE_END.sub("", cleaned)
        cleaned = cleaned.removesuffix("```").strip()
    return cleaned


class GenerationError(RuntimeError):
    """Raised when the model did not return a usable script."""


def generate_script(
    title: str,
    claim: Claim,
    *,
    settings: Settings | None = None,
) -> str:
    """Generate the repro script for ``claim`` and sanity-check its shape."""
    conditions = "\n".join(f"  - {item}" for item in claim.conditions) or "  (none given)"
    missing = "\n".join(f"  - {item}" for item in claim.missing) or "  (nothing flagged)"
    user = USER_TEMPLATE.format(
        title=title,
        claim_text=claim.claim_text,
        method=claim.method,
        environment=claim.environment,
        metric=claim.metric,
        reported_value=claim.reported_value,
        conditions=conditions,
        missing=missing,
    )
    raw = complete(SYSTEM_PROMPT, user, json_mode=False, settings=settings)
    script = strip_code_fences(raw)

    if not script:
        raise GenerationError("Model returned an empty script.")
    if "RESULT:" not in script:
        raise GenerationError(
            "Generated script never prints a `RESULT:` line, so nothing could be "
            "compared. Refusing to run it."
        )
    return script
