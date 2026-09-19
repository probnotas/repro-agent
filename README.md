# repro

A paper replication agent for model-based robot control research.

Give it an arXiv link. It gives you back a **report card** saying whether the
paper's headline empirical claim actually reproduces when you run it.

It is **not a summarizer**. The unit of value here is *a claim that survived
execution* — or an explicit, itemized statement of why the paper does not say
enough for the claim to be tested at all.

```
╭───────────────────────────── repro report card ──────────────────────────────╮
│ paper        Random-Shooting MPC over a Learned Dynamics Model is            │
│              Sample-Efficient on Pendulum Swing-Up  (arXiv:demo-0001)        │
│ claim        A two-layer MLP dynamics model trained on 2000 uniformly-random │
│              Pendulum-v1 transitions supports random-shooting MPC that       │
│              reaches a mean episode return of -210.                          │
│ metric       mean_episode_return                                             │
│ reported     -210                                                            │
│ ours         -194.5 ± 0 (N=1)                                                │
│ delta        +7.4%                                                           │
│ tolerance    ±25%                                                            │
│ verdict      REPRODUCED                                                      │
│ model        anthropic/claude-sonnet-5                                       │
│ artifacts    runs/demo-0001/20260919T154841Z                                 │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## The four verdicts

| verdict | what it means |
| --- | --- |
| **`REPRODUCED`** | The script ran and our number landed inside the tolerance band around the paper's number. |
| **`NOT_REPRODUCED`** | The script ran and produced a number, but it fell outside the tolerance band. **This is a statement about this run, not a verdict on the paper** — see [Limitations](#limitations). |
| **`UNTESTABLE`** | The paper does not specify enough to run a fair test. The report lists exactly which items are missing. This is a first-class, valuable outcome, not a failure — a paper that cannot be checked is worth knowing about. |
| **`RUN_FAILED`** | The generated script crashed, timed out, or never printed a result. Nothing is concluded about the paper; the stack trace is in `runs/`. |

`repro verdicts` prints this table in the terminal.

## How it works

Seven stages, each writing its artifacts to disk before the next begins:

1. **fetch** — arXiv API for metadata, PDF via `pymupdf`, cached in `.cache/<arxiv_id>/` so re-runs are free and offline.
2. **extract** — the LLM returns a structured claim: the testable sentence, the environment, the metric, the reported value, and — critically — a per-field record of *what the paper actually stated* versus what a re-implementer would have to guess.
3. **triage** — if too much is missing, stop here and emit `UNTESTABLE` with the specific list. The agent does not guess and pretend.
4. **generate** — otherwise, the LLM writes a single self-contained repro script against the harness in `src/repro/harness.py`. Its last line must be exactly `RESULT: <metric>=<float>`.
5. **execute** — the script runs in a **subprocess** (never `exec()` in-process), once per seed, with a hard timeout. Script, stdout, stderr and exit code are written to `runs/<arxiv_id>/<timestamp>/seed_<n>/`.
6. **compare** — parse `RESULT`, aggregate across seeds (mean ± sample std), compare to the reported value within the tolerance band.
7. **report** — `runs/<arxiv_id>/report.md` plus a colored terminal table, including a mandatory **ASSUMPTIONS** section listing every value the agent had to invent.

Every number in a report traces back to a file under `runs/`.

## Install

Python 3.11 or newer.

```bash
git clone https://github.com/probnotas/repro-agent.git
cd repro-agent
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Everything here runs on classic-control environments on a laptop: no GPU, no
MuJoCo licence. If you would rather not pull the ~2.5 GB CUDA build of PyTorch,
install the CPU wheel first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

Check it works:

```bash
repro demo        # ~15s, no network and no API key needed
pytest            # ~30s, no network
```

## Get an OpenRouter key

All model calls go through [OpenRouter](https://openrouter.ai), never a vendor
SDK directly. One key, any model.

1. Create a key at **https://openrouter.ai/keys**.
2. Copy the example env file and fill it in:

```bash
cp .env.example .env
```

```ini
# .env
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=anthropic/claude-sonnet-5
```

`.env` is gitignored. The key is read from the environment only, is never
printed, and never appears in a report or a run artifact.

**Which model?** The default is `anthropic/claude-sonnet-5` — it is reliable at
both the structured extraction in stage 2 and the code generation in stage 4,
and its large context window swallows a full paper. Run `repro models` to see
every available slug with live pricing, and switch by editing `OPENROUTER_MODEL`
or passing `--model`:

```bash
repro models --filter claude
repro run 1708.02596 --model openai/gpt-5.4
```

## Quickstart

```bash
# the whole pipeline on a real paper
repro run 1708.02596

# just the claim, no execution -- cheap, one model call
repro extract https://arxiv.org/abs/1708.02596

# re-read a report you already generated
repro show 1708.02596

# everything you have ever run, with verdicts
repro list

# available model slugs and prices
repro models --filter gpt
```

### `repro run` options

| flag | default | what it does |
| --- | --- | --- |
| `--tolerance` | `0.25` | Relative band around the paper's number that counts as reproduced. |
| `--seeds` | `0,1,2` | Comma-separated seeds. Each is a separate subprocess. |
| `--timeout` | `600` | Hard per-seed timeout in seconds. |
| `--model` | `$OPENROUTER_MODEL` | OpenRouter slug for this run only. |
| `--refresh` | off | Ignore the cache and re-fetch the paper. |

## The fixture demo — no network, no API key

This matters for demoing on bad wifi or with a dead key:

```bash
repro demo
```

It runs a shipped fixture end-to-end. Stages 1, 2 and 4 are replaced by files in
`tests/fixtures/`; **stages 5, 6 and 7 run for real** — it really collects
transitions, really fits a dynamics model, really runs MPC, and really compares
the measured number against the fixture's. Takes about 15 seconds.

```bash
repro demo                        # the testable fixture -> REPRODUCED
repro demo --case untestable      # the UNTESTABLE triage path
repro demo --tolerance 0.05       # the same run, tighter band -> NOT_REPRODUCED
repro demo --seeds 0,1,2          # all three seeds (~45s)
```

The fixtures are **synthetic papers**, not real ones, so the demo never implies a
verdict about anybody's published work. See `tests/fixtures/README.md` for how
the fixture's reported number was measured.

## The harness

Generated scripts stay small because they import primitives from
`src/repro/harness.py`:

```python
from repro.harness import collect, ForwardModel, mpc_control, perturb, sample_efficiency_curve

data  = collect("Pendulum-v1", 2000, policy="random", seed=0)  # -> (s, a, s') arrays
model = ForwardModel.for_env(make_env("Pendulum-v1"), seed=0)  # small MLP, predicts the delta
model.fit(data, epochs=100, seed=0)
model.one_step_error(holdout)                                  # mean squared prediction error

ret = mpc_control("Pendulum-v1", model, horizon=15, n_candidates=500, episodes=5, seed=0)
perturb(env, "mass", 1.5)                                      # mass / damping / friction
sample_efficiency_curve("Pendulum-v1", [500, 1000, 2000, 4000])
```

Dependency-light (gymnasium, torch, numpy) and classic-control only —
`Pendulum-v1`, `CartPole-v1`, `MountainCarContinuous-v0`, `Acrobot-v1`.

`perturb` raises rather than silently doing nothing if the environment has no
matching parameter. A perturbation experiment that quietly did not perturb
anything would be worse than no experiment.

## Tests

```bash
pytest                  # ~25s, no network
pytest -m "not slow"    # skip the two full-execution tests
```

191 tests covering claim JSON parsing (clean, fenced, prose-wrapped, malformed,
lossy), the tolerance and verdict logic at its boundaries, harness shapes and
determinism, the `UNTESTABLE` triage path, the OpenRouter 401/402/429 branches
with the HTTP layer mocked, subprocess execution and timeouts, and the report's
mandatory assumptions section. **No test touches the network.**

## Limitations

Read this before you quote a verdict at anyone.

- **A `NOT_REPRODUCED` verdict may reflect the agent's assumptions, not the
  paper.** Every value the paper left unstated was chosen by a language model,
  and any one of those choices can move the number more than the effect being
  tested. That is exactly why the ASSUMPTIONS section is mandatory and why it is
  printed next to the verdict rather than buried. Read it first. If the
  assumptions look wrong, the verdict is about them.

- **Results vary by model slug.** A different `OPENROUTER_MODEL` will extract a
  different claim, write a different script, and may well reach a different
  verdict on the same paper. The slug is recorded in every report for this
  reason. A verdict without its model slug is not a result.

- **Classic-control substitution is not a faithful reproduction of
  MuJoCo-scale results.** When a paper reports on HalfCheetah or Ant and the
  agent substitutes Pendulum-v1, it is testing whether the *mechanism* shows up
  at toy scale — not whether the paper's number is right. Those are very
  different questions. The substitution is declared as an `ASSUMPTION:` line,
  and it should usually be read as "this claim was not really tested."

- **Compute budgets are laptop-sized.** Generated scripts are capped at a few
  minutes per seed. A method that needs a long training run will look worse than
  it is.

- **One claim per paper.** The agent extracts the single headline claim. A paper
  whose contribution is a suite of results is not well served by one number.

- **Three seeds is not a statistical result.** The default `0,1,2` gives you a
  mean and a rough spread, not a significance test.

- **The tolerance band is a convention, not a standard.** 25% is a default, not
  a principled threshold. Set it deliberately with `--tolerance`.

- **An `UNTESTABLE` verdict is about the paper's text, not its quality.** Many
  excellent papers are underspecified in their PDF and fully specified in their
  released code. repro reads the PDF.

## Design notes

- **One module talks to the model.** `src/repro/llm.py` exposes a single
  `complete(system, user, json_mode=True) -> str`. It handles 401 (bad key),
  402 (out of credits) and 429 (rate limit, with exponential backoff) as
  distinct, human-readable errors. Nothing else in the codebase makes an HTTP
  call to a model.
- **JSON mode is a hint, not a contract.** Support varies by model on
  OpenRouter, so the parser strips fences, recovers the outermost object, and
  retries once — and the pipeline degrades gracefully when a provider rejects
  `response_format` outright.
- **The model slug is never hardcoded at a call site.** It comes from
  `OPENROUTER_MODEL` (or `--model`) with one default in `config.py`.
- **Generated code never runs in-process.** Always a subprocess, always with a
  hard timeout, always with its artifacts on disk.
- **Nothing is silently fabricated.** A script that exits 0 without printing
  `RESULT:` is a failure, not a zero. A run that produces a number the paper
  never stated is `RUN_FAILED`, not `REPRODUCED`.

## License

MIT.
