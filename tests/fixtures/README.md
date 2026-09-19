# Fixtures

These are **synthetic papers**, not real ones. They exist so `repro demo` and the
test suite can run the whole pipeline with no network and no API key, and so the
demo never implies a verdict about a real group's published work.

- `demo_testable/` — a fully-specified claim about Pendulum-v1. `repro demo`
  runs it end-to-end: it really collects data, really fits a dynamics model,
  really runs MPC, and really compares the result to the fixture's number.
  The verdict is whatever the run produces.
- `demo_untestable/` — a vague claim that trips the triage stage, exercising the
  UNTESTABLE path.

Each fixture directory holds:

| file | what it is |
| --- | --- |
| `meta.json` | the paper record (id, title, authors, abstract) |
| `text.txt` | the paper body the extract stage would normally see |
| `claim.json` | the canned claim, standing in for a live extract call |
| `script.py` | the canned repro script, standing in for a live generate call (testable case only) |

## Where the testable fixture's number comes from

`demo_testable` claims a mean episode return of **-210** at 2000 transitions.
That is not invented: it is the value this harness actually produces, measured
over seeds 0, 1 and 2 (-195, -215, -218). Table 1 in `text.txt` was measured the
same way. The fixture is written this way so the demo exercises a real
comparison rather than a rigged one.

To watch the NOT_REPRODUCED path instead, tighten the band:

    repro demo --tolerance 0.05
