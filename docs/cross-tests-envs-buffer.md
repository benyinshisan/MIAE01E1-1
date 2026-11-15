# Issue #10 — independent environment and buffer verification

Author: 2260031814 陈博文, who implemented neither Issue #3 nor Issue #4.
Base: main `11a1e4a`. No production code is changed by this test branch.

## Acceptance coverage

| Issue requirement | Evidence |
|---|---|
| Pong observations `(4,84,84)` uint8, six actions | Actual ALE ROM, both train/eval factories, space membership and factory metadata |
| Four ALE frames per decision | `ale.getFrameNumber()` before/after five nonterminal steps in each mode |
| A test catches wrong frame advance | Negative control injects ALE frameskip=4; the same frame assertion rejects measured **16**, not 4 |
| Reproducible seeded resets | Fresh train/eval instances with seed 1 reproduce the first screen, reset frame count and next 20 fixed-action observations |
| Different seeded initial conditions | Seeds 1–5 produce different reset frame counts and subsequent trajectories; see clarification below |
| Life loss ends training, not evaluation | Deterministic life-counter environment passed through the actual factory wrapper chains; true game-over still ends evaluation |
| Full buffer overwrites oldest; writer slot excluded | Public-API batches at/beyond wrap contain only expected recent complete transitions, never the successor-only slot |
| Continuous `(obs,next_obs)` alignment | Time-encoded frames, actions and rewards checked before fill, at first wrap, and after repeated wraps |
| Insufficient samples raise | Empty and partially filled buffers raise ValueError rather than return a smaller batch |

Additional checks cover last-two-frame max pooling, reward summation, terminal
flags, stable returned batches after later overwrites, and reproducible sampling.
All buffers in tests are small; no production-sized allocation is needed.

## Clarification of “different seed, different first frame”

The Issue's wording is too strong for a static reset screen. With actual
ALE/Pong-v5 and the default wrappers, seeds 1, 2, 3, 4, 5 produced reset episode
frame counts **13, 27, 33, 9, 37**, respectively. Their initial pixels were
identical, but with subsequent fixed NOOP actions the first divergence from seed
1 occurred at decision **8, 7, 12, 6**, respectively.

The test therefore verifies seeded initial conditions using **both** frame
counts and a short fixed-action trajectory. It still requires exact first-frame
and trajectory equality for repeated use of the same seed. A finite number of
no-op choices can also collide across distinct seeds; no arbitrary pairwise
inequality is assumed. This is an explicit proposed interpretation of the
acceptance criterion for team review, not a change to the environment's behavior.

## Test scope

The four-frame assertion uses nonterminal actions; early termination inside an
action-repeat block can legitimately advance fewer frames. Pong itself has no
life counter, so life-loss behavior is tested with a deterministic Gymnasium
environment through the actual factory, not by pretending Pong loses lives.

The buffer contract stores one observation array and reserves one successor
slot after wrap. Sampling is with replacement, and `len(buffer)` retains its
existing capacity-based public semantics. Continuous transitions are checked
here. Trainer reset/truncation integration and Agent bootstrapping are later
integration tasks, not established by these tests.

## Reproduction and result

Run `python -m pytest -q` with repository runtime dependencies, PyTorch and
pytest installed. Actual Atari checks are mandatory; there are no dependency
skips or expected-failure markers masking failures.

Verified on Windows, 2026-11-15: **21 passed** (18 new checks plus 3 existing
tests), Python 3.12.16, Gymnasium 1.0.0, ALE 0.10.1, NumPy 2.5.4, OpenCV 5.0.0,
PyTorch 2.4.1+cu121, pytest 9.2.0. Initial strict reset-pixel tests failed as
described above; the criterion was clarified using the measured evidence.
