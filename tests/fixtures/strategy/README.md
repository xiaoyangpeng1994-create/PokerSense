# Strategy mock fixture corpus

This directory contains deterministic synthetic data for the target
multi-scenario / multi-player strategy architecture.

## Files

- `v1/schema.json`: machine-readable fixture envelope schema.
- `v1/fixtures.jsonl`: one self-contained fixture per line.
- `v1/manifest.json`: count, SHA-256, coverage summary, and exclusions.
- `../../../tools/generate_strategy_mock_fixtures.py`: source generator and
  semantic validator.

The corpus currently contains more than 200 fixtures covering:

- 2–9-player preflop scenarios for every required action-line family.
- Exact player-count routing and HU-provider rejection for 3–9 players.
- Exact position matching plus bounded stack, pot, and last-aggressive-size
  abstraction, including combined dimensions and threshold rejection.
- Flop, turn, and river at every active-player count from 2 through 9.
- HU postflop reached from 6-player and 9-player preflop histories.
- Input `UNKNOWN`, `LOW_CONFIDENCE`, and `CONFLICT` for every critical field.
- Vision/manual/config/derived/inferred provenance, same-value consensus,
  cross-source conflict, null, and low-confidence source resolution.
- Stack and pot-odds boundaries, all-in and side-pot allocation.
- Canonical fold/check/call/bet/raise reconstruction, labeled and ambiguous
  all-ins, and inconsistent chip-delta rejection.
- Consecutive-frame temporal confirmation, changed candidates, frame gaps,
  UNKNOWN resets, preserved CONFLICT, and stable/missing visual slots.
- Confirmed, ambiguous, and rejected hand boundaries from hero-card changes,
  postflop resets, conflicts, and explicit versus absent slot-to-seat mappings.
- Atomic state/event commits and active-hand replacement, including invalid
  event identity and existing-successor rollback.
- State, context, range, provider, resolver, fusion, stale, and capture-binding
  failures.
- Structured interpolation differences preserved through resolver, cache,
  fusion, Advice serialization, and UI output.
- Built-in and caller-supplied hard gates, including all-pass, single-failure,
  multi-failure, audit serialization, and fail-closed Advice behavior.
- Ordered Cache, Preflop DB, Presolved, and Model fallback, including early
  stop, rejection fall-through, write-through refill, and all-source miss.
- Hash-pinned multiplayer strategy-asset intake for 3-player preflop, 3-way
  flop, and 4-way turn, plus missing node, malformed node, bad hash, license,
  version, and full capability-digest checks. These assets are synthetic only.
- Exact and Monte Carlo equity anchors.
- All Advice states: `READY`, `PARTIAL`, `ABSTAIN`, and `STALE`.
- Evidence-chain, debrief, and performance workload fixtures.

## Generate and verify

```bash
python3 tools/generate_strategy_mock_fixtures.py
python3 tools/generate_strategy_mock_fixtures.py --check
python3 -m pytest -q tests/strategy/test_mock_fixture_dataset.py
```

Never edit generated JSON/JSONL files directly. Change the generator, regenerate,
and review both the generator diff and manifest hash.

## Intended use

Every fixture identifies its product requirements, module function IDs, test
IDs, inputs, expected terminal stage, Advice contract, provider lookup state,
assertions, and numeric tolerances. Tests should select fixtures by ID or tag;
they should not depend on file order.

Money is encoded as decimal strings. Mock Provider results are deterministic
contract values and are explicitly marked `synthetic-only`.

## Acceptance boundary

This corpus can support contract, unit, property, integration, routing,
rejection, fault-injection, and workload-shape tests. It cannot replace:

- Golden parity exported from a real, licensed strategy Provider.
- Replay data captured from a real supported poker client.
- Performance measurements from the declared target hardware.
- Clean-install and live UI acceptance on macOS or Windows.

Those exclusions are recorded in `manifest.json`. Synthetic tests passing must
not be reported as those external acceptance gates passing.
