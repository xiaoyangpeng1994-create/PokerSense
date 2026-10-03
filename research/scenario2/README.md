# Frozen scenario 2 source handoff

This directory is synthetic offline research source. It emits no live advice
and does not qualify a production policy. The four original commits ending at
`38a97da7f550047a12bfacd2e8174b1c225a0419` remain unchanged ancestors.

The frozen scenario uses board `2c 5d 9h Js`, P1 `As Ad` / `Tc Td`, P2
`Kh Kd` / `8c 8d`, four equal joint deals and 44 conditional rivers per joint.
All seats, stacks, prefix, rules and the C/B2/B4 action abstraction come from
the pinned constructor/config. Expected tree: 29 chance, 2848 decision, 4776
terminal nodes; 1488 information sets. `VISIBLE-CATALOG2.json` is the original
readable catalog, copied byte-for-byte; its SHA-256 is
`0732a79b410d138699678088a64d7b5e7925b59185d33c0d994a89e013648e06`.
The S11 JSONs are historical frozen evidence, not new checks performed here.
The historical pickle tree is excluded; its hash is only a historical reference.
The scoped `.gitattributes` disables text normalization in this directory so
Git preserves the frozen source/evidence byte hashes across platforms.

## Included sources

- `scenario2_factory.py`: the actual scenario constants/setup in a lazy portable
  source-hash-guarded wrapper. `load_game(repo_root)` loads the namespace; it
  does not construct an arena or tree on module import.
- `own_reach_collector.py`: the exact collector, unchanged bytes.
- `paired_observer.py`: AST-equivalent original `PairedObserver` and
  `mean_policy`; no new learning algorithm.
- `run_ab.py`: a portable closure binding around the original snapshot,
  `DualObserver`, coverage, exact BR scoring, zero-step restore and per-seed
  run/save function bodies. `SOURCE-MAP.json` identifies every extraction,
  original file/snippet hash and transformation. It explicitly separates
  extracted logic from the new path/dependency binding.
- `AVERAGE-SOURCE-EXCERPTS.json`: original source excerpts and definitions.

A remains `sum_t t * pi_owner[I] * pre_update_sigma[I,a]`; B remains
`sum_t pi_owner[I] * pre_update_sigma[I,a]`. Both commit after a successful
whole native sweep. Saved missing/zero/invalid rows stay `NOT_SCORED`; no
fallback fills an average. The quality threshold remains `1/20` BB. The native
regret-matching snapshot retains its original unvisited-regret convention.

## Safe source check

From the checkout root, using an existing compatible Python environment:

```sh
python -B research/scenario2/run_ab.py --verify-source-only
```

This command hashes files only. CLI execution has no train/resume mode and
requires the explicit source-check flag. Module imports construct no game,
tree, solver or policy and write no checkpoints.

## Cloud computation boundary

Only after a separately approved cloud protocol and budget ledger is frozen,
`create_runtime(repo_root, output_dir, total_budget_seconds=...,
global_save_reserve_seconds=..., seed_save_reserve_seconds=...,
setup_deadline=...)` can reconstruct the original compact tree and bind the
extracted callables. Output must be a fresh directory outside the checkout.
The returned `run_seed(seed, restore, deadline_abs, used_before)` has its
original cumulative-time argument, fixed 8192/16384 save points, native/equal/
A/B serialization and evaluation. It does not supply a new batch scheduler:
the cloud caller must enforce all approved seed/global caps, carry previously
spent time forward, reserve evaluation/save/cleanup time, and prevent retries
or seed selection. Source packaging grants no cloud compute approval.

Dependencies are the repository's existing declarations: compatible Python
3.11-3.13 and pinned `pokerkit==0.7.5` for the native game. No environment,
installation or extra dependency is bundled or performed by this handoff.

**Historical continuation is not available from this checkout alone.** Native,
equal, A/B checkpoints and bound resume receipts are deliberately excluded.
The extracted historical restore/save formats are not a new authenticated
resume protocol. Do not call the restore path without a separately verified
complete state/provenance chain and remaining budget; do not reset old budgets.
Seed `2026100501` stopped at 16168 and its average evidence remains
`QUARANTINED/NOT_SCORED`. Seeds `2026100502` and `2026100503` reached 16384
and their A/B quality results remain HOLD. Do not replay or promote that batch.

## Validation limits

This publication checks exact input bytes, Python syntax, AST equivalence of
extracted logic and lightweight module imports only. No `create_runtime`,
tree construction, training, BR scoring, checkpoint restore/save, full suite,
live game or capture was run. Runtime behavior of the new binding remains
`NOT_RUN`; source equivalence does not establish game-wide qualification.
The repository's production/default/learning files are unchanged.
