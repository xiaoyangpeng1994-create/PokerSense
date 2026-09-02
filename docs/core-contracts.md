# Core Contracts

Poker Intelligence Engine Phase 0 core domain objects. All objects are
immutable (deep immutability) and JSON-serializable via the serializer
(see `docs/serialization.md`).

> This describes the FROZEN contracts as implemented in Task 1B ~ 1D.

---

## Money

| Object | Semantics |
|---|---|
| `ChipAmount` | Strictly non-negative money value (stack/pot/bet). Backed by `decimal.Decimal`. |
| `ChipDelta` | Signed money change (net_result/profit_loss/chip_change). Backed by `decimal.Decimal`. |

Rules:
- Money is NEVER a `float`.
- `ChipAmount` rejects negative values.
- `ChipDelta` may be negative.
- Arithmetic: `ChipAmount - ChipAmount -> ChipDelta`; `ChipAmount + ChipDelta -> ChipAmount` (raises `InvalidStateError` if result < 0).

---

## Observation layer

| Object | Semantics |
|---|---|
| `ObservationField[T]` | A single recognized value + `confidence` + `source` + `evidence` + `validation_status`. Generic. |
| `RawObservation` | One frame's raw observation of the table. |

Key facts:
- `RawObservation` is **observed data** (from Vision), NOT authoritative state.
- `ObservationField.value` may be `None` when `validation_status == UNKNOWN`.
- `evidence` is deep-frozen (read-only Mapping).

---

## State layer

| Object | Semantics |
|---|---|
| `PlayerState` | One player's state in the current hand. |
| `PokerState` | The system's authoritative, immutable game state. |
| `StateContext` | Read-only context prepared by the Orchestrator for the State Engine. |
| `ValidationResult` | Result of validating a candidate PokerState. |
| `StateEvent` | A single event in the append-only event stream. |

Key facts:
- `RawObservation != PokerState`: observed data vs authoritative state.
- `PokerState` only protects structural correctness; it does NOT infer what happened.
- `PlayerState` stores no `last_action` (that is expressed by `StateEvent`).
- `StateEvent != ActionType`: `ActionType` is the player-action vocabulary; `EventType` (in `StateEvent.event_type`) is the event-stream vocabulary (which also has lifecycle events like `HAND_START` / `DEAL` / `STREET_CHANGE` / `HAND_END`).

---

## Async

| Object | Semantics |
|---|---|
| `RequestContext` | Ties a Slow Path request to a specific hand/state_version. |

Used later for stale-result protection.

---

## Hand

| Object | Semantics |
|---|---|
| `HandSummary` | Settlement summary (final_pot, winners, winnings, net_result). |
| `HandHistory` | Complete append-only record of one hand. |

Rules:
- `winnings`: player_id -> `ChipAmount`.
- `net_result`: player_id -> `ChipDelta`.

---

## Reports

| Object | Semantics |
|---|---|
| `EquityReport` | Equity / pot-odds result. |
| `StrategyReport` | Strategy recommendation from a strategy tier. |
| `ReasoningReport` | Poker Reasoning Layer output. |
| `Decision` | Final decision from the Decision Engine (single exit point). |

Key facts:
- `ReasoningReport` does NOT store a hidden chain-of-thought — only an auditable structured summary.
- `Decision` is the only object produced by the Decision Engine.

---

## Opponent

| Object | Semantics |
|---|---|
| `OpponentProfile` | Aggregate behavioral statistics for a player across hands. |

Frequency fields (`vpip`/`pfr`/`cbet_freq`/`threebet_freq`/`bluff_freq`) are ratios in `[0, 1]`; `af` (aggression factor) is non-negative with no upper bound.

---

## Strategy target contracts

Module: `poker_engine.strategy`

The target architecture adds immutable, multi-player contracts without
changing the released `PokerState`/`Decision` wire behavior:

| Object | Semantics |
|---|---|
| `GameConfig` / `DecisionSeat` / `PotState` / `LegalAction` | Complete game, seat, pot, and legal-action input |
| `InputProvenance` / `ContextQuality` | Field source, quality, evidence, and hard decision gates |
| `RangeDistribution` | Versioned per-seat inferred range; not an observed fact |
| `DecisionContext` | Hand/state/request-bound 2–9-player strategy input |
| `ProviderCapability` / `StrategyCandidate` | Explicit applicability and versioned Provider output |
| `MatchDimension` | Structured abstraction difference with requested/matched values, distance, approved maximum, and derived score; propagated Provider → Candidate → Advice → UI |
| `GateResult` | Named PASS/FAIL/SKIPPED decision gate; FAIL requires stable reason codes and prevents READY Advice |
| `ActionOption` | One action/total-street size branch with its own exact Decimal frequency and source label |
| `StrategyRouter` | Exact player-count/street/stack/action-line filtering and result selection within one source layer |
| `TieredStrategyRouter` | Ordered Cache → Preflop DB → Presolved → Model fallback; stops after the first layer with a usable candidate |
| `JsonStrategyAssetProvider` | Read-only, SHA-pinned strategy-node adapter with capability digest, license/version evidence, and fail-closed node parsing |
| `Advice` | The only target strategy output: READY/PARTIAL/ABSTAIN/STALE |
| `DesktopFrame` | Atomic analysis + optional Advice UI update; hand/state mismatch fails closed to STALE |
| `TemporalConsensus` / `TemporalConsensusResult` | Per-field and per-visual-slot consecutive-frame confirmation; pending values are UNKNOWN and CONFLICT remains explicit |
| `HandBoundaryPolicy` / `HandBoundaryDetection` | Deterministic SAME_HAND/CONFIRMED/AMBIGUOUS classification; dealer and stack reset evidence require explicit slot-to-seat mappings |
| `HandMemory.record_transition` / `replace_active_hand` | Prevalidated atomic commits for state+events and completed-hand+successor lifecycle replacement; failures leave all streams and active identity unchanged |
| `PotCalculation` | Exact main/side-pot allocation plus unmatched-chip returns |
| `build_decision_context` | Validates request identity and derives seats, legal actions, pots, active count, and effective stacks from `PokerState` |
| `RangeUpdate` / `JointRangeAssignment` | Versioned Bayesian update metadata and collision-free multi-player concrete holdings |
| `PotEquity` / `MultiwayEquityResult` | Exact weighted Hero share for each main/side pot and the total contested pot |
| `PairwiseSpr` / `PotOddsMetric` / `NormalizedActionSize` | Decimal-derived SPR, immediate break-even equity, and explicit BB/pot/raise sizing |
| `EquityCacheQuery` / `EquityCacheEntry` | Canonical SHA-256 query identity and bounded TTL/LRU result storage with CI/provenance |
| `AdaptiveEquityReport` | Deadline-budgeted exact or seeded Monte Carlo result with COMPLETE/PARTIAL and numerical confidence |
| `HuPreflopBlueprintProvider` | Optional, integrity-checked HU preflop Adapter with source commit, manifest, shard, exact event token, and 169-class evidence |
| `PreflopRfiHeuristicProvider` | Bundled 6/9-handed unopened RFI fallback with pinned MIT source, asset hash, 169-class coverage, explicit HEURISTIC label, and no invented size/EV |
| `StrategyCycle` / `SlowHandle` / `SlowRefinement` | Immediate Fast Advice plus async result identity, PENDING/APPLIED/NO_UPDATE/DISCARDED/FAILED lifecycle |
| `ActualActionRecord` / `HandDebrief` / `HandReview` | Exact Advice identity binding, action/size deviation, per-decision EV loss, and conservative whole-hand totals/completeness/max-loss aggregation without pairing missing actions |
| `AdviceExplanation` | Deterministic zh/en rendering of unchanged Advice values and reason codes |
| `ActionEvEstimate` / `EvGapResult` | Exact immediate/branch EV with explicit UNKNOWN and complete-legal-action gap gating |
| `ContextQualityPolicy` / `RequestContextFactory` | Required-field minimum confidence gates and thread-safe unique expiring request creation |
| `ConfidenceAggregate` | Named perception/state/match/range/numerical factors; minimum wins and missing required factors force zero |
| `ExploitAdjustmentPolicy` / `ExploitAdjustment` | Sample/quality/weight/logit/KL-gated exponential tilt that preserves baseline support and returns HEURISTIC metadata or the unchanged baseline |
| `DecisionFusion` / `FusionOutcome` | Keeps the Router's one selected baseline, optionally applies one auditable opponent adjustment, and builds Advice without blending Provider abstractions |
| `EvidenceAuditPolicy` / `EvidenceAudit` | Deterministic input/state/range/provider evidence references, SHA-256 chain identity, named missing links, and an incomplete-chain confidence cap |
| `RangePriorQuery` / `PreflopRfiRangePrior` | Capability-bounded 6/9-handed first-in-raise lookup that expands reviewed 169 classes to blocker-safe concrete combos and returns UNKNOWN instead of random fallback |
| `LocalResolverConfig` / `LocalResolverProvider` | No-shell JSON subprocess protocol with capability checks, request deadline, timeout/output bounds, identity/version/convergence validation, and fail-closed ProviderResult mapping |
| `StrategyCacheQuery` / `StrategyTemplate` / `CachingStrategyProvider` | Canonical provider/asset/engine key and identity-free cached strategy safely rebound to the current request |

The implemented slices provide contracts, deterministic state derivation, and
a FakeProvider regression loop. The optional HU Blueprint Adapter adds real
100BB/no-ante Golden-backed root and `b300` paths without bundling the upstream
solver or assets. A separate bundled PreflopR fallback covers only explicit
6/9-handed, 100BB/no-ante/no-rake unopened raise/fold charts. It is always
heuristic, excludes upstream player-count and BB fallbacks, and does not change
the released desktop behavior.
`LegalAction.amount_semantics` distinguishes
zero-cost actions, additional chips, and total-street bet/raise targets. They do
not yet add an exact multi-player strategy asset or change the current desktop UI. Range blocker
and joint-assignment functions intentionally require concrete combos such as
`AsQc`; abstract classes such as `AKs` must first be expanded by a versioned
range asset.

`calculate_side_pots(..., settle_uncalled=False)` is used while betting is
open: active players can still match and contest a leading commitment, so that
tranche remains provisional. Unmatched chips are returned only when betting is
settled; this prevents normal unequal SB/BB commitments from being rejected.

---

## Immutability & time conventions

- All objects use `@dataclass(frozen=True)` + deep immutability (`list -> tuple`, `dict -> MappingProxyType`, `set -> frozenset`).
- All timestamps are timezone-aware `datetime` (naive is rejected).
- All money is `ChipAmount` / `ChipDelta` (never float).
