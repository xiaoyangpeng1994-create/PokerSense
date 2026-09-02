# Serialization

Centralized, explicit, reversible serializer for the Phase 0 core contracts.

Module: `poker_engine.core.serialization`

---

## Public API

```python
serialize(obj) -> dict
deserialize(type_, data) -> obj
```

- `serialize` accepts a single top-level domain object and returns a JSON-safe
  `dict` (only `dict` / `list` / `str` / `int` / `float` / `bool` / `None`).
- `deserialize` requires an explicit target type and recovers the domain object.

---

## schema_version

- Current version: `1`.
- The `"schema_version": 1` field appears **only** on the top-level result of
  `serialize()`.
- Nested domain objects carry `__type__` but **do not** repeat `schema_version`.
- `deserialize` fails fast if `schema_version` is missing or unsupported.

---

## Type tag

- Every domain object is tagged with a stable `__type__` string (e.g.
  `"PokerState"`, `"Card"`, `"ChipAmount"`).
- The tag uses a stable Domain Type Name, **not** a Python module path.
- `deserialize(type_, data)` checks that `type_` matches `data["__type__"]`;
  mismatch raises `SerializationError`.
- `__type__` and `schema_version` are reserved keys. A free mapping (e.g.
  `payload` / `evidence`) that contains one of these keys raises
  `SerializationError` (never silently interpreted/overwritten).

---

## Enum representation

- In a known position (e.g. `PokerState.street`), an enum serializes to its
  `.value` string and deserializes as `EnumType(value)`.
- In a generic position (e.g. `ObservationField.value`, a `payload` value), an
  enum serializes as a tagged object so its type is recoverable:
  ```json
  {"__type__": "Enum", "enum": "ActionType", "value": "raise"}
  ```
- Unknown enum values fail fast (`SerializationError`).

---

## Money representation

- `ChipAmount` / `ChipDelta` / `Decimal` are ALWAYS serialized as strings.
  ```json
  {"__type__": "ChipAmount", "value": "2.345"}
  ```
- No float, no `round()`, no `quantize()`, no fixed 0.01 precision.
- Deserialize: string -> Decimal -> ChipAmount/ChipDelta (exact value preserved).
- Invalid money strings / negative ChipAmount fail through the domain invariant.

---

## datetime representation

- All timezone-aware datetimes serialize as ISO-8601 (`datetime.isoformat()`).
- Deserialize via `datetime.fromisoformat()`; a naive datetime is rejected.
- No Unix float timestamps.

---

## Nested domain objects

- Every domain object — top-level OR nested — carries a stable `__type__` tag.
- Only the top-level result of `serialize()` also carries `schema_version`;
  nested objects do not.
- `deserialize` validates that a nested object's `__type__` matches the expected
  type before reconstructing it.

## frozenset representation

- `frozenset` / `set` serialize to a tagged object with deterministic ordering:
  ```json
  {"__type__": "FrozenSet", "items": ["a", "b", "c"]}
  ```
- Deserialize restores a `frozenset` (not a tuple).

## generic Decimal representation

- A bare `decimal.Decimal` in a generic position serializes as:
  ```json
  {"__type__": "Decimal", "value": "2.345"}
  ```
- Deserialize restores an exact `Decimal` (no float/round/quantize).
- `ChipAmount` / `ChipDelta` remain the money types; a bare `Decimal` is
  supported only in generic positions (not registered as a top-level type).

## unknown type tag

- In `_any_back`, a dict carrying a `__type__` tag MUST be recognized; an
  unknown tag raises `SerializationError`. A dict without `__type__` is treated
  as a plain mapping.
- A free mapping (payload/evidence) that itself contains a reserved serializer
  key is rejected on BOTH serialize and deserialize.

## float handling

- Generic non-finite floats (`NaN` / `±Infinity`) are rejected on serialize.
- `schema_version` must be a real `int` (a `bool` is rejected).

---

## tuple / list representation

- `tuple` and `list` serialize to JSON `list`; deserialize restores `tuple`
  through the formal constructor (which re-applies its invariants).

## Mapping representation

- Free mappings (e.g. `payload`, `evidence`, `platform_rules`) serialize to JSON
  `dict`; values are recursively serialized.
- On deserialize, the domain constructor re-applies `deep_freeze`, restoring the
  read-only immutable representation.

## ActionType-keyed mapping

- `StrategyReport.action_frequencies` (a `Mapping[ActionType, float]`) is
  serialized as a deterministic array, sorted by `ActionType.value`:
  ```json
  [
    {"action": "fold", "frequency": 0.2},
    {"action": "raise", "frequency": 0.8}
  ]
  ```
- Sorting guarantees stable JSON/log/replay/snapshot output.

---

## Error behavior

- All failures raise an exception (never silent fallback / `None`).
- `SerializationError` (subclass of `PokerEngineError`) is used for serializer
  protocol errors; domain invariants raise their own errors (e.g.
  `InvalidStateError`, `ValueError`, `TypeError`).

## Round-trip guarantee

```
obj -> serialize -> json.dumps -> json.loads -> deserialize -> equivalent obj
```
holds for every supported domain object, with money precision preserved exactly.

## Strategy schema v1

Target-architecture `DecisionContext` and `Advice` use the explicit
`poker_engine.strategy.serialization` module:

```python
strategy_serialize(value) -> dict
strategy_deserialize(type_, payload) -> DecisionContext | Advice
```

The strategy envelope has `schema_version=1` and a stable `type`. Money and
probabilities remain decimal strings. The existing core schema remains at v1;
legacy `RequestContext` payloads that omit the additive `expires_at` and
`deadline_ms` fields continue to deserialize with `None` defaults. Advice
`action_options` is also additive: new payloads preserve each source action and
total-street size frequency, while older schema-v1 Advice without the field
deserializes with an empty option tuple. `confidence_factors` and
`missing_confidence_factors` are additive audit fields; older payloads keep the
legacy scalar confidence with empty factor metadata. `evidence_chain_id`,
`evidence_complete`, and `missing_evidence` are also additive; older schema-v1
Advice loads without a structured chain while new payloads round-trip the
evidence audit result. `input_provenance` is an additive per-field source and
quality record used by the Live Coach UI to distinguish Vision, manual,
configuration, derived, and inferred inputs; older Advice defaults it to an
empty tuple. `match_dimensions` is an additive audit list for interpolated
strategy results; every item preserves the requested value, matched value,
distance, and allowed maximum distance. Older exact/heuristic Advice without
the field still loads with an empty tuple, while a newly constructed
`INTERPOLATED` candidate is invalid unless it supplies at least one dimension.
`gate_results` is another additive Advice audit list. New Advice records the
built-in and caller-supplied decision gates as `PASS`, `FAIL`, or `SKIPPED`;
older schema-v1 payloads deserialize with no gate audit. A `READY` Advice with
any failed gate is invalid even when loaded or reconstructed outside the
builder.

The desktop wire uses an atomic `DesktopFrame(RealtimeAnalysis, Advice?)`.
`StateSnapshot` carries `hand_id + state_version`; serialization converts a
mismatched or expired Advice to `STALE` and removes actions before it reaches
JavaScript. Legacy analysis-only streams remain accepted.
