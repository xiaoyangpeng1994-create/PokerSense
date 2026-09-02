# ADR-002 — Visual Slot Observation Semantics（v3）

- 状态：**ACCEPTED / IMPLEMENTED（Task 7A 完成，待 External re-FROZEN）**
- 日期：2026-08-19（v3）
- 影响范围：Frozen Core（additive）、Serialization（Task 1E）、Confidence Gate（Task 5）、State Engine（Task 3，7A 不消费）、Opponent Model（未来）

---

## 决策

**Option B — additive slot-aware observation structure**（ACCEPTED）。

- 新增 `SlotObservation[T]`。
- `RawObservation` 新增 `slot_stacks` / `slot_actions`（additive，默认空 tuple）。
- 保留现有 `stacks` / `action`，不删不改。
- `slot_id` = visual geometry only，NOT player identity。

---

## 契约（v3 最终定稿）

### SlotObservation[T]

```python
@dataclass(frozen=True)
class SlotObservation(Generic[T]):
    slot_id: int
    field: ObservationField[T]
```

不变量：
- `slot_id` int、非 bool、`>= 0`。
- `field` 是 `ObservationField`。
- frozen + deep immutable。

### RawObservation 新增 additive 字段

```python
slot_stacks: tuple[SlotObservation[ChipAmount], ...] = ()
slot_actions: tuple[SlotObservation[ActionType], ...] = ()
```

- `slot_stacks` / `slot_actions` 是 additive 默认字段，**放在 `overall_confidence` 之后**（两者都有默认值，dataclass 合法）。
- **原因**：保留历史 `RawObservation` 位置构造兼容——历史 trailing positional 参数（`actor` 之后的那个）必须继续绑定到 `overall_confidence`。
- 每个 tuple 内 `slot_id` 唯一、严格递增。
- tuple 化 + deep-freeze。

---

## Serialization（Blocker 1 FINAL DECISION）

**全局 `schema_version` REMAINS 1**。

- 不引入 per-type schema versioning，不 bump 全局版本。
- `SlotObservation` 扩展是 additive：`slot_stacks` / `slot_actions` 默认空 tuple。

**Migration contract**：
- OLD `RawObservation` payload（无 `slot_stacks` / `slot_actions`）→ 新 deserializer 必须产出 `slot_stacks=()`、`slot_actions=()`。
- NEW `RawObservation` serialization 明确写出 `slot_stacks` / `slot_actions`。

---

## Confidence Gate（Blocker 2 FINAL DECISION）

- `slot_stacks[i].field` 走 `stacks` threshold（0.99）。
- `slot_actions[i].field` 走 `action` threshold（0.99）。

**blocked_fields path 编码视觉 slot_id，不是 tuple index**：

```
slot_stacks[slot_id=2]
slot_actions[slot_id=5]
```

（bracket 值 == `SlotObservation.slot_id`，**不是** tuple 位置。）

blocked_fields ordering：
1. 现有原始字段（固定顺序）
2. `slot_stacks` 按 slot_id 升序
3. `slot_actions` 按 slot_id 升序

**只有真正被 Confidence Gate demote 的 field 才进入 blocked_fields**；`UNKNOWN` / `CONFLICT` 原样通过时**不得**错误记为 gate-blocked。

---

## State Engine

7A 不消费 `slot_*`，Task 3 行为不变。

---

## Migration 风险

低：纯 additive；schema_version 不 bump，旧 payload 兼容（默认空 tuple）。

---

## 待办（7A 内完成）

- [x] `SlotObservation[T]` 契约 + invariants
- [x] `RawObservation` additive 字段
- [x] serializer 支持（schema_version 保持 1）+ migration 测试
- [x] Confidence Gate 逐 slot + deterministic blocked path（`slot_stacks[slot_id=N]`）
- [x] 测试 + docs
