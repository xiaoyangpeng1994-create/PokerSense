# Confidence Gate

字段级置信度门控。低置信度的关键字段被降级为 `UNKNOWN`（value=None），从而不会作为可靠的 canonical data 进入 State Engine。

## 职责

`RawObservation → Confidence Gate → Sanitized RawObservation`。逐字段清洗，**不**整体丢弃 observation。

## Frozen thresholds（architecture v0.2.1）

| 字段 | threshold |
|---|---|
| hero_cards | 0.995 |
| board_cards | 0.995 |
| street | 0.999 |
| pot | 0.99 |
| stacks | 0.99 |
| bet_size | 0.99 |
| action | 0.99 |

边界规则：`confidence == threshold` → PASS；`confidence < threshold` → BLOCK（不用 `>`）。

## 未定义 threshold 的字段

`actor` / `dealer_pos` / `overall_confidence` 无 threshold → **不做 numeric gating**，保持原值。未来由 Vision/Platform Contract 决定。

## 字段转换规则

- `VALID` 且 `confidence >= threshold` → 原样保留
- `VALID` 且 `confidence < threshold` → `value=None, status=UNKNOWN`（保留 confidence/source/evidence/timestamp）
- `LOW_CONFIDENCE` → 一律 `UNKNOWN`（不因 numeric 高而升级）
- `UNKNOWN` → 保持 UNKNOWN
- `CONFLICT` → 保持 CONFLICT（不偷换成 UNKNOWN）

## 不变量

- 纯函数、deterministic，绝不修改输入对象（返回新对象）。
- thresholds 固定顺序 + defensive copy + immutable 暴露。
- 自定义 thresholds 必须完整含 7 字段，未知字段/非法值（bool/NaN/inf/<0/>1）fail fast。

## 集成

`ApplicationOrchestrator.process_observation` 内部先 gate，再 build `StateContext`（填充实际 thresholds），再 StateEngine transition。

```python
ApplicationOrchestrator(state_engine, hand_memory, confidence_gate=None)
```

## 不实现

resampling / retry / frame waiting / Decision blocking / 整体拒绝。
