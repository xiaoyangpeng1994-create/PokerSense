# Application Orchestrator

中央编排器 / 系统唯一调度入口。负责协调 `RawObservation → StateEngine → HandMemory`，只决定「什么时候调用谁」，不实现任何扑克算法。

## 职责与边界

**负责**：模块调用、数据流、lifecycle、state-first/event-second 持久化、no-op/invalid 短路、determinism。

**不负责**：poker rules、state reconciliation、card validation、bet/stack 计算、actor identity mapping、street inference、equity、strategy、LLM、Vision、Confidence Gate 阈值。

## Public API

```python
class ApplicationOrchestrator:
    def __init__(self, state_engine, hand_memory) -> None
    def start_hand(initial_state, started_at=None) -> None
    def process_observation(observation) -> OrchestrationResult
    def complete_hand(hand_id, summary, ended_at=None) -> HandHistory

@dataclass(frozen=True)
class OrchestrationResult:
    transition: StateTransitionResult   # 不重复 state/events/validation/changed
    persisted: bool                     # 是否真实写入了 memory
```

## 关键行为

- `previous_state` 唯一来源：`HandMemory.latest_state(active_hand_id)`。`RawObservation` 无 `hand_id`，靠 `active_hand_id` 定位。
- `StateContext` 由 Orchestrator 内部构造（`context.previous_state == previous_state`）。
- 无 active hand 收到 observation → `OrchestratorError`（不 bootstrap、不 silent）。
- 持久化：state-first → event-second，固定顺序，禁止重排。
- no-op / invalid → 不写 memory（`persisted=False`）。
- determinism：不调用 `datetime.now/random/uuid`，时间全部来自显式输入。

## 写原子性（v1）

采用 fail-fast 策略（无 rollback/事务）。StateEngine 输出已被契约强约束（version 严格递增、event.state_version 引用已存在 state），正确使用下 record 不会失败；失败即 programmer/system error，上抛。

## Phase 0 场景边界

见 `docs/adr/ADR-001-phase0-scenario-rescheduling.md`。7 类需要身份/下注语义的场景（Single Raised/3Bet/4Bet/Limp/Fold/All-in/新 Hand）延期到 TableMap/StateEngine v2/Replay/Slow Path。

## 不做

bootstrap、action/bet 建模、hand boundary、async、stale filter、replay engine。
