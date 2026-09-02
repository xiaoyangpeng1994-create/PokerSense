# ADR-001 — Phase 0 Scenario Acceptance Rescheduling

- 状态：Accepted
- 日期：2026-08-19
- 影响范围：Phase 0 验收口径（Task 4 Orchestrator）

## 背景

`architecture.md v0.2.1` 中，Task 4 的验收标准曾表述为「所有 Phase 0 规定场景全部通过」（含 16 个 scenario）。但 Task 3 State Engine 已封板，出于「不猜语义」原则，明确**不解析** `actor / dealer / stacks / bet_size / action` 的语义，不产生 action events、不更新 player committed/stack/status。

这导致 16 个 Phase 0 场景中，涉及玩家身份与下注语义的场景（Single Raised / 3Bet / 4Bet / Limp / Fold / All-in）**在当前 contract 边界下无法被正确建模**——不是 Orchestrator 实现缺陷，而是底层语义未定义。

## 决策

Task 4 本轮验收调整为以下**可现实完成**的集合：

- Orchestrator lifecycle（start_hand / process_observation / complete_hand）
- internal StateContext building
- state-first / event-second persistence
- no-op transition
- invalid transition
- deterministic persistence
- duplicate observation
- street / board progression
- pot update / pot regression
- card / street conflict
- HandMemory integration

## 延期（不取消）

以下场景延期到对应未来阶段，**不做 cancell**：

| 场景 | 延期到 |
|---|---|
| Single Raised Pot action modelling | TableMap + StateEngine v2 |
| 3Bet / 4Bet / Limp | TableMap + StateEngine v2 |
| Fold lifecycle | TableMap + StateEngine v2 |
| All-in | Stack/Committed 语义层 |
| automatic new-hand detection | Vision / TableMap |
| full event-sourcing replay | Replay Engine |
| async stale-result filtering | Slow Path Orchestrator（Phase 3） |

## 后果

- 不修改 Frozen Core Contracts（`RawObservation/PokerState/StateEvent/...` 均不变）。
- Phase 0 Feasibility Matrix 中 `Missing Frames` 从 SUPPORTED 改为 **PARTIALLY SUPPORTED**（仅容忍合法跳帧，不具备真正 temporal completion / 时序补全）。
- Replay assessment 保持：当前 HandMemory = recorded state/event readback，**不是** full Event Sourcing Replay。
