# STRATEGY-001 失败复现与归因

任务：STRATEGY-001（Issue #23）· run_id `STRATEGY-001-20260915T233757Z-r2`
固定基线：`dc72a4176f81689e95a9238a49fb2d268dcd4a56`

本文件的每个数字都来自**本轮实跑**，不复制历史报告值。

---

## 1. 复现环境与命令（可复跑）

```powershell
$env:PYTHONPATH='src;.'
$env:PYTHONUTF8='1'
& 'C:/Users/Administrator/.codex/runtimes/pokersense-v6-clean-20260908/Scripts/python.exe' tools/validate_threeway_models.py `
  --input   configs/strategy/examples/threeway-river-response-manual.json `
  --protocol configs/strategy/examples/threeway-validation-protocol-v1.json `
  --output  G:/PokerSense_private/strategy001_repro_20260916_v1/validate-v1

& '...python.exe' tools/study_opponent_uncertainty.py `
  --input  configs/strategy/examples/threeway-river-response-manual.json `
  --output G:/PokerSense_private/strategy001_repro_20260916_v1/uncertainty-v1
```

| 项 | 值 |
| --- | --- |
| repo HEAD | `dc72a4176f81689e95a9238a49fb2d268dcd4a56` |
| `tools/validate_threeway_models.py` sha256 前 16 | `9230cddb262ea7bb` |
| `src/.../threeway_policy_evaluation_v1.py` sha256 前 16 | `75b9af69582f4d29` |
| Python | 3.13.14（`pokersense-v6-clean-20260908`） |
| 依赖实况 | numpy **2.3.5** / opencv **4.10.0**（`pyproject.toml` pin：2.4.6 / 4.14.0.94 ⇒ **不一致**） |
| A 退出码 / 耗时 | **0** / **2 s** |
| B 退出码 / 耗时 | **2** / **5 s**（**设计内**：`tools/study_opponent_uncertainty.py:95` → `return 2 if any(... INSUFFICIENT_EVIDENCE)`，非崩溃） |

**A 复现结果**（与历史一致）：

```
world_cases=30  fixed_policy_evaluations=60  all_complete=true
fallback_cases=0            selected_id=calling_public_v1
screening_status=FAIL_STRESS_SCREEN
promotion_decision=NO_EMPIRICAL_PROMOTION
calibration_status=NOT_REAL_CALIBRATION
```

**B 复现结果**（与历史一致）：6 组全部保留，两组判定分列

| group | selected | status |
| --- | --- | --- |
| n6-facing_bet | `manual_reference` | `FAIL_DECLARED_WORLD_SCREEN` |
| n6-unopened | `null` | `INSUFFICIENT_EVIDENCE` |
| n7-facing_bet | `manual_reference` | `FAIL_DECLARED_WORLD_SCREEN` |
| n7-unopened | `null` | `INSUFFICIENT_EVIDENCE` |
| n8-facing_bet | `manual_reference` | `FAIL_DECLARED_WORLD_SCREEN` |
| n8-unopened | `null` | `INSUFFICIENT_EVIDENCE` |

**与历史基线的差异**：**无**。18/30 负比较、fallback=0、筛选状态全部一致。因此历史结论在本轮基线上成立，不需要版本/输入定位。

---

## 2. 逐场景条件 EV 与相对基线差（chips，净增量）

`selected` 指 log loss 选中的 `calling_public_v1` 策略本；`ref` 为原手工策略。完整 18 条：

| group | world | Δ vs ref | Δ vs check/fold | Δ vs check/call |
| --- | --- | ---: | ---: | ---: |
| n6-facing_bet | reference_control | −0.1607 | **+35.6915** | −0.1607 |
| n6-facing_bet | value_heavy | **−60** | −80 | **−60** |
| n6-facing_bet | rank_aware | −2.6244 | +26.8273 | −2.6244 |
| n6-unopened | value_heavy | 0 | −80 | −34.5222 |
| n6-unopened | rank_aware | 0 | +17.1458 | −1.9043 |
| n6-unopened | check_trap | 0 | +98.9676 | −4.9472 |
| n7-facing_bet | reference_control | −0.1607 | +39.5170 | −0.1607 |
| n7-facing_bet | value_heavy | −60 | −80 | −60 |
| n7-facing_bet | rank_aware | −2.5384 | +30.3003 | −2.5384 |
| n7-unopened | value_heavy | 0 | −80 | −34.5222 |
| n7-unopened | rank_aware | 0 | +20.0821 | −1.8131 |
| n7-unopened | check_trap | 0 | +106.7761 | −3.5498 |
| n8-facing_bet | reference_control | −0.1607 | +43.3424 | −0.1607 |
| n8-facing_bet | value_heavy | −60 | −80 | −60 |
| n8-facing_bet | rank_aware | −2.4524 | +33.7734 | −2.4524 |
| n8-unopened | value_heavy | 0 | −80 | −34.5222 |
| n8-unopened | rank_aware | 0 | +23.0183 | −1.7450 |
| n8-unopened | check_trap | 0 | +114.5846 | −1.9345 |

其它：`fallback_cases = 0`（**概率与次数均为 0**）；30 个世界全部 `complete`；无 `planning_errors`；
`conditional_river_EV only`，**不是**全手 bb/100，也**不**给统计置信区间。

---

## 3. 两个可在数据中直接观察到的结构性事实

### 事实 1：`ref` 与 `check/call` 在全部 9 个 facing_bet 世界中 EV **完全相同（精确分数相等）**

9 行 facing_bet 里 Δvs_ref 与 Δvs_check/call 逐位相同（例：n6 `-30444/189457` 两列一致）。
⇒ 在这些世界中，**三个比较对象实际只有两个是独立的**；「原手工策略」这一列不提供额外信息。

*可能原因（本轮未在代码中核实，故只列为假设）*：手写 reference 在 river 面对下注时恒为 call，因此在 EV 上与 check/call 基线重合。

### 事实 2：`ref` 与 `selected` 在全部 6 个 unopened 世界中 Δ=**0（精确相等）**

⇒ 未下注根上，log loss 选出的策略本与原手工策略本在本次评价中**没有产生差异**；该根的「新增挑战」并未真正区分两个候选。

**这两条事实的后果**：headline `18/30` 中有 9 行（9 个 facing_bet 负比较）是**同一比较的重复计数**。
即：当前筛选计数把「selected 劣于 check/call」同时记了一次「劣于 ref」。
这不改变「存在失败」这一结论，但**改变失败规模的可解释性**，必须在下一轮修正口径后再报数。

---

## 4. 失败分类 A / B / C / D

| 类别 | 本轮判定 | 依据 |
| --- | --- | --- |
| **A 实现 / 结算 / 信息集错误** | **未发现** | 30 世界全部 `complete`，`planning_errors` 为空，`fallback_cases = 0`，精确分数自洽（同一比较两列逐位相等）；无异常退出（A exit 0）。既有反例（QQ/TT/JJ）已在历史轮次修复并保留为回归。 |
| **B 对手范围 / 响应模型错设** | ✅ **主因** | ① `value_heavy` 世界 Δvs_check/fold = **−80**（两种根都是）：面对强价值范围时，不是最激进基线更好，而是**直接弃牌更优**，而所选策略没有充分弃牌。② `rank_aware` / `check_trap` 世界 Δvs_check/call 为 **−1.7 ~ −4.9**（check/fold 差值为大正数）：**过度跟注**。③ 选中候选由**合成 calling 生成器**产生 ⇒ 系统性偏好跟注。 |
| **C 动作网格 / 公开历史覆盖不足** | **本轮未触发** | `fallback_cases = 0`：没有任何世界因未覆盖历史而退化。零 fallback **不等于**能应对网格外动作（网格外下注仍不支持），但**不是本次 18 条的原因**。 |
| **D 桌规 / 字段 / 真实数据不足** | **长期缺口，非本次主因** | `opponent_dataset_v1` 的严格准入下，真实可准入记录 = **0**；straddle 子类型、抽水、封顶保持 **UNKNOWN**（未确认为零）；`calibration_status=NOT_REAL_CALIBRATION`。这些限制本轮**未**直接造成 18 条负比较，但决定了本轮结论无法升级为真实对手证据。 |

**总判定**：本轮失败 = **B 为主 + 比较口径缺陷（事实 1/2）干扰读数**；无 A；C 未触发；D 为长期约束。

---

## 5. 模块能力地图与「决定复用」清单

| 模块 | 能力 | 本轮是否复用 | 结论 |
| --- | --- | --- | --- |
| `threeway_policy_evaluation_v1.py` | 冻结策略评价、fallback 记录、精确 Fraction EV | ✅ 实跑 | **复用**；需补比较口径 |
| `response_model_calibration_v1.py` | 有限候选 log loss 选择、训练/验证隔离 | ✅ 间接（选择结果 `calling_public_v1`） | **复用**；不扩大模型族 |
| `robust_policy_selection_v1.py` | 最坏相对损失选择 | ✅ 间接 | **复用**；结论按 FAIL/INSUFFICIENT_EVIDENCE 表述 |
| `opponent_dataset_v1.py` | 真实数据准入闸门 | 未实跑 | **保持严格**；不用不足数据拟合 |
| `terminal_multiway_v1.py` | 终结 river all-in 条件 EV | 未实跑 | 保持边界，不与一般三人模块混淆 |
| `study_opponent_uncertainty.py`（工具） | 6 组多假设选择 | ✅ 实跑 | **复用**；exit 2 是设计信号需在文档中说明 |

**本轮没有新建任何 robust / 求解模块**，符合「不得再建一套同功能模块」。
