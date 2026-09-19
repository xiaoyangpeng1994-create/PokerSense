# AA 真实手牌确认 V1（TARGET-S）

本文件定义 **TARGET-S** 真实手牌确认合同，以及实现它的模块
`src/poker_engine/desktop/aa_real_hand_confirmation.py`。

一句话结论：**真实手牌的验收只认「人 + 绑定证据 + append-only 修订」这一条路。
机器候选、#31 开发标注、手工假设表单，全都不能产出验收 receipt。**

---

## 1. 为什么需要这一层

PHASE 8 的只读审计（Issue #27 `5742993909`）确认：

- `aa_semantics` 的 action `interpretation_status` 只有三个取值 ——
  `UNKNOWN` / `VISIBLE_LABEL_CANDIDATE` / `PRICE_DERIVED_CANDIDATE`，
  **从不产生 `CONFIRMED`**；
- 因此「第一手真实牌局验收」在当前架构里**构造上不可达**。

PHASE 8A（`5743174281`）据此裁定：新增一条人工确认通道，并明确规定
`DO_NOT_MUTATE_MACHINE_CANDIDATE` —— 机器输出保持原样，`CONFIRMED` 是
**另一层带 provenance 的派生 view**。

本文件是 TARGET-S（threeway 河牌 solver-ready）那一半的实现说明。
TARGET-P / PHH 全手明确延后。

---

## 2. TARGET-S 与 TARGET-P 的边界（务必分清）

| | TARGET-S（本轮实现） | TARGET-P / PHH（延后） |
| --- | --- | --- |
| 内核 | threeway 河牌决策文档 | PokerKit `NT` 全手 PHH |
| 起点 | **河牌起点**（本街投入全为 0） | **本手开局**（强制下注之前） |
| 筹码语义 | 河牌起点**剩余**筹码 + 本手**已投入** | **开局**筹码 + `antes` + `blinds_or_straddles` |
| 行动历史 | 河牌历史（P0 目标为空） | **从 preflop 起的全部**街道行动 |
| commitment 来源 | 逐座**聚合**已投入（stack delta 可确认） | 强制下注与逐街行动**明细** |

**`TARGET_S_READY` 不等于 `TARGET_P_PHH_READY`。**

原因：PHH 的 `starting_stacks` 是开局筹码，PokerKit 需要 `antes`、
`blinds_or_straddles` 与完整 action 序列才能重放整手牌。
stack delta 只给出**聚合值**，拆不出 ante / SB / BB / straddle / 逐街明细，
**不能冒充 PHH 的强制下注或 action**。

本轮**不修改** `aa_phh_shadow.py`，不加任何 PHH action bridge。
PHH 继续保持当前 confirmed-only gate。

---

## 3. 四层东西，严格区分

| 层 | 是什么 | owner | 能否成为真实手牌事实 | 能否成为训练 gold |
| --- | --- | --- | --- | --- |
| **A** 机器观测 / 候选 | `aa_semantics` 的三态候选 | 观测管线 | **否**（永不） | 否 |
| **B** 开发标注 | `tools.aa_annotation_records`（#31） | #31 | **否**（绝不自动） | 否 |
| **C** 人工确认真实手牌事实 | **本模块** | `aa_real_hand_confirmation` | **唯一权威** | **否**（绝不自动） |
| **D** holdout gold / 训练真值 | 预留 + freeze adapter | `aa_holdout_plan` | — | 需独立 freeze 动作，当前**不可达** |

- A 只以「被绑定的证据摘要」身份被 C 引用，C 的判断不由 A 推导；
- B 的 annotation row **不得**被 C 读取（有测试断言）；
- C → D 无自动路径。

### 旧手工假设路径必须保留，但不是验收权威

`AAHandInput` / `aa_analysis_records` 的 scope 是
`MANUAL_HYPOTHESIS_OFFLINE_NOT_LIVE_ADVICE`：在表单里把事实标成
`human_confirmed` 得到的是**离线情景分析**，不是验收。

该路径**保留原样**（不删除、不破坏、不改行为），但：

- 它**不能**生成 `TARGET_S_REAL_HAND_CONFIRMED`；
- 只有本模块的 `target_s_acceptance()` 能生成这个状态。

测试 `test_manual_hypothesis_human_confirmed_is_not_acceptance` 直接证明这点。

---

## 4. 确认记录（每条事实一行，append-only）

```json
{
  "schema_version": "aa-real-hand-confirmation-v1",
  "confirmation_id": "rhc-<hex>",
  "hand_id": "observed_deal_...",
  "observed_epoch": "...",
  "capture_session_id": "...",
  "source_ref":  {"media_sha256": "...", "recording_id": "..."},
  "source_digest": "<sha256>",
  "marker": "RIVER_START",
  "source_frame": 15033,
  "evidence_digest": "<sha256>",
  "snapshot_digest": "<sha256>",
  "fact_key": "hero_seat",
  "value": 1,
  "reviewer": "...",
  "recorded_at": "...",
  "revision": 1,
  "supersedes": null,
  "provenance": "human_confirmed",
  "audit_note": "可选，仅供审计"
}
```

绑定规则：

- **无 `evidence_digest` ⇒ 拒绝 CONFIRMED**（必须是 64 位 hex）；
- `hand_id` / `observed_epoch` / `capture_session_id` / `source_digest`
  与同手已有记录不一致 ⇒ 拒绝；
- 自由文本**只能**落在 `audit_note`，**永远不能**成为 `value`
  （每个 `fact_key` 都有结构化校验，一段话过不了任何一项）。

---

## 5. 修订与冲突

- **不可原地覆盖**。更正 = 追加新行，`revision = prev + 1`，
  `supersedes = 当前 head 的 confirmation_id`。
- `supersedes` 指向的**不是**当前 head ⇒ `RevisionError`（stale）。
- 已有确认却不带 `supersedes` ⇒ `RevisionError`。
- **CONFLICT（fail closed）**：双 head / 重复 revision / 两个修订 supersede
  同一父行 / 无 live head / head 不是最高 revision。
  ⇒ 该事实 `UNCONFIRMED`，并列出 `conflicts`。
- 持久化：`confirmations.jsonl` + 文件锁 + 逐行摘要链（prev_digest）。
  参考 #31 的 infrastructure 纪律，**不复用其 annotation truth 语义**。
  **不建数据库。**

---

## 6. P0 收敛：Hero 是河牌首个行动者

第一手受控采集主动挑 **Hero = river first-to-act**，因此：

```
river public history = []
```

这是**合法设计，不是遗漏**：`aa_hand_input._replay()` 规定空历史后只要
`node.pending[0] == hero_seat` 就是合法的 Hero 未结束决策点。

但空历史**不能靠用户写一个 `[]`**：

- 必须确认事实 `hero_is_first_river_actor = CONFIRMED`；
- `action_order[0] == hero_seat` 必须成立；
- 两者任一不成立 ⇒ `history` 未确认 ⇒ `TARGET_S_NOT_READY`。

本轮**不实现**逐条 action 人工确认（那是 Hero 后手时的 P1）。

---

## 7. TARGET-S 必需事实

| 项 | fact_key | 说明 |
| --- | --- | --- |
| A 手/epoch 身份 | 记录级 `hand_id` + `observed_epoch` | 同手所有记录必须一致 |
| B 采集身份 | 记录级 `capture_session_id` + `source_digest` | 同上 |
| C 已结束 | `ended_hand_confirmed` | 必须 `True` |
| D Hero 座位 | `hero_seat` | 必须 ∈ ACTIVE |
| E Hero 手牌 | `hero_cards` | 2 张合法且互异 |
| F 公共牌 | `board_cards` | **恰好 5 张且互异**，与 Hero 手牌不重叠 |
| G 座位集合 | `seat_set` | 覆盖 ACTIVE ∪ ALL_IN ∪ FOLDED |
| H 恰好 3 名 ACTIVE | `active_seats` | **必须 len == 3** |
| I ALL_IN 与 ACTIVE 分离 | `all_in_seats` | 与 `active_seats` 不得相交 |
| J 行动顺序 | `action_order` | ACTIVE 的一个排列，且 `[0] == hero_seat` |
| — 庄家 | `dealer_button_seat` | 位置顺序依据 |
| K 河牌起点筹码 | `river_start_stacks` | 覆盖 `seat_set` |
| L 开局筹码 | `opening_stacks` | 覆盖 `seat_set` |
| M 已投入（**派生**） | `hand_committed` | 见 §8 |
| N 本街投入为 0 | `street_wagers_zero` | 必须 `True` |
| O 底池显示 | `pot_display` | `{raw, value}` 并列 |
| P 历史为空 | `hero_is_first_river_actor` | 见 §6 |
| Q 桌规 | `table_rules` | 带 `revision` |
| R 其他费用 | `other_fees` | `{value, confirmed}`，**未知不得默认 0** |
| S 无边池 | `no_side_pot` + 派生校验 | 见下 |
| T 无未完成动作 | `no_pending_action` | 必须 `True` |
| — 本手 straddle | `straddle_posted_this_hand` | `optional_explicit_utg` 时**必须逐手观测** |

**边池双重校验**：除 `no_side_pot` 外，模块还用**派生出的**逐座已投入核对 ——
三名 ACTIVE 必须相等，且不得有座位高于该水平。勾一个框盖不住数字矛盾。

**per-hand straddle 不可由桌规推出**：桌规只声明金额；
`straddle_mode == optional_explicit_utg` 时本手是否真下过是**观测**，
未确认 ⇒ `TARGET_S_NOT_READY`。

---

## 8. Stack-delta commitment 派生（C1–C8）

```
hand_committed[seat] = opening_stack[seat] - river_start_stack[seat]
```

**只在全部条件成立时**才是 `CONFIRMED` 派生：

- **C1/C2** HAND_START 与 RIVER_START 两个时点本身可信；
- **C3** 座位身份连续（两快照座位集合相同，且等于确认的 `seat_set`）；
- **C4** 期间无筹码侧变更，**逐项独立断言**：
  无 rebuy/top-up、无退还筹码、无提前派彩、无 jackpot/cashout/保险；
- **C5** 期间无 rake/fee 从**筹码侧**扣除；
- **C6** 两快照均为**非负整数**且同一精度（浮点/缩写即拒绝）；
- **C7** 每个 `delta >= 0`；
- **C8** **独立信号交叉校验**：`Σ(delta) == RIVER_START 的 pot_display`。

任一失败 ⇒ 整条派生 `UNCONFIRMED`。
**禁止**：平摊、补零、用底池反推逐座投入。

派生值的 `provenance` 是 **`derived_from_confirmed_observations`**，
`derivation = "stack_delta_v1"`，**不是** `human_confirmed` ——
一个算出来的数不能冒充人的陈述。

### HAND_START 的定义（不要绑死成「发牌后」）

实际 UI 可能不会在发牌后、强制下注前停留。因此定义为：

> 本手第一笔 forced-bet 借记发生**前**，最后一个稳定、
> 座位身份连续的 stack snapshot。

它可以是上一手结算完成后、下一手强制下注前的稳定 inter-hand 帧，
但必须能证明与下一手的**边界连续性**。跨换座 / 补码 / 离桌 / 重新入座
⇒ 该座开局筹码 `UNCONFIRMED`。

### RIVER_START 的定义

第五张公共牌稳定出现，且任何 river action 尚未发生。P0 额外要求
`hero_is_first_river_actor = CONFIRMED`。

### PAYOUT

只用于手边界与结算一致性，**不用于 solver 起点输入**。
不因 payout 看起来合理而倒推 river-start 的未知项。

---

## 9. 桌规

**桌级人工确认**（带 `revision` 与 `rules_digest`）：
`table_size` / `small_blind` / `big_blind` / `ante` / `ante_mode` /
`straddle_amount`（**仅金额**）。

`min_bet` **不单独声明**：PHH 直接取 `big_blind`，另设独立声明只会制造第二个真值源。

**必须逐手观测**：`optional_explicit_utg` 下 straddle 是否真下过、由谁下；
实际盲注张贴座位（换座后首手 / dead button / 补盲）；本手实际 rake/fees；jackpot drop。

**绝不默认 0**：`ante` 未声明或 `ante_mode` 未知 ⇒ 未确认；
`other_fees` 未知 ⇒ 未确认，不得当 0。

---

## 10. confirmed_view 与 acceptance receipt

`confirmed_view(store, hand_id)` 每次都**重新派生**全部内容：
逐事实解析 → 重新计算派生值 → 重跑所有结构校验。磁盘上的任何
「已验收」字样都不被信任（本模块也不存这种字段）。

返回：`status` / `confirmed` / `derived` / `unconfirmed` / `conflicts` /
`blocking` / `checks` / `derivations` / `evidence_refs` / `bundle_digest`。

`target_s_acceptance(store, hand_id)` 是**唯一**能产出
`TARGET_S_REAL_HAND_CONFIRMED` 的闸门。receipt 绑定：

- `hand_id` / `observed_epoch` / `capture_session_id` / `source_digest`
- `confirmation_bundle_digest`
- `river_start_snapshot_digest`
- `derived_commitment_digest`
- `rules_digest` / `facts_digest`
- `implementation_version`

并**明确写出**：

```
phh_ready          = false
strategy_assessed  = false
target_p_phh_ready = false
```

**TARGET-S confirmed ≠ strategy validated。**
对手范围与响应模型仍是手工假设（`manual_river_start_assumption`），
`NOT_ASSESSED` 继续保留。

---

## 11. 受控采集协议（只设计，本轮不启动）

三个 marker，每个标记人需要确认的东西：

| Marker | 时点 | 人工确认 |
| --- | --- | --- |
| **HAND_START** | 本手第一笔 forced-bet 借记**之前**的最后一个稳定帧（可在上一手结算后） | 逐座开局筹码、庄家、座位集合、本桌生效规则、手边界连续、无待处理补码 |
| **RIVER_START** | 第 5 张公共牌出现后、**河牌首次下注之前** | Hero 座位/手牌、board 5 张、逐座筹码、逐座状态（ALL_IN 与 ACTIVE **分列**）、**恰好 3 ACTIVE**、**无边池**、河牌首个行动者（= Hero）、底池显示、本街投入全为 0、无未完成动画、本手 straddle 是否真下过 |
| **PAYOUT** | 结算数字稳定显示 | 逐座派彩、退还筹码、本手实际 rake/fees、下一手尚未开始 |

环境要求：全座筹码与每个 action 金额同屏可见、无弹窗遮挡、
UI 缩放与分辨率全程不变、board 三次过渡完整可见。只需 1 手。

---

## 12. 状态边界

- `REAL_HAND_ACCEPTANCE_PENDING` —— **继续 YES**。
  本轮只是让严格验收路径在架构上**可达**；还没有真实受控录像，
  没有一手真实牌局被验收。
- `NOT_ASSESSED` —— **继续 YES**。

本模块不声明：real hand accepted / PHH ready / holdout accepted /
gold ready / strategy validated / profitability validated / GTO complete。

正向消费许可仍 CLOSED；freeze adapter 仍 NOT ATTACHED。

---

## 13. 测试

`tests/desktop/test_aa_real_hand_confirmation.py`（35 例，全部合成数据，
不读 raw media）覆盖：证据绑定、身份一致性、append-only 修订、stale
supersede、双 head 冲突、自由文本不得为事实、机器候选不得升级、
#31 annotation row 不得升级、手工假设 `human_confirmed` 不得产出 receipt、
Hero 先手硬门、恰好 3 ACTIVE、ALL_IN 不算 ACTIVE、边池、stack delta 的
C1–C8 逐条 fail closed、marker 状态门、完整正例与 receipt 的
`phh_ready=false` / `strategy_assessed=false`。
