# USABLE-001：开局边界重复 epoch 修复 + 普通河牌终局（ORDINARY_RIVER_CLOSED）

状态（三段分开记，不要混成一个）：
- 桌面层重复 epoch 缺陷：**`FIX_ACCEPTED_WITH_SCOPE`**（审查已接受）；
- 普通河牌终局语义：**`CODE_TESTS_PASS` / `REAL_REPLAY_NOT_YET_PROVEN`**（真实录像命中数 **0**，第一阻塞见第 4 节）；
- 真实牌局验收：**`PENDING`**；策略强度：**`NOT_ASSESSED`**。

范围：桌面层边界缺陷 + 新的**并列**终局类型。未改内核、未改求解器、未改计算上限、未重标 ROI、未放宽任何 `UNKNOWN`、
**未为追求命中而改动任何规则**。

## 1 桌面层重复 epoch（已修，含回归）

### 1.1 症状

同一段历史回放（源帧 1260–1290）在桌面链上：

| 帧 | epoch | 状态 | 账本 |
|---|---|---|---|
| 1264 | `observed_deal_1263` | `MULTI_POST_DEAL_CANDIDATE`（7 座扣款） | `OBSERVED_HAND_COMMITMENTS_CANDIDATE` |
| 1278 | `observed_deal_1277` | `DEALER_ADVANCE_CONTEXT_CANDIDATE`（0 扣款） | 回到 `HAND_COMMITMENTS_UNKNOWN` + `opening_post_vector_incomplete` |

即：**已经解析出来的开局账本，在 14 帧后被一个重复 epoch 覆盖并重新污染**。对照组（纯 `AA8StateAdapterV2`，历史成功运行使用的那条链）在同一批帧上**连续 67 帧保持已解析**。

### 1.2 机制

`aa_live_context.LiveStateAdapter.observe()` 的 dealer 前移分支只要求

- `dealer_run >= 2`，且 `dealer_value != detected`，且 `row["board_count"] == 0`

**没有排除「本手已经拿到合格 opening」**。因此 dealer 座位读数变化（此录像中为读数抖动）会无条件 `_new_epoch()`，而 `AA8HandLedgerCandidate._reset()` 对新 epoch 找不到 multi-post ⇒ 追加 taint，账本失效。

### 1.3 修复

新增 `LiveStateAdapter._anchored_opening()`，**逐条镜像 `_reset` 的合格判据**（同 epoch 恰好 1 个事件、`status == "MULTI_POST_DEAL_CANDIDATE"`、`debits` ≥ 3 座、本 epoch 尚无已观察动作），命中时该分支**只更新 dealer 读数、不新开 epoch**。

刻意保持最小：
- 只保护「**已被账本接受**的开局」，不扩大语义；
- 只覆盖**已证明**的情形（空 board、pot 仍为 0、本手无动作）；
- 「本手已有动作后的读数变化」**不在本轮范围**（未被证明），代码注释已显式标注，需要结算证据才能处理，本守卫不声称覆盖它。

### 1.4 回归测试

`tests/desktop/test_aa_live_opening_boundary.py` + 夹具
`tests/fixtures/aa_reference_hands/opening_rows_1260_1290_v1.json`
（源帧 1260–1290 的 **adapter 之前** 的行，逐字段取自冻结的开发录像；1260–1262 来自 `aa8_first_hand_entry_dense_v2`，1263+ 来自 `aa8_first_hand_full_v1`，因此 **CI 不需要 `G:` 私有池**）。

夹具与仓库内**人工复核**的 `eight_session_boundary_v1.json` 独立对上：`initial_posting_slots = [0,1,2,3,4,5,7]`、空座 6、hero `200 → 196`（与修复前的实测 `258/294/528/340/200/286/None/256 → 255/290/522/338/196/284/None/248` 逐座一致）。

反例先行（新测试拷到上一 head `064894b` 的独立只读 worktree 实跑）：**8 failed / 2 passed**；本轮 **3 passed**。两条在旧 head 上即为绿的用例是**防止过度修复**的守卫（夹具自检、以及「无合格 opening 时 dealer 前移仍必须重锚」）。

## 2 本次牌桌的强制投入规则（已核查，**未改 ≥3 座**）

### 2.1 可确认：强制投入总额恰为 17

三手牌**各自**在开局首次出现非零底池，且**都恰好是 17**：

| 帧 | 底池 | 底池非零帧 |
|---|---|---|
| 8647 | 0 → 17 | 8647 |
| 9814 | 0 → 17 | 9814 |
| 11130 | （已是）17 | 11130 |

仓库内 `configs/game/aa-shadow-rules-v2.json` 声明该桌为
`small_blind 1 / big_blind 2 / ante 2（per_dealt_player）/ mandatory_utg straddle 4`，8 人桌。该桌本次连续三手都被观测到 **5 个已发牌座位**（空座恒为 0、2、6），于是：

```
5 座 × ante 2 = 10，+ SB 1 + BB 2 + UTG straddle 4 = 17
```

**与观测到的开局底池逐手吻合。**

### 2.2 因此：`>= 3 seats` 本来就**可满足**，不得放宽

按上述结构，该桌开局有 **5 个座位同时扣款**（全部已发牌座位都下 ante，其中 3 座另加盲注/抓头）⇒ `posting_comparison` 与 `HandTransitionCandidates.min_posting_seats` 的 ≥3 座要求**在该桌成立**。**本轮未改动任何座位数门槛**，也不建议改。

### 2.3 真正的阻塞是「扣款不可见」，不是座位数

三个开局窗口内，**没有任何座位的识别筹码出现过下降**（`distinct seats that ever decreased = []`），而同窗口 `pot` 在变。`causal_street_wagers_v2.status = WAGERS_UNKNOWN`、`street_price = None`、glyph 全为 null ⇒ **价格通道也不可用**。

结论：`posting_comparison` 是**基于筹码差值**的方法，而本次录像的筹码读数在开局前后不变，因此该方法在此录像上**结构性看不到**强制投入。这是读数/显示问题，与座位门槛无关。

> 证据分级：2.1 的算术吻合是**推断**（配置自身标注 `verification_status: "simulation"`，其 source 亦写明 “not verified live room accounting”）；要把它升级为**已确认**，需要用户从客户端确认该桌结构，或取得**至少一帧筹码确实下降**的开局证据。

## 3 普通河牌终局：`ORDINARY_RIVER_CLOSED`

### 3.1 定位与边界（避免假装覆盖全部结束方式）

新增**并列**终局，**不改动**现有 all-in 终局（`closed` / `full_river` / `terminal` 一字未动）：

- `hand_phase.ordinary_terminal.kind == "ORDINARY_RIVER_CLOSED"`
- `hand_phase.ordinary_terminal_semantics == "ORDINARY_RIVER_CLOSED_ONLY; preflop/flop/turn fold-out endings are NOT covered"`

**只覆盖**：普通（非两家人 all-in）牌局走到**完整河牌**并完成结算。
**不覆盖**：翻前 / 翻牌 / 转牌弃牌结束 —— 后续单独实现（本轮不做）。

### 3.2 后验确认三条件

| | 条件 | 实现 |
|---|---|---|
| **C1 河牌完成** | `street_candidate == "river"` + 5 张合法互异公牌（复用既有 `full_river` 谓词）+ `pending_actions == 0`，且 **`closed` 为假** | 记录 `river_complete_frame`（首次满足的那一帧，不随后续帧前移） |
| **C2 结算正证据** | 同 epoch 内 `confirmed_frame > river_complete_frame` 的 `unallocated_positive_cash`（余额增加）。**底池读数为 0 本身不足以推断结算**；底池读不到也不阻塞 | 形成 `settlement` 证据列表 |
| **C3 下一手边界** | **唯一确认器**：观察到**新的、非空** epoch，且该帧 `> river_complete_frame` | 置 `confirmed_by_epoch_frame` |

参与座位取自**本 epoch 自己的 `participants`**（状态 ∈ {active, all_in, folded}，≥2 座），**不读 `ledger.opening_evidence`** ⇒ 开局账本缺口不会连带禁掉本终局。

关于 C3 用「事件帧」：`_new_epoch()` 用**创建它的那一帧**给事件打戳，而创建发生在 `observe()` 同一次调用内 ⇒ **首次观察到新 epoch 的帧就是事件帧**，故 `frame > river_complete_frame` 等价于设计要求的「新 epoch 事件帧晚于河牌完成帧」。epoch 变成 `None`（场景失支持/遮挡）是**挂起**，**永不确认**。

### 3.3 三条限制（写入输出而非注释）

- **L1 一手之隔**：普通终局只能在下一手开始后才确认 ⇒ **录像最后一手永远无法确认**。硬停止的录像最后一手只能是 PENDING。这是「不猜」的代价。
- **L2 不依赖 ledger**：如上，参与集合不取自开局账本。`ledger_status` 原样记入证据，供下游**分开**要求账本。
- **L3 不弱化 all-in**：`ordinary_terminal` 一律 `canonical_verified = False`、`strategy_eligible = False`、`card_showdown_verified = False`、`rake_verified = False`；不设置 `settlement_rules_verified`；不占用 `terminal_observation_frame`；不与 `closed` 同时成立。

### 3.5 生命周期契约（在任何下游消费者出现之前钉死）

`ordinary_terminal` **是确认事件，不是状态**：

| 字段 | 语义 | 存活范围 |
|---|---|---|
| `hand_phase.ordinary_terminal` | **确认事件**。只在**确认那一行**出现，之后立即清空 | 一行 |
| `hand_phase.last_ordinary_terminal` | **最近一次确认的持久快照**，带自己的 `epoch` 与 `belongs_to_current_epoch` | 直到下一次确认 |
| `hand_phase.ordinary_river_pending` | C1+C2 已成立、C3 未到，**只是等待**，不是结果 | 逐行判断 |

清空发生在 `observe()` 里 `_phase()` 返回之后，因此**覆盖 `_phase` 的每一条 return 路径**（包含被遮挡/丢 epoch 的早退），事件不可能跨行残留。

`last_ordinary_terminal.belongs_to_current_epoch` 每行重新计算（`epoch == hand_phase.epoch`），因此**旧手的终局不可能被误当成当前手**；消费者也可以直接比较 `last_ordinary_terminal.epoch` 与 `hand_phase.epoch`。

守护测试（`tests/desktop/test_aa_ordinary_river_terminal.py`）：
确认行之后连续三行 `ordinary_terminal is None`；跨第 2/3/4 个 epoch 时 `last_ordinary_terminal.epoch == "h"` 且 `belongs_to_current_epoch is False`；
遮挡行既不带事件也不带 pending；未确认前两个字段都为空。

### 3.5 已知局限（如实标注）

C2 依赖「结算出现**可见的余额增加**」。§2.3 已测得**本次录像的筹码读数在开局前后不变**，因此该录像的支付是否可见**尚未验证**。第 4 节的真实回放给出了答案：**多数手上不可见**（3/5），但**至少一手成立过**（`observed_deal_14593`）——见 4.2 与 4.3。

### 3.6 反例测试（`tests/desktop/test_aa_ordinary_river_terminal.py`，12 项）

1. 确认只发生在下一手边界之后（边界前只有 `AWAITING_NEXT_HAND_BOUNDARY`）。
2. `ordinary_terminal_semantics` 明示不覆盖范围。
3. 底池 0 但无支付证据 ⇒ 不结算。
4. 河牌后 **epoch 丢失（None）⇒ 永不确认**（对应本次 157.7 s 静止缺口这一类）。
5. 河牌后录像立即结束 ⇒ 只 PENDING（对应硬停止）。
6. **两家人 all-in ⇒ 只走 all-in 终局，不产生普通终局**（互斥）。
7. 遮挡帧（`insurance: VISIBLE`）⇒ `SUSPENDED`，既不确认也不显示 PENDING。
8. 翻前弃牌结束 ⇒ 本终局不覆盖（不产生任何普通终局声明）。
9. **确认事件一次性**：确认行之后连续三行 `ordinary_terminal is None`。
10. **快照不泄漏**：跨第 2/3/4 个 epoch，`last_ordinary_terminal.epoch` 仍是 `"h"` 且 `belongs_to_current_epoch is False`；回到该 epoch 时该标志翻为 `True`（证明它是算出来的，不是常量）。
11. 未确认前 `last_ordinary_terminal is None`。
12. 确认事件不随「状态标志」存活：紧随其后的遮挡行不再带事件。

反例先行：新测试拷到上一 head `064894b` 实跑 **全部失败**；本轮 **12 passed**。

## 4 真实录像连续回放（`aa-live-20260917-0430`，head `fbda7c52`）

只读、**连续顺序**（`grab()` 逐帧、不 seek、不稀疏孤帧），整段 **0–27,287 帧**一次跑完。

### 4.1 命中统计

| 指标 | 数量 |
|---|---|
| epoch（含 `None` 挂起段） | **25** |
| 达到 **full river** 的 epoch | **6** |
| 产生 `ordinary_river_pending` | **1**（`observed_deal_14593`） |
| 产生 `ORDINARY_RIVER_CLOSED` | **0** |

6 个 full-river epoch：`None`、`observed_deal_8622`、`observed_deal_11609`、`observed_deal_14593`、`observed_deal_17323`、`observed_deal_24560`。

### 4.2 逐手第一阻塞

| epoch | river 帧 | river 座位数 | `pending_actions` | 河牌**之后**确认的 credit | 结果 |
|---|---|---|---|---|---|
| `None`（挂起段） | 8397 | 6 | **非 0** | 119 | C1 不成立（epoch 为空，按设计永不确认） |
| `observed_deal_8622` | 9512 | 4 | **非 0** | 0 | C1 与 C2 皆不成立 |
| `observed_deal_11609` | 12325 | 5 | 0 | **0** | **C2 不成立** |
| **`observed_deal_14593`** | 15033 | 5 | 0 | **1** | **pending 成立**（38 帧），确认时已无 pending |
| `observed_deal_17323` | 18170 | 6 | 0 | **0** | **C2 不成立** |
| `observed_deal_24560` | 25674 | 5 | 0 | **0** | **C2 不成立** |

⇒ **第一阻塞是 `payout_credit_not_visible`（3/5 手）**：这些手在河牌帧与 epoch 结束之间，**没有任何该 epoch 的 credit 被确认**。

> 注意区分：credit **通道本身很忙**（整段 26,980 帧有 `unallocated_positive_cash`，累计 **218,310** 条），但这些多为开局/退款类 credit（`_new_epoch` 的 `new_post_comparison` 与 `stable_visual_balance_increase`）。**"该 epoch 有 credit" ≠ "有河牌后的支付"**；C2 要求 `confirmed_frame > river_frame`，本表最后一列正是这个严格计数。所以不是通道静默，而是**缺"河牌后的支付"这一类**。

### 4.3 第二阻塞：`observed_deal_14593` 为何成立却没被确认（帧级证据）

C1+C2 在真实数据上**可以**成立——这手在 **f15033** 河牌完成（`rf=15033`），并且在 **f15173–f15185** 连续 38 行带着有效 pending（`settle=1, seats=5, pend_act=0, pending=True`）。随后：

```
f15186–f15197  complete=False → rf=None → settle=0 → pending=False   ← 12 帧河牌读数中断，pending 被撤回
f15198         epoch 切到 observed_deal_15197（与上一 epoch 直接相邻，中间没有 None）
```

⇒ 到 f15198 发生 epoch 切换时，**`self.ordinary_river` 已经被清空**，C3 没有可确认的对象。

成因（**我的实现缺陷，不是设计上的"丢 epoch"**）：`river_frame` 在**河牌完整性一旦中断**就被清零，而 hand 收尾动画会让 5 张公牌的读数短暂消失 12 帧。也就是说，「这手有完整河牌」是一个**已经成立的本手事实**，却被一次显示抖动撤销了。注意 epoch 序列显示这里是 **A→B 直接切换，中间没有 `None`**，因此这不是 3.4 第 4 条所覆盖的场景。

**最小改法（本轮未执行，待授权）**：`river_frame` 只在 **epoch 变化**时清零，不再因公牌读数抖动而清零；`closed`（all-in）仍在 eligibility 层抑制。残余风险如实标注：若 adapter 长时间不切 epoch，pending 会挂在该 epoch 上；但 C3 要求 pending 的 epoch == 刚刚结束的 epoch，且 epoch 名唯一，因此不会跨手错误归属。

### 4.4 结论

`ORDINARY_RIVER_CLOSED` 当前状态应记为 **`REAL_REPLAY_NOT_YET_PROVEN`**：C1+C2 已在真实数据上成立过一次（`observed_deal_14593`），但**该手最终未被确认**，所以**命中数为 0**。第一个要修的缺口是 4.3（本实现缺陷），其后是 4.2 的 `payout_credit_not_visible`（证据缺口）。两者都**尚未授权修改**。

## 5 本轮未做

未改 else 分支的语义、未改 `closed`/`full_river`/`terminal`、未改座位门槛、未改 `_posting` 的比较锚点与 `expires`、未重标 ROI、未调阈值、未动 `LiveCausalWagers`、未改前端、未合并发布。翻前/翻牌/转牌弃牌结束**未实现**（后续单独做）。

**已定位成因但本轮明确未改**：
- 4.3 的 `river_frame` 过早清零（我这一轮实现内部的缺口；改法已在 4.3 写出）——本轮授权是「只做验证与一个生命周期守卫」，因此**只报告、不动手**；
- 4.2 的 `payout_credit_not_visible`——**不为了让结果通过而放宽 C2**。

## 6 验证

- 全仓 `pytest -v`：**4124 passed / 1 skipped / 2 warnings / 414.33s**（上一基线 4120/1，本轮 +4 = 生命周期契约测试）。
- `tests/desktop` 相关五个文件：**86 passed**。
- `flake8 src tests tools`：0 项。
- 旧 head 反例：两个新测试文件在 `064894b` 上 **8 failed / 2 passed** → 本轮 **15 passed**。
- 真实回放：见第 4 节（连续 0–27,287 帧，命中 `ORDINARY_RIVER_CLOSED` = **0**）。
- 端到端回放（真实池帧 1260–1330）：桌面链 epoch 由「1264 MULTI_POST + 1278 DEALER_ADVANCE」变为「**仅 1264 MULTI_POST**」，与纯 `AA8StateAdapterV2` 的历史行为一致。
