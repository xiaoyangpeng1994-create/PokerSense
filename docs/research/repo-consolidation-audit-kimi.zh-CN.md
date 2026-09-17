# PokerSense 全仓库 PR/架构收口审计（Kimi 第一审）

> **AUDIT ONLY · NO PRODUCT CODE CHANGE · NO MERGE AUTHORIZATION**
> 本文件是独立审计意见，不构成任何合并授权。审计者未修改、未合并、未 rebase、未关闭任何 PR。
>
> - 审计日期：2026-09-17
> - main 基线：`fb703e22459e0f5df69ff1c4005ef868817fd653`（Merge PR #18）
> - 审计对象：开放 PR #17 / #19 / #20 / #24 / #26 / #28 / #29
> - 证据分级：**【Git】** 本地实跑 git/gh 命令所得；**【读码】** 直接读分支上的文件；**【实跑】** 审计者本人运行的测试/命令；**【文档】** 仓库文档自述（未经独立复核时不升级为事实）；**【判断】** 审计者的独立推断。
> - 审计用引用全部为 `origin/*` 只读引用与独立 worktree；本地主 checkout（脏且落后）未被信任、未被修改。

---

## 0. 一句话结论

**主链 #20 → #24 → #28 是真实、连续、质量高于平均的一条 stacked 实现链，应按序合并；#26 独立低风险可并；#19 需先裁决「老回放轨 vs 新复查台」的人工确认归属再定；#17 已被 #20 取代应关闭（结论迁入 issue）；#29 是纯研究文档，随 #28 之后合入。当前没有发现阻塞合并的 P0 代码缺陷；最大的结构性风险不在代码，而在 merge 链的 base 管理与 CI 对 `solver-tools` 的覆盖缺口。**

---

## 1. PR 与 Git 关系（实跑重建，不采信 PR 描述）

### 1.1 元数据（【Git】`gh pr view --json`，2026-09-17 实拉）

| PR | 标题（缩） | base | head 分支 | head SHA | mergeable | ahead/behind main |
|---|---|---|---|---|---|---|
| #17 | AA 字样留出资格审计 | `main` | `codex/aa-glyph-holdout-v1` | `0a27045d3aaf6617eaee1e07831144b058f84c27` | **CONFLICTING** | 5 / **2** |
| #19 | 人工确认台账+复盘导出 | `main` | `codex/aa-human-confirmation-ledger-v1` | `1285b9e48195bdd773ed07d7aafb2ba8d41c5e22` | MERGEABLE | 1 / 0 |
| #20 | AA 八座观察+动作解释+多人分析 | `main` | `codex/aa-live-recognition-v1` | `dc72a4176f81689e95a9238a49fb2d268dcd4a56` | MERGEABLE | 15 / 0 |
| #24 | STRATEGY-DIAG 固定策略诊断 | `codex/aa-live-recognition-v1` | `codex/strategy-001-research` | `7fcd6db030baf7a54a69a41e9fc485e6957f7aac` | MERGEABLE | 25 / 0 |
| #26 | GITHUB-001 Git 同步链路 | `main` | `codex/github-001-integration` | `0003a376faf61fa6a88f6e0b0e1c1b1323402b8a` | MERGEABLE | 5 / 0 |
| #28 | USABLE-001 真实牌局输入 | `codex/strategy-001-research` | `codex/usable-001-hand-review` | `0b7c40ad9531672dcd45dae475d5785ebefccd2b` | MERGEABLE | 45 / 0 |
| #29 | 开源架构基准（研究轮） | `codex/usable-001-hand-review` | `codex/open-source-arch-benchmark` | `a8af7f4c30e08bd0a83baac2009e857e7a08e846` | MERGEABLE | 39 / 0 |

### 1.2 祖先包含关系（【Git】`git merge-base --is-ancestor` 实跑）

```
main (fb703e2)
 └── #20  aa-live-recognition-v1   (dc72a41, +15)
      └── #24  strategy-001-research   (7fcd6db, +10 over #20)
           └── #28  usable-001-hand-review (0b7c40a, +20 over #24)
                └── #29  open-source-arch-benchmark (a8af7f4, 分叉点 fbda7c5, +1)

独立分支：#17（落后 main 2 个 merge：#16、#18）、#19（+1）、#26（+5）
```

实测结果：

- `#20 ⊂ #24`：**YES**（#20 的 head 是 #24 的祖先）
- `#24 ⊂ #28`：**YES**
- `#28 ⊄ #29`：#29 从 #28 的第 14 个提交 `fbda7c5` 分叉，**不含** #28 最后 6 个提交（P0.1 守卫 `12aaf6a`、P1 影子 oracle `7f3a92b` 等）
- `#19 ⊄ #20`、`#17 ⊄ #20`、`#26 ⊄ main`：均 **NO**
- #28 worktree 三方一致：`local worktree HEAD == origin/codex/usable-001-hand-review == gh headRefOid == 0b7c40a`（【Git】）

结论：

- **#20/#24/#28 是一条真 stacked 链**。单独 merge 其中任何一个而跳过前者没有产品意义：#28 的 45 个提交里只有 20 个是自己的，其余 25 个属于 #20+#24。
- **#29 的内容是 1 个研究文档**（`docs/research/pokersense-open-source-architecture-benchmark.zh-CN.md`，+484/−0），但其基线戳是 `fbda7c5`，比 #28 当前 head 落后 6 个提交。
- **#19、#26 各自独立**，不与主链共享提交；与主链同时 merge 时存在文本冲突（见 §6 R12），全部在 `AGENTS.md` / `README*` 文档层，无代码冲突（【Git】`git merge-tree` 实测）。
- **#17 落后 main 两个合并（#16、#18）且 CONFLICTING**；其分支 5 个提交（新工具 `tools/aa_glyph_holdout_v1.py` +612、测试 +331、配置 +612）在 #20 分支上**全部不存在**（【读码】逐文件比对）。

### 1.3 各 PR 独有内容（【读码】diff 逐文件）

- **#20**（链基座）：新增 15 个 `desktop/aa_*.py`（识别会话 `aa_session`、读数 `aa_reader`、状态适配 `aa_live_context`、复查台 `aa_review`、服务端 `aa_server` 等）+ 7 个 `strategy/*_v1.py`（三人河牌内核 `threeway_river_v1`、多人终局 `terminal_multiway_v1`、对手数据集、响应校准、河牌边界、稳健策略选择、策略评估）+ 配套测试/文档。这些文件在 main 上**全部不存在**（【Git】`git ls-tree` 对比）。
- **#24**：`aa_study_records.py`（+1062，离线研究报告入复查台）+ 路径诊断/范围实验工具（`tools/threeway_path_diagnostics.py`、`tools/threeway_range_experiment.py`）+ 研究文档 5 篇。
- **#28**：`aa_hand_input.py`（+861）、`aa_analysis_records.py`（+889）、`aa_phh_shadow.py`（+614）、`aa_semantics.py`（+136/−2）、`aa_live_context.py`（+48/−15）、`aa_server.py`（+111/−1）+ U2 试用启动器 + 12 篇研究文档 + 大量测试。深审见 §3、§4。
- **#26**：`tools/git_sync.py`（+777）+ 测试（+521）+ `docs/GITHUB-SYNC-V1.zh-CN.md`。离线 dry-run 通过：缺凭据时 `blocked by credential` / 远端只读失败时 `blocked by remote read`，不落盘、无解析错误（【实跑】，由并行的只读子审计执行）。
- **#19**：`tools/aa_confirmation_ledger.py`（+323，只追加 JSONL 台账，payload 逐字段 sha256 哈希链）+ `tools/aa_replay_viewer.py` 扩展 + `ui/aa-replay/` 导出。
- **#17**：负结论载体——"在符合条件的同 session 回放里**找不到任何够格的 AA 字样留出素材**"，glyph 行数 0（+0/−0），逐原因排除：3 例 OFFSCREEN、1 例 OVERLAY。其 `aa-glyph-freeze-v1` 的 21 个 glyph 字典在 main 与 #20 上**逐字节一致**（【读码】哈希比对）。

---

## 2. 整体架构：每一种事实由谁拥有

### 2.1 事实所有权表（【读码】+【Git】，路径均为分支相对）

| 事实类别 | 拥有者（main） | 拥有者（主链新增） | 备注 |
|---|---|---|---|
| capture / 帧接入 | `realtime/frame_source.py`、`tools/aa_record_session.py` | `desktop/aa_sources.py`（#20）+ `aa_device_lock.py` | 两轨并存，见 R4 |
| vision / 视觉识别 | `perceptual/vision/*`、`tools/aa8_*`（约 40 个识别工具） | `desktop/aa_reader.py`（#20，AA8Reader 时序链） | 产品 IP，不可替代 |
| temporal / 时序确认 | `realtime/temporal_consensus.py`、`tools/aa8_state_adapter_v2.py` | `desktop/aa_reader.py`（帧号 +1 / PTS 递增否则整棵 reset） | 上游无对应物 |
| state / 游戏状态 | `core/models.py::GameState`、`state_engine/engine.py` | `tools/aa8_state_adapter_v2.py`（AA 轨，`observed_state_v2`） | 两套状态语义并存：canonical vs observed-candidate |
| seat/dealer/actor | `tools/aa8_dealer_v2.py`、`tools/aa8_participation_v2.py`、`realtime/hand_boundary.py`（`ActorTracker`） | `desktop/aa_live_context.py::LiveFrameEvidence`（#20） | actor 跟踪有两路（`hand_boundary.py` 与 `aa8_actor_ring_v2.py`），无 docstring 说明从属（R8） |
| epoch | `tools/aa8_state_adapter_v2.py::_new_epoch`（`epoch_events`） | `desktop/aa_live_context.py` 子类化扩展 | 无独立 epoch 模块；epoch 名不含状态（既有教训） |
| hand ledger | `tools/aa8_strategy_state_v2.py::AA8HandLedgerCandidate` | `desktop/aa_live_context.py::LiveHandLedger`（#28） | 三态：`HAND_COMMITMENTS_UNKNOWN` / `SUSPENDED` / `OBSERVED_..._CANDIDATE` |
| UNKNOWN/candidate/confirmed | 全仓约定；AA 轨在 `aa_semantics.py` | #28 强化（见 §3） | `aa_semantics` 从不赋 `CONFIRMED`（grep 三赋值点证实） |
| provenance | `source_frame` / `source_sha256` 逐帧穿线 | #28 分析记录 `content_seal`（全文 sha256） | |
| analysis records | 无 | #24 `aa_study_records.py`、#28 `aa_analysis_records.py` | 两个 store 域不同（研究报告 vs 手工分析），非重复 |
| PHH / PokerKit | `state_engine/reviewed_replay.py`、`reviewed_completion.py`（main 已有的离线 oracle 先例） | #28 `aa_phh_shadow.py`（无生产调用方） | 可选依赖 `solver-tools` extra，钉 `pokerkit==0.7.5` |
| solver | **无**（`strategy/gtopen_provider.py` 是 fail-closed loopback 客户端，非求解器） | 无 | 仓库内零 CFR 实现 |

### 2.2 两套扑克规则实现（既有事实，#29 已立决策矩阵）

1. canonical 规则：`strategy/state.py::calculate_legal_actions`（:138-227）+ `contracts.py::DecisionContext.legal_action_types` + `state_engine/action_reconstruction.py`（显式拒绝强制投入 `forced_action_not_supported`）。
2. 视觉旁路：`desktop/aa_semantics.py::_interpret` 只给 `PRICE_DERIVED_CANDIDATE`，`legal_action_verified=False`，理由 `full_history_raise_reopening_and_rules_not_verified`。

这是**已知且被文档化**的双轨，不是暗坑；#29 第 6 行决策为 `REPLACE_INTERNAL`（先差分后删）。审计同意该方向。

### 2.3 重复 / 残留 / dead code（【读码】，由并行只读子审计交叉复核）

- **equity 三套并存**：`equity/montecarlo.py::equity_vs_range`（全组合枚举）vs `equity/enumeration.py::enumerate_equity` vs 调用方 `realtime/equity.py`、`strategy/adaptive_equity.py`、`strategy/multiway_equity.py`。无文档说明各自地位。
- **WPK 旧轨全量保留**：`tools/wpk_state_adapter.py`、`tools/wpk_verify_v1.py`、`docs/WPK-FIRST-HAND-REVIEW.md` 等。文档口径（`README.zh-CN.md`）已是 "AA PRIMARY"，但 WPK 代码未标注"仅回归"。**保留本身可接受**（`AGENTS.md` 明示为回归锚点），但缺一处"此轨冻结"的显式声明。
- **根目录 35+ `_debug_*.py`** 与 `handoff/incoming/` 整套 Python 快照入库：是遗留实验/备份物，应移出或标注。
- **文档漂移**：`docs/architecture.md` §11 仍写 "PokerSense 不依赖任何第三方 poker 库"，而 main 已 vendored `phevaluator` 且 #28 引入 `solver-tools` extra；`docs/research/PLAN-*.md` 九个带日期计划已完成但未归档（`RESEARCH_AND_SELECTION.md:36` 已自承）。
- **未发现"文档已撤回但代码仍在按旧语义运行"的实例**。WPK 是最接近的一项，但其回归用途是被明文保留的。
- **未发现测试仍验证已撤回语义**（抽查 `tests/equity/test_adaptive_equity.py`、WPK 系、`tests/state_engine/`、`tests/test_realtime.py`，均与现行 docstring/文档一致）。`tests/geometry/test_layout_artifacts.py` 有意引用 `handoff/incoming/` 快照（防布局漂移），属有意为之。

### 2.4 双处维护同一规则（#28 引入，需点名）

`desktop/aa_live_context.py::_anchored_opening` **逐条镜像** `tools/aa8_strategy_state_v2.py::AA8HandLedgerCandidate._reset` 的合格判据（同 epoch 恰 1 事件、`MULTI_POST_DEAL_CANDIDATE`、`debits ≥ 3`、本 epoch 无动作）。docstring 已自承 "Mirrors ... exactly"。**风险**：账本规则一旦演进，适配器守卫会静默漂移。建议（不在本轮执行）：把判据提取为共享谓词。

---

## 3. 重点审查 PR #28（head `0b7c40a`，三方一致）

### 3.1 ordinary river terminal 是否真的安全 —— **是（在当前证据下）**

机制（【读码】`aa_semantics.py::_phase`）：

- **C1（latch）**：`full_river`（river + 5 张合法互异公牌）且 `pending_actions == 0` 且非 all-in `closed`，需**连续两个一致观测**才 latch（P0.1 G1）；latch 取连续段**首帧**（`row["frame"] - (streak-1)`），行连续性由 `observe()` 的 gap-reset 保证。G2（`pending_actions == 0`）**只在 latch 时**查阅。
- **C2（结算正证据）**：同 epoch、`confirmed_frame > river_complete_frame` 的正额 `unallocated_positive_cash`。底池为 0 本身永远不够。
- **C3（唯一确认器）**：观察到**新的、非空**字符串 epoch 且确认帧 > river 帧。

我独立核实的关键路径（【读码】`tools/aa8_state_adapter_v2.py::observe`）：

- **被阻塞的行在父适配器里会 `epoch = None` 并返回 `observation_blocked: True`** —— 所以"一个带着新 epoch 的 blocked 行误触发 C3"在生产链上**不可达**：blocked 行的 `observed_epoch` 只会是 `None`，而 C3 要求非空字符串 epoch。方向 fail-closed（丢 pending，而不是误确认）。
- 单元层面 `_phase` 的 epoch-flip 确认分支**不查 `blocked(row)`**：若未来出现第二个 `observed_state_v2` 生产者，该洞会打开。**当前不可达**，记 R6（P2）。

误报/漏报独立评估：

- **漏报方向**：真实回放（【文档】§8，27,287 帧连续回放）5/5 具名 full-river 手全部 latch、全部恰好闭合一次；G1 代价 = 1 帧延迟，G2 代价 = 0。无名 `None` 段 119 帧完整河牌未 latch，文档如实标注"未逐帧归因"——该段按设计永不确认，无功能影响。
- **误报方向**：需要"连续两行一致的误读 + 同 epoch 真钱动 + 新边界"三件套。P0 阶段的对抗 harness 找到过单帧 glitch 假阳性（§4.7 文档），P0.1 G1 已堵（RED→GREEN 测试在案）。残余窗口=两行一致误读，危害上限是**误分类**（钱真的动过才结算），不是提前收盘。
- **跨 hand 泄漏**：`ordinary_terminal` 是一次性事件（`observe()` 末尾无条件清 `ordinary_terminal_event`，覆盖 `_phase` 全部 return 路径）；`last_ordinary_terminal` 快照自带 epoch 与 `belongs_to_current_epoch`，测试跨 4 个 epoch pin 住。**未见泄漏路径。**

### 3.2 UNKNOWN / candidate 能否被错误提升 —— **结构上不能**

`aa_semantics.py` 的 `interpretation_status` 只有三个赋值点：`UNKNOWN`、`VISIBLE_LABEL_CANDIDATE`、`PRICE_DERIVED_CANDIDATE`（【读码】grep 证实）。**从不赋 `CONFIRMED`。** 因此 PHH 闸门的 `CONFIRMED` 动作要求在当前流水线里**没有任何真实手能满足**——这是设计使然，不是漏洞（见 §4）。

### 3.3 HAND_COMMITMENTS_UNKNOWN 能否被当成完整事实 —— **不能**

5 手真实闭合全部携带 `ledger_status: HAND_COMMITMENTS_UNKNOWN` 且 `canonical_verified / card_showdown_verified / rake_verified / strategy_eligible` 全 `False`（【文档】§4.6 + 【读码】`aa_semantics.py:198-199`）。`ORDINARY_RIVER_CLOSED` 证明的是"这手已结束"，不等于"可核对的完整手牌记录"。

### 3.4 REAL_HAND_ACCEPTANCE 应否继续 PENDING —— **应该**

理由：① ledger 缺口未闭合（开局扣款在识别 stack 向量上结构性不可见，§2.3 文档与代码一致）；② PHH 导出对真实手在结构上不可能（§4）；③ 闭合 ≠ 完整记录。任何把 PENDING 改成 ACCEPTED 的动作都需要新的证据源，而不是现有代码的重新解读。

### 3.5 我实跑的验证（【实跑】，非采信 CI）

环境：`pokersense-v6-clean-20260908` Python（含 pokerkit 0.7.5），`PYTHONPATH='src;.'`：

- 深审相关 8 个测试文件（terminal / phh_shadow / opening_boundary / semantics / analysis_records / hand_input×3）：**166 passed, 0 failed**（pokerkit 在场，15 个 `needs_poker_kit` 用例真实执行）。
- 全量基线：**4158 passed / 1 skipped / 413.94s**（本审计在 #28 worktree 上实跑，见 §8）。

---

## 4. PHH + PokerKit 影子预言机（`aa_phh_shadow.py`，614 行，逐行审读）

| 审计问题 | 结论 | 证据 |
|---|---|---|
| 是否真的没有生产调用方 | **是** | 【读码】grep 全 `src/`：`shadow_export/ingest/round_trip` 仅自引用；`aa_server.py` 未接线 |
| 是否只接受 confirmed facts | **是** | 五门：`seat_order`（须 `seat_order_confirmed is True`）、`antes`/`blinds`/`min_bet`（桌规声明且 mode 非 unknown）、`starting_stacks`（逐座）、`actions`（逐条 `interpretation_status == "CONFIRMED"` 且有 `phh` 文本；**任一候选动作 → 整体拒绝**，不允许部分动作表） |
| UNKNOWN 能否被默认值补齐 | **不能** | `_declared`/`_chips` 只返回 None 或合法值；`NOT_EXPORTABLE` 路径不写任何默认；`optional_explicit_utg` straddle 刻意不可映射（声明金额 ≠ 本手已下） |
| NOT_EXPORTABLE 是否泄漏可误用 PHH | **不泄漏** | 拒绝路径无 `phh` 键（测试逐条断言 `"phh" not in report`）。⚠️ 但 `POKERKIT_REJECTED` 且 `document is not None` 时**会带 `phh`**（构造成功、重放/载入失败的情形）——与文档"拒绝绝不带 phh"的口径有出入，记 R7（P2） |
| PHH round trip 是否完整 | **是** | 7 字段（variant/antes/blinds/min_bet/stacks/actions/seats）全部 MATCH；`seats` 可选头部必须写入，否则反向返回 `seat_order=None` 而不是冒充座位号；反向产物状态为 `PHH_ROUND_TRIP`，喂回正向闸门**必被拒**（防"洗闸"，有专项测试） |
| legality / pot / commitment / payout / side pot 正确性 | **基本正确，两处诚实降级** | 非法动作表 → `POKERKIT_REJECTED`（构造在 guard 内）；守恒检查无抽水下成立；**`payout_per_seat` 用 `ChipsPulling` 而非 `payoffs`**（credit=结算余额增量 ≠ 净盈亏，文档 §4 记录了先写错被实测纠正的过程）；`side_pot_split` 是同一次重放的**自洽检查**，诚实标注 `comparable_against_poker_sense=False` |
| `unallocated_positive_cash` ↔ `ChipsPulling` 映射 | **合理** | 两者都是"结算时刻的余额增量"；测试独立验证 650（pull）≠ 350（payoff），错位 `seat_order` → MISMATCH |
| 测试是否自证假绿 | **未见** | 期望值全部独立手算（650/350、900+600 分层、call-then-fold 资格集）；`POKERKIT_NT_REQUIRED` 有对照安装库 `HandHistory.required_field_names["NT"]` 的锚定测试；反例测试（错位订单、非法动作、单池误报回归）证明差分非空断言 |

**审计补充判断（【判断】）**：该 oracle 当前的**真实双引擎差分力只有 `payout_per_seat` 一项**（conservation 在无抽水下接近同义反复；forced_bets 是 `NOT_COMPARABLE`；side_pot 是自洽检查）。这对 P1"首轮"定位是足够的，但不应被引用为"PokerKit 已全面验证 PokerSense 规则"。

**CI 缺口（重要）**：CI 只装 `.[dev,perceptual,desktop]`，**15 个驱动 PokerKit 的用例（含库字段锚定测试）在 CI 全部跳过**。本机绿 ≠ CI 有保护。文档 §9 已自承并留下"是否让 CI 装 `solver-tools`"的待裁决项——审计认为**合并 #28 前必须裁决**（记 R2）。

---

## 5. 开源替代边界（建议，不动代码）

### A. PokerSense 应继续自己维护

1. **capture / vision / temporal / seat-dealer-actor（数据侧）**：上游结构性不覆盖（PokerKit/OpenSpiel 的输入是牌面字面量），且是产品差异化入口。
2. **不确定性层：UNKNOWN/candidate/confirmed 三态、epoch、taint、provenance、账本缺口记账**。【实测】（#29 §1.2-D）PokerKit 把未知牌判"不能赢"、手牌评估抛 `KeyError(Rank.UNKNOWN)`——**不确定性裁决永远不能被上游接管**。
3. **范围模型**（1326 组合权重表 + 置信/证据链，`aa_range_assets_v2`）：上游无对应物；对外互操作用转换器，不降维成 169。
4. **UI / 会话持久化 / 分析记录（`aa-analysis-record-v1` 信封）**：产品 UX。
5. **视频/帧级重放**（`replay/capture_replay.py` 逐帧 sha256）。

### B. 可逐步交给上游（PokerKit 优先，它已是 main 的既有先例）

1. **hand-history 序列化 → PHH**（`WRAP_EXTERNAL`）：PHH 主文件 + PokerSense sidecar（provenance/epoch/taint/抽水参数；**不要把 provenance 塞进 `# commentary`**）。
2. **手牌级确定性重放 → `list(HandHistory)`**（`WRAP_EXTERNAL`，仅手牌层）。
3. **action legality / betting state → `REPLACE_INTERNAL`**：先差分证明等价，再删 `strategy/state.py::calculate_legal_actions` 分支；金额口径（`ADDITIONAL|TOTAL_STREET` vs `cbr` "加注到"）必须在适配层显式换算。
4. **settlement / payout → `WRAP_EXTERNAL`**：只在参与者牌全部已知时调用（未知判负硬前提）。
5. **forced-bet 期望向量 / side-pot 对账 → `USE_AS_ORACLE`**：观测留自研，期望值由上游算，逐已确认手对账；差分零分歧后可降级。
6. **BR / exploitability 定义**：抄 PokerRL `ValueFiller.py:21-101` 的算法定义，用 OpenSpiel（Apache-2.0）在 Kuhn/Leduc 上校准自研 40 行实现。
7. **明确不接**：PokerRL/Deep-CFR（HU 硬断言+仅 Linux+二进制不可审计）、GTOpen（无许可）、TexasSolver/postflop-solver 作代码依赖（AGPL，仅独立进程参照）、rlcard（无增量价值）。

---

## 6. 风险清单（12 条，按严重度排序）

### R1 · P1 — stacked 链的 base 管理与合并顺序是硬依赖
- **Evidence**：#24 base=`codex/aa-live-recognition-v1`、#28 base=`codex/strategy-001-research`、#29 base=`codex/usable-001-hand-review`（【Git】§1.1）；祖先链 #20⊂#24⊂#28（§1.2）。
- **Risk**：单独 merge #28（或乱序 merge）会把 #20+#24 的 25 个提交一起带进 main，或者在 base 分支被删后 PR 悬空；close 链中任何一环，后续 PR 必须 retarget 且其 diff 会突变。
- **Impact**：merge 事故 / 审查范围失控。
- **How to verify**：merge 前 `gh pr view <n> --json baseRefName`；每 merge 一环后确认下一环被 retarget 到 main 且 diff 只剩自有提交（`git rev-list --count origin/main..<head>`）。
- **Blocks merge**：YES（对 #24/#28/#29）。

### R2 · P1 — CI 不覆盖 `solver-tools`，PHH 差分在 CI 无保护
- **Evidence**：【读码】`.github/workflows/ci.yml` 只装 `.[dev,perceptual,desktop]`；【实跑】本机有 pokerkit 时 15 个 `needs_poker_kit` 用例才执行；【文档】`pokersense-phh-pokerkit-shadow-oracle.zh-CN.md` §9。
- **Risk**：闸门口径或 PokerKit 行为漂移只在本地可见；`POKERKIT_NT_REQUIRED` 锚定测试在 CI 跳过。
- **Impact**：fail-closed 证据在 CI 上弱化；未来回归无声。
- **How to verify**：裁决并执行二选一：CI 增加一个装 `solver-tools` 的 job（注意会带入 `phevaluator`，影响其它 skip 项），或显式接受"该层仅本机验证"并写进 README。
- **Blocks merge**：YES（对 #28，合并前需裁决，不一定需要改代码）。

### R3 · P1 — 人工确认存在两条轨：#19 老回放轨 vs #20/#28 新复查台
- **Evidence**：#19 `tools/aa_confirmation_ledger.py` + `ui/aa-replay`（【读码】）；#20 `desktop/aa_review.py`（复查台）+ #28 `/api/hand-input/facts/{issue_id}` 从复查记录带入（【读码】`aa_server.py` diff）。
- **Risk**：两套"人确认了什么"的账本并存，语义各自演化；#19 的 JSONL 台账与复查台记录互不知晓。
- **Impact**：确认事实的单一来源原则被稀释；后续基于"人工确认"的闸门（如 PHH 的 `human_confirmed`）会有两套答案。
- **How to verify**：裁决：#19 概念移植到新轨后 close，或明确分工（回放台账 vs 复查台）写入 AGENTS.md。
- **Blocks merge**：YES（仅对 #19）。

### R4 · P1 — 老 desktop UI 轨与新 AA desktop 轨并存，生死未声明
- **Evidence**：main `desktop/server.py`+`app.py`+`live.py`+`ui/app.js` vs #20 `desktop/aa_server.py`+15 个 `aa_*` 模块+`ui/aa-live/`（【Git】ls-tree）。
- **Risk**：两个服务端、两套 UI 同时存在，新贡献者无法判断哪条是产品轨；老轨测试仍占用套件时间。
- **Impact**：架构腐化；未来改动可能落在错误的轨上。
- **How to verify**：在 AGENTS.md / README 声明老轨状态（维护中/冻结/待删）。
- **Blocks merge**：NO。

### R5 · P2 — `_anchored_opening` 镜像 `_reset` 判据，双处维护
- **Evidence**：【读码】`aa_live_context.py:116-133` vs `tools/aa8_strategy_state_v2.py::_reset`。
- **Risk**：账本合格判据演进时，适配器守卫静默漂移，重新引入 #28 已修的"重复 epoch 覆盖已解析账本"缺陷。
- **Impact**：P0 级缺陷回归的潜伏期。
- **How to verify**：提取共享谓词，或加一个"两处判据一致"的契约测试。
- **Blocks merge**：NO。

### R6 · P2 — `_phase` 的 C3 确认分支不查 `blocked(row)`（单元级隐患，当前生产不可达）
- **Evidence**：【读码】`aa_semantics.py:177-200`（确认在 blocked 检查之前）；【读码】`tools/aa8_state_adapter_v2.py::observe`（modal 行 `epoch=None` + `observation_blocked`，使该路径不可达）。
- **Risk**：若未来出现第二个 `observed_state_v2` 生产者（不重放父适配器的 modal 语义），blocked 行携带新 epoch 会误确认普通终局。
- **Impact**：潜伏的误确认路径。
- **How to verify**：在确认分支加 `and not blocked(row)` 或注释+测试钉住"blocked 行永不确认"（现有测试只覆盖同 epoch blocked）。
- **Blocks merge**：NO。

### R7 · P2 — `POKERKIT_REJECTED` 在重放阶段失败时携带 `phh` 键
- **Evidence**：【读码】`aa_phh_shadow.py:498-504`（`if document is not None: refusal["phh"] = document`）。
- **Risk**：与"拒绝绝不带 phh"的口径有出入；只看"有没有 phh 键"的消费者会捡到一份重放失败的 PHH。
- **Impact**：低——该 PHH 已通过 PokerKit 解析器，失败发生在重放/载入阶段；但口径应当统一。
- **How to verify**：要么文档写明"NOT_EXPORTABLE 才保证无 phh"，要么失败报告里把 phh 挪进 `evidence` 子键。
- **Blocks merge**：NO。

### R8 · P2 — equity 三套实现与 actor 两路跟踪并存，无从属文档
- **Evidence**：【读码】`equity/montecarlo.py` vs `equity/enumeration.py` vs `realtime/equity.py`/`strategy/adaptive_equity.py`/`strategy/multiway_equity.py`；`realtime/hand_boundary.py::ActorTracker` vs `tools/aa8_actor_ring_v2.py`。
- **Risk**：同一问题多个答案，调用方各取所需。
- **Impact**：长期维护成本；结果不一致时无法定位"哪个是对的"。
- **How to verify**：为每套写一行"何时用哪个"的归属说明。
- **Blocks merge**：NO。

### R9 · P2 — oracle 的真实双引擎差分力只有 `payout_per_seat` 一项
- **Evidence**：【读码】§4 表格；`forced_bets` = NOT_COMPARABLE；`side_pot_split` = 自洽检查；conservation 在无抽水下近同义反复。
- **Risk**：外部引用"PokerKit 已差分验证 PokerSense"时高估覆盖。
- **Impact**：结论通胀。
- **How to verify**：报告与 README 中写明当前差分覆盖清单。
- **Blocks merge**：NO。

### R10 · P1 — #17 与 main 冲突且功能被 #20 取代，仍挂起
- **Evidence**：【Git】CONFLICTING、落后 2 merge；【读码】#20 无其全部 5 个文件；glyph 字典逐字节一致；负结论（0 合格素材）对视觉方向有决策价值。
- **Risk**：负结论随 PR 关闭而丢失；或长期挂起造成"还有待办"的错觉。
- **Impact**：决策依据丢失 / 仓库噪声。
- **How to verify**：把结论与证据迁入 issue 后 close。
- **Blocks merge**：YES（对 #17 本身：不应 merge，应 close）。

### R11 · P2 — 文档漂移：architecture.md §11 与九个已完成 PLAN-*.md
- **Evidence**：【读码】`docs/architecture.md` §11 "不依赖任何第三方 poker 库" vs main 已 vendored phevaluator + #28 `solver-tools`；`RESEARCH_AND_SELECTION.md:36` 自承九个计划未归档。
- **Risk**：新贡献者按过期架构事实行动。
- **Impact**：低-中。
- **How to verify**：更正 §11；PLAN-*.md 归档。
- **Blocks merge**：NO。

### R12 · P2 — 并行 PR 间的文档文本冲突（机械性，无代码冲突）
- **Evidence**：【Git】`git merge-tree` 实测：#28×#26 冲突 1 文件（`AGENTS.md`）；#28×#19 冲突 3 文件（`AGENTS.md`、`README.md`、`README.zh-CN.md`）。
- **Risk**：merge 顺序不当时需要手工解文档冲突；`AGENTS.md` 是"当前状态"追加日志，冲突解决可能丢行。
- **Impact**：低。
- **How to verify**：按 §7 顺序 merge；解冲突时保留双方追加块。
- **Blocks merge**：NO。

---

## 7. PR 最终暂定分类

| PR | 当前作用 | 是否被后续包含 | 独有价值 | 主要风险 | 暂定动作 | merge 前条件 |
|---|---|---|---|---|---|---|
| **#17** | glyph 留出审计（负结论） | 功能被 #20 取代（`aa8_special_modes.py` + `AA-LIVE-MONITOR-V1`）；分支文件在 #20 上不存在 | 负结论本身（0 合格同 session 素材，0 glyph 行）有决策价值 | R10：与 main CONFLICTING、落后 2 merge | **SUPERSEDED** | 不 merge；结论+证据迁入 issue 后 close |
| **#19** | 人工确认台账（老回放轨） | 否（独立于链） | 哈希链 JSONL 台账 + 逐字段来源绑定；解除 #20 `source_issue_id` 占位 | R3：与新复查台双轨；R12 文档冲突 | **BLOCKED** | 裁决人工确认归属（移植概念 or 明确分工）后再定 |
| **#20** | 链基座：AA 识别会话+复查台+三人河牌内核 | 被 #24/#28 包含（作为历史） | 全部 15 个 `aa_*` + 7 个 `*_v1` 研究模块，main 上不存在 | R1（链基座，merge 它即改变 #24/#28 的 diff 口径） | **MERGE_CANDIDATE** | 链式 merge 第一步；全量测试绿；确认 #24 被正确 retarget |
| **#24** | 策略诊断+离线复盘接入 | 被 #28 包含（作为历史） | `aa_study_records` + 路径诊断/范围实验工具 + 5 篇研究文档 | R1 | **MERGE_CANDIDATE** | 随链在 #20 之后；基线测试绿 |
| **#26** | Git 同步链路修复 | 否（独立） | `git_sync.py`：凭据助手顺序/超时/退出码/三方 SHA；离线 dry-run 通过；无危险操作 | R12（AGENTS.md 文本冲突） | **MERGE_CANDIDATE** | 无硬条件；建议先于主链 merge 以减小冲突面 |
| **#28** | 真实牌局输入+普通河牌终局+PHH 影子 oracle | 包含 #20/#24（作为历史） | U1/U2 全链 + P0/P0.1 终局（真实回放 5/5 闭合）+ P1 oracle；§3/§4 深审通过 | R1、R2、R5、R6、R7、R9 | **MERGE_CANDIDATE** | ① 在 #20/#24 之后；② R2 裁决（CI solver-tools）；③ `REAL_HAND_ACCEPTANCE_PENDING` 字样随代码一起进 main |
| **#29** | 开源架构基准（纯文档） | 否（+1 doc commit，基线 `fbda7c5` 落后 #28 六提交） | 17 行决策矩阵 + 许可核对 + 未知信息失败实测 | R13（基线戳旧；研究轮可接受） | **RESEARCH_ONLY** | 无代码风险；建议 rebase 到 #28 head 后随链 merge，或单独 merge |

### 推荐 merge / close 顺序（独立推导）

```
1. #26   （独立、低面、先清掉；解 AGENTS.md 冲突时保留双方追加块）
2. #20   （主链基座）
3. #24   （确认 GitHub 已把它 retarget 到 main 且 diff 只剩自有 10 提交）
4. #28   （确认 retarget 后 diff 只剩自有 20 提交；R2 裁决先行）
5. #29   （rebase 到新 main 或直接 merge 单文档）
6. #19   （R3 裁决后：移植概念 close，或明确分工后 merge）
7. #17   （结论迁 issue 后 close，不 merge）
```

---

## 8. 审计者实跑清单与最终基线

| 项 | 结果 |
|---|---|
| `git fetch origin --prune` + 7 PR 元数据 `gh pr view` | §1.1（2026-09-17 实拉） |
| `git merge-base --is-ancestor` × 6 组 | §1.2 |
| `git ls-tree` main vs #20 文件对比 | §1.3 / §2.1 |
| `git merge-tree` #28×#26、#28×#19 | R12 |
| 深读代码：`aa_phh_shadow.py`（全 615 行）、`aa_semantics.py`（全 393 行）、`aa_live_context.py`（全 301 行）、`aa_analysis_records.py` / `aa_hand_input.py`（头部与校验段）、`aa8_state_adapter_v2.py::observe`（modal 段）、#20 `aa_review.py` / `threeway_river_v1.py`（头部与 `_validate`）、#24 `aa_study_records.py`（头部） | §3 / §4 |
| 深读测试：`test_aa_ordinary_river_terminal.py`（全 400 行）、`test_aa_phh_shadow.py`（全 510 行） | §3 / §4 |
| 深读文档：P0/P0.1 终局报告（443 行）、PHH 影子 oracle 报告（241 行）、#29 基准（484 行） | §3 / §4 / §5 |
| 【实跑】8 个深审测试文件 | **166 passed / 0 failed**（pokerkit 在场） |
| 【实跑】#28 worktree 全量 `pytest -v` | **4158 passed / 1 skipped / 43 warnings / 413.94s**（2026-09-17 本审计实跑；与 P0 文档 4128 + P0.1 五项 + PHH 影子 25 项的账目一致） |
| 并行只读子审计 ×2（main 架构地图；#17/#19/#26 diff 审查），关键结论均由我本人用 `origin/*` 引用独立复核 | §1.3 / §2 / §7 |

### 验证盲区（如实声明）

- 未重跑 27,287 帧真实回放（约 23 分钟，私有池 `G:\PokerSense_archive`）；§3.1 的回放结论引用【文档】+ 代码机制核实，未升级为【实跑】。
- 未逐行读 `aa_analysis_records.py` 全部 889 行与 `aa_hand_input.py` 全部 861 行（读了头部契约、校验段与全部相关测试）。
- 未审查 `ui/aa-live/*.js` 前端逻辑（JS 测试 968+548 行未逐行读）。
- `AGENTS.md` 的追加日志（#20 +180、#28 +238 行）只做了抽查，未逐条核实其中的过程声明。
