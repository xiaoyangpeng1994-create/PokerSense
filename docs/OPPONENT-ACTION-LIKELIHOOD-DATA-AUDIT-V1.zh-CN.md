# 对手行动似然：历史 A 扑克数据审计 V1

## 审计结论

在 `main af67b4bb3ba56da2d4fdb70e0731583afc7aaf40` 上，现有结构化历史资料中**可准入的真实 ACTION OPPORTUNITY 为 0**。行动字形、actor cue episode、帧和河牌检索片段分别是不同的候选单位，不能充当完整机会分母。因此当前既不能拟合玩家级行动似然，也不能拟合同规则总体或“玩家类型”似然；后续完整留出集也无法由这些资料组成。

本轮先读既有结构化 JSON/Markdown 与代码合同，并独立核对四个私有来源的文件 SHA256。没有读取或上传视频、图片，没有采集或使用摊牌作为决策时特征。机器可读的公开聚合审计在 `configs/strategy/evaluation/action-likelihood-historical-audit-v1.json`；它不包含原始玩家身份或私有记录。

## 单位与实际覆盖

一个可用单位必须是**玩家轮到行动时的一次机会**，包括没有识别出实际动作或任一字段 UNKNOWN 的机会。它须绑定可审阅的 session/hand/player 身份、完整游戏机会总账、当时的公开牌、位置和行动顺序、可靠的底池/筹码/此前公开行动、完整合法动作及金额语义、实际动作、严格早于动作开始的证据、规则指纹、特殊模式状态和拒绝原因。后来的摊牌或牌力只能放入按 hand/opportunity 绑定的离线标签侧车，不能进入决策时特征。

| 来源单位 | 核对数量 | 不能提升为机会数据的原因 |
| --- | ---: | --- |
| 已复核可见动作 | 36 候选、30 MATCH、6 误报；30 中对手 25、Hero 5，河牌动作总共 2 | 只从出现动作字形的片段抽取；完整合法菜单 0、决策前因果锚与状态缺失，漏掉的机会没有分母。 |
| actor cue episode | 32；其中 28 与动作绑定、4 无动作候选、2 个动作无 episode，合并复核队列 34 | cue 不等于实际轮到行动；841 帧 actor UNKNOWN、28 帧 unsupported，自动 check/timeout 和 detector miss 未排除。 |
| 2026-09-22 七池连续基准 | 120,089 顺序帧、57 river retrieval fragment | 120,089 行均 `ANALYSIS_DISABLED`，完整合法状态 0；片段不是独立手或行动机会，AI reference 不是人类 gold。 |

30 条可见动作来自同一次 AA8 录制的两手候选完整牌局；其中 `aa8_late_dev_hand_005` 含 bomb 特殊模式。另一个物理录制只到大厅；旧九座来源的规则与完整手未证实。当前没有第二个可确认同规则的 gameplay session，不能按 session 作真实训练/验证隔离；物理座位也不能当稳定对手身份。两套现有机会合同都没有审阅过的离线摊牌牌力标签。

### 来源回执

以下为 `G:/PokerSense_private/` 下的相对路径及实测 SHA256；只引用聚合结论，不将文件提交到公开仓库。

| 角色 | 相对路径 | SHA256 |
| --- | --- | --- |
| 决策 readiness | `aa8_decision_opportunities_v1_20260914_v10/readiness.json` | `e10be9812b7beb24cd8da4792166fbc94903493831192f2ddc799a2f474cec10` |
| 可见动作复核 | `aa8_action_truth_review_20260914_v1/result-v4.json` | `264be3050dc1fd7834caaff8358dd1cd1938b00073503a8a21065eed52850e73` |
| actor episode | `aa8_actor_episode_census_20260914_v6/episodes.json` | `1bb9cfda2a12ed49fb167ffdf8f243c7a7b6e20644fb0fce875979e1a2cfd9b8` |
| 七池连续基准 | `historical-benchmark-20260922/continuous-aggregate.json` | `349d9895dd07962a8d2d3b8fce6fdf0b2f6b11d7e38913d95a7dae14f29f5461` |

`decision_opportunities_v1.py` 已规定完整机会总账、UNKNOWN 原因和整组拒绝；`build_aa8_decision_readiness.py` 当前只生成候选动作总账，并把身份、决策前 board/pot/stacks/history、合法菜单和规则指纹置空。`opponent_dataset_v1.py` 的简化接口也不能凭 JSON 中的 reviewer/哈希认证全机会覆盖。仓库只有合成样例与 `NOT-DATA` 模板，没有运行时写入完整机会的链路。

## 对建模的约束

- **玩家级：INSUFFICIENT_DATA。** 无跨手/跨 session 可确认的稳定对手身份，也没有每个玩家的全部机会分母。
- **总体或 archetype：INSUFFICIENT_DATA。** 同规则与特殊模式不可混合；现有候选不是完整机会，不能从 25 条可见对手动作估计总体频率。
- **牌力条件似然：INSUFFICIENT_DATA。** 可见动作的后来底牌真值缺失；只挑摊牌手也会产生选择偏差。公开信息条件下的 `P(action|public state)` 即使以后可估，也不能单独识别 `P(action|hidden hand,public state)`，因而不能直接产生有辨别力的范围更新。

本审计的下一工程步骤只能是冻结缺失行也保留的 prospective 协议，给工具做合成控制验证，并让真实数据门槛继续拒绝模型拟合。合成评分可证明代码运转，不能称历史行为已学习或真实校准通过。`REAL_HAND_ACCEPTANCE_PENDING=YES`、经验策略 `NOT_ASSESSED`。
