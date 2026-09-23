# 对手行动似然校准 V1：历史数据门槛与合成控制

## 唯一结论

**`INSUFFICIENT_DATA`。** 已审计的历史 A 扑克结构化资料没有一个具备完整分母和决策前合法状态的可准入 ACTION OPPORTUNITY。不能从已出现的 25 条对手动作候选或 120,089 帧中拟合真实玩家级、总体或牌力条件行动似然，也不能给它们制造跨 session 的 held-out 校准结论。离线合成控制模型与策略配对只是检验工具是否工作；生产范围、感知/真值语义、BASELINE V1、实时建议及自动操作均未改变。

历史审计先于模型代码固定在 `bda4bbb10bed8b83869bdcb820d4251d265a8d98`，公开聚合清单 SHA256 `d761e42050cb7d67ac9efd3eab1057bf9ed151ffe26ec033197bcd5c46eb3291`。详细来源、人数和 UNKNOWN 缺口见 `docs/OPPONENT-ACTION-LIKELIHOOD-DATA-AUDIT-V1.zh-CN.md`。`tools/audit_action_likelihood_historical_v1.py` 本地只哈希四份结构化证据，不读取媒体；当前回读为 `VERIFIED_INSUFFICIENT_DATA`，eligible real opportunities **0**。CI 的 `--public-only` 明示不复验本机私有文件。

## 决策时与离线标签严格分离

合成机会行包括 session、hand、opponent 与物理 seat、规则指纹、5 张公开牌、位置与行动顺序、当时的 pot/stack/投入、此前公开行动、完整有限合法菜单、实际动作、来源 SHA 和 UNKNOWN 原因。后来摊牌/牌力字段不在该行的白名单；独立的离线标签侧车对全部合成行均记录 `LATER_STRENGTH_NOT_AVAILABLE`。模型拟合只读取预登记训练 session 的机会行，拒绝额外字段、缺失、哈希失配、重复机会或混入验证 session。留出评分保留全部行，零概率、未知上下文与拒绝项分别计数。

本轮最简单模型是 `alpha=1` 的 Dirichlet 频数估计，条件仅为**公开合法菜单、精确价格比与已声明规则**；`player_dirichlet_v1` 另用稳定合成 opponent ID，`pooled_dirichlet_v1` 合并对手。牌面、位置、筹码和历史保留在机会记录中，但当前有限合成控制没有足够上下文覆盖去训练它们的作用。没有隐藏牌特征或经审计的揭示标签，因此估计的是 `P(action | public menu, price[, player])`，**不是** `P(action | hidden holding, public state)`；不能声称这些模型会给范围后验提供有效手牌辨别率。

## 先冻结协议，后实现模型

合成协议、生成器及数据在 `dc0bd69775eedcd88eed3c3a9fee6cb94bd14f68` 提交，先于模型实现或评分。协议 SHA256 `61e21217fa7dfb41837538731fa9628f4f6abb45266679ebfc98a5c9177031dc`，384 个完整合成机会文件 SHA256 `7cbbc108f8a801f5f4a56f57a7dea7325557f9bddd15850b8427515715ecac45`；离线标签侧车 SHA256 `27d15e98825a1d816b0b966d404e40e7eb9b026f529473322d6f7d39913ca64b`。同一规则的合成 session 整体划分：训练 4×48=192，风格稳定留出 2×48=96，玩家风格互换留出 2×48=96。两个模型及 `alpha=1` 在协议中预声明；没有根据留出结果选择/调参。协议写明生成机制，作者可见，属于**时间顺序上前瞻的合成控制**，不是盲法或真实泛化测试。

| 合成留出（各 96 机会） | 模型 | log loss ↓ | 多类 Brier ↓ | 5 档最高动作 ECE ↓ |
| --- | --- | ---: | ---: | ---: |
| 风格稳定 | pooled | 0.9038 | 0.5772 | 0.0776 |
| 风格稳定 | player | **0.7510** | **0.4566** | **0.0453** |
| 风格互换 | pooled | **0.8200** | **0.5414** | **0.0776** |
| 风格互换 | player | 1.2959 | 0.8245 | 0.4327 |

四次评分均覆盖全部 96 个机会，零概率、未知上下文回退和拒绝均为 0。身份特定频数能记住合成稳定风格，也会在同一身份换风格时严重失配。这里的 ECE 是 96 个合成机会上的描述值，没有真实样本置信区间；两种模型的比较不能为历史 A 扑克选择赢家。

## 冻结策略评估的条件影响

只在 BASELINE V1 已支持的 `n6-facing_bet` 与 `n6-unopened` 公开局面进行配对。每个模型/局面只从训练机会编译一份公开历史策略表，在已冻结的稳定与漂移世界上复用；世界评价仅调用未修改的 `threeway_policy_evaluation_v1.evaluate_policy_book`，世界 SHA 两方相同，联合组合/节点预算与 V1 相同。BASELINE V1 六份原始表和 30 个曝光开发世界未改；旧 30 项自检仍为 `c1a725f3f95a76261fb041af200e45bae02b6836202b36ec56578f0afd680aae`。

8/8 合成配对完成，无 blocked 或 fallback；相对 V1：4 项正差、0 项负差、4 项相等，4 项根动作不同。四个 facing-bet 配对均从 V1 的 call 改成 raise-to-80，条件 EV 增加精确 `1921/40` = 48.025 筹码；四个 unopened 配对动作和 EV 不变。pooled/player 在这些有限世界中落到同一根动作，两个世界的 facing-bet 增益也相同，故**此策略压力样本不能区分模型选择**。模型知识差来自同一策略内核，不是 GTO regret；合成世界也不是现实对手频率。行为漂移已经在 held-out 校准中暴露显著退化，不能用四个正 EV 行掩盖它。

独立审阅发现并复现一项桥接缺陷：若把训练菜单里的 `bet:20` 折成通用 `bet`，冻结树会给 `bet:10/20/40/80` 各分配同一权重，改变下注总概率。现已保留精确 `bet:20`、`raise:40` 键；树内核的定向测试核对了额外金额权重为零。**策略树仍有训练支持之外的状态**：训练 facing-bet 价格为 `1/7`，树在不同 Hero 下注后可出现 `2/13`、`4/17`；`raise:40` 不合法时剩余动作会重新归一化。冻结世界中的行为生成权重也允许其他金额。因此这 8 项仅是明确披露外推的合成接线诊断，不能称整棵树的响应概率已获校准。预测 API 现仅接受决策前公开快照与已知合成对手 ID；实际动作只供离线拟合/评分。

逐项结果在 `configs/strategy/evaluation/action-likelihood-synthetic-results-v1.json`：规范字节 SHA256 `0b51702f4ea3dd89a37c423f7a3cd93f403dc727e3697943d7c91e74e31cbf55`；排除计时的摘要 `1cb28a511c080d06ac1a7815fec4b5a1fc726ace87569eb26b5d21f8e7f1f0a5`；候选模块 SHA256 `3bc8cfd384f3e360980edf25f68a803fe759a5ef91405d8c77f7890c4c2ae3de`。

复现：

```powershell
$env:PYTHONPATH='src;.'
$env:PYTHONUTF8='1'
python tools/audit_action_likelihood_historical_v1.py
python tools/generate_action_likelihood_synthetic_v1.py check
python tools/strategy_evaluation_v1.py check
python tools/evaluate_action_likelihood_v1.py --output <new-result-path.json>
```

历史审计 tool 的默认路径只在本机私有证据存在时可复验；其他机器使用 `--public-only` 并标记未复验。输出路径须新建，不能覆盖既有结果。`REAL_HAND_ACCEPTANCE_PENDING=YES`，经验策略 `NOT_ASSESSED`。

**推荐的下一步**是独立完成同规则、含 Hero 与对手、缺失行不删的行动机会总账，审阅动作前 actor/menu/state/onset 和稳定 session/player 身份，再冻结真实 session 留出协议；若要估计隐藏牌条件似然，还需要来源绑定且考虑摊牌选择偏差的离线标签。这是数据前提建议，本任务不自动启动 RANGE_POSTERIOR_V2。
