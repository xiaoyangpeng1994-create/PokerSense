# 多人翻牌前开源策略资产调研

## 结论

本轮没有找到可以直接满足 PokerSense `PRV-003` 的、许可清楚且对 3–9 人每个人数、
位置、筹码深度和行动线提供独立求解结果的开源多人 GTO 资产。因此 `PRV-003`
仍保持未完成，不能用 6-max 牌表、相邻人数继承或多份 HU 策略拼接来冒充。

本轮接入的只有 `bmorrow10/preflopR` 中**明确写出的 6-handed 与 9-handed unopened
open-raise 列表**，并固定标记为低置信度 `HEURISTIC`。它用于验证 Provider、路由、
资产审计和 UI 来源披露链路，不是多人 GTO 的替代品。

## `bmorrow10/preflopR`

- 仓库：<https://github.com/bmorrow10/preflopR>
- 固定 commit：`aed511d0451aea33a14f7e9204595fc2211f233f`
- license：MIT
- 固定源文件 SHA-256：
  `75a8a7288bbe39361313ea4183514784f91ab462f9d4c99d3b9d50a8394b6090`

上游 README 明确说明牌表只是 GTO-approximate，**不是 solver-derived**。源码实际只写出
2、6、9 人牌表；所谓 3–5 和 7–8 人支持来自 fallback，而不是对应人数的独立策略。
源码的 fallback 还存在额外风险：优先尝试 9 人同位置，并在没有任何键时回退到最宽的
`9_BTN`。例如 BB 没有显式 RFI 牌表，却可能被显示成 BTN 牌表。

PokerSense 的处理规则是：

1. 只导入源码中明确存在的 6/9 人键，不执行 fallback；
2. 排除 BB，因为 unopened RFI 不存在需要 BB 主动开池的正常决策；
3. 仅匹配 NLHE cash、preflop、unopened、100BB、no ante、no rake；
4. 输出只有二元 raise/fold 频率，不补造 raise size 或 EV；
5. 输出 `match_kind=heuristic`、`confidence=0.4`，并携带 commit、源文件 hash、
   生成资产 hash、range key 和限制列表；
6. Exact Provider 与该结果同时存在时，Router 必须选择 Exact。

生成工具 `tools/import_preflopr_open_ranges.py` 直接解析固定上游 R 文件，只接受评审过的
13 个显式键。上游键集合、表达式或内容变化都会使 `--check` 失败。自动测试进一步遍历
每个键的全部 169 手牌类别，验证二元输出与提交资产一致。

## 未采用的候选

### `MatthewPDingle/GTOpen`（2026-08-23 复核）

- 仓库：<https://github.com/MatthewPDingle/GTOpen>
- 当前公开 API 明确提供 2–9 人 Preflop Lab、limp/cold-call/raise/ante/rake、逐节点
  action-major 169 类策略，以及逐玩家 best-response gap。
- 这比静态 6-max 牌表更接近 `PRV-003`：若许可问题解决，可通过
  `/api/preflop/status` 和 `/api/preflop/node` 导出带收敛证据的版本化节点，再进入
  `JsonStrategyAssetProvider` 和 Golden parity 流程。

当前不能接入，原因有两层：

1. 2026-08-23 检查的仓库根目录没有 `LICENSE` 文件，直接读取
   <https://raw.githubusercontent.com/MatthewPDingle/GTOpen/master/LICENSE> 返回 404；README
   中自称 “open-source” 不能替代明确许可。GitHub 的
   [Licensing a repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
   说明，无许可证时默认版权法适用，不能据此复制、修改或分发。
2. 上游明确披露 Preflop Lab 在翻牌终点使用 equity-realization model；多人 equity 使用 product
   approximation，完整 postflop solver 仍是 Heads-Up。3 人以上 CFR 给出的是该模型的一组均衡，
   不是唯一完整游戏 GTO。因此即使未来补充兼容许可，也必须标记 `MODEL`/有界近似、保存每名
   player gap 和模型版本，不能直接宣传为完整多人 GTO。

2026-08-23 在本机把上游固定到提交
`4aee435bdeb155b25f0c8140e707a8342ce4356f`，仅放入 Git 忽略的
`.upstream/GTOpen/` 研究目录，没有复制源码或策略资产到 PokerSense。Apple M1 Pro
CPU-only Release 构建成功；上游 Solver 测试为 104 passed、1 ignored、0 failed。

本地 API 实测了一个 3 人 20BB、BTN/SB/BB、2BB open、最多一次 raise、无 rake、
`raw` realization 的最小场景：估算/构建均为 13 nodes、6 action nodes、0.016224 MB；
求解在第 25 次检查时报告三名玩家 gap
`[0.0001664055, 0.0001695890, 0.0001211527]`，总 gap `0.0004571471 BB`。
根节点返回 BTN 的 `Fold 58.858263% / Raise 2BB 41.141740%`，并提供
`2 actions × 169 classes = 338` 个策略值。该结果只证明 API、数组方向、CPU 构建和
最小多人求解路径可执行，不是独立 Golden，也不证明默认 realization 模型、任意深树或
真实 WPK 场景的策略正确性。

结论：保留为“待上游许可证 + 独立 Golden 验证”的优先本地研究候选。可以通过本地 API
继续做 Adapter/性能实验，但不得复制、打包、发布源码或生成资产，也不加入当前正式
Provider 能力清单。

随后实现了 PokerSense `GTOpenPreflopProvider`，不复制上游代码，只访问 loopback API。
真实端到端使用同一固定 checkout，在三人 20BB、BTN/SB/BB、Hero AKo、允许 limp、
2/2.5/3BB open 与 all-in 的场景构建 4,270 nodes、1,710 action nodes、5.771688 MB；
CPU-only 100 iterations 后报告 model gap `0.008207490846030292 BB`，Adapter 成功读取
Hero class index 155，并保留 Fold/Call、三个 raise size 和 all-in 的独立频率。该数据已作为
`MOCK-GTOPEN-3P-ROOT-AKO` 的 synthetic-local-model trace 保存，仅用于回归合同；它不是
独立正确性 Golden、许可资产或真实 WPK 策略证据。

### `exinori/DCFR-SOLVER`

仓库采用 MIT 并公开 6-max preflop MCCFR，但只覆盖 6-max，不满足 3–9 人逐人数能力边界；其
postflop 也不是多人 Provider。可以作为未来 6-max 单点交叉验证来源，不能单独完成
`PRV-003~005`。

### `AHTOOOXA/poker-charts`

仓库代码虽然采用 MIT，但牌表文件标明来自 GreenCharts PDF、Pekarstas 和 GTO Wizard
等第三方来源。仓库许可不能自动证明这些衍生策略数据可再分发，因此 PokerSense 不复制、
不打包这些牌表。

### `chirenonhive/poker-solver`

项目采用 MIT，但公开能力是 Heads-Up，不提供 3–9 人独立多人策略，因此不能用于
`PRV-003`。

## 后续满足 `PRV-003` 所需证据

真正的多人 Preflop Provider 至少需要：逐一声明 3–9 人覆盖；位置和行动线；stack、ante、
rake 与 bet-size abstraction；源版本和资产 hash；每个人数的 Golden 查询；以及明确许可和
求解质量指标。缺少任一项时只能降级为 heuristic 或拒答。
