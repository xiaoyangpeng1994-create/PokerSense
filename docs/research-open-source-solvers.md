# 开源策略层调研（2026-09-06）

> 背景：owner 明确「**自己用，不考虑商用**」，许可约束从"必须 MIT 级"放宽。
> 本次调研重新评估此前被许可红线否决的候选。

---

## 一、核心结论（先看这个）

### 1. 不存在开源的「多人（3+）翻后」NLHE 求解器——这是行业空白，不是我们没找到

三路独立证据：

1. 关键词 `multiway poker solver CFR` 全站仅 **1 个结果**（GTOpen）；`nlhe solver 3-handed OR multiway OR multiplayer postflop` 仅 3 个结果且全部不相关
2. **GTOpen 自己的描述**写得很清楚："**heads-up postflop** CFR (CPU/CUDA), **multiway Preflop Lab**" —— 连它也是翻后单挑
3. 商业软件同样如此：PioSOLVER / GTO+ 的翻后求解也都是 heads-up

多人翻后求解的复杂度爆炸（博弈树随人数指数增长 + 需要 abstraction），至今是研究前沿，Pluribus 级别的成果需要 120GB 内存 + 数十小时训练。

### 2. 但这**不阻塞**我们——因为"6-8 人桌"要拆成两个完全不同的问题

| 阶段 | 人数模型 | 现状 |
|---|---|---|
| **翻前** | **multiway 6-8 人** | ✅ 已用上游 9 人同名键派生解决（conf 0.3） |
| **翻后** | 翻前动作打完后，**多数底池只剩 2 人** | 单挑求解器可直接用 |

**关键桥梁**：`postflop-solver` 是**唯一实现 bunching effect** 的求解器——即正确扣除已弃牌玩家的牌对牌堆的影响，"supports up to four folded players（**6-max game**）"。这正是让"单挑求解"在 6 人桌上准确的那块拼图。

**结论：翻前多人图表 + 翻后单挑求解（带 bunching）= 覆盖真实牌局的大多数决策。**

### 3. 延迟现实：河牌可行、转牌勉强、翻牌必须预计算

参考 `TexasHoldemSolverJava` 官方基准（与 PioSolver 对比）：

| 街 | PioSolver | TexasHoldemSolverJava | 我们的可用性判断 |
|---|---|---|---|
| flop | 7.91s | 98s | ❌ 实时不可行，须离线预计算 |
| turn | 1.5s | 4.21s | ⚠️ 勉强（真人决策窗口几十秒） |
| river | 0.56s | **0.06s** | ✅ 实时可行 |

`postflop-solver` 声称性能超过 PioSolver/GTO+，实际会明显好于上表，但**翻牌圈的树规模仍是数量级差距**。

---

## 二、候选对比

| 项目 | ★ | 语言 | 许可 | 人数 | 覆盖 | 集成形态 | 维护 | 结论 |
|---|---|---|---|---|---|---|---|---|
| **b-inary/postflop-solver** | 366 | Rust | **AGPL-3.0** | 2（+bunching 适配 6-max） | 翻后 | **库（crate）** | ⚠️ 2023-10 停止 | 🥇 **首选** |
| **bupticybee/TexasSolver** | 2535 | C++ | **AGPL-3.0** | 2 | 翻后 | **预编译 Windows 二进制 + CLI** | ✅ 活跃 | 🥈 **最省事** |
| bupticybee/TexasHoldemSolverJava | 913 | Java | **MIT** | 2 | 翻后 | jar + JRE11 | ❌ 已停维护 | 🥉 许可最宽松但慢 |
| noambrown/poker_solver | 165 | Python/C++ | 未标注 | 2 | **仅河牌** | CLI + JSON | ✅ 活跃 | 河牌专项，可作为交叉验证 |
| krukah/robopoker | 218 | Rust | **MIT** | 多人（Pluribus 路线） | 全街 | 库，需 Postgres | ✅ 活跃 | ❌ 见下 |
| MatthewPDingle/GTOpen | 8 | Rust | **无** | 翻前 2-9 / 翻后 2 | 翻前+翻后 | HTTP :3737 | ✅ | ❌ 已在册否决 |
| a9876543245/DEEPFOLD-SOLVER | 411 | C++ | **All rights reserved** | 2 | 翻后 | 桌面端 | ✅ | ❌ 非开源 |
| ZenithPoker/ZippySolver | 42 | C++ | 未标注 | — | — | WSL 编译 | ❌ README 止于「7. TODO」 | ❌ 烂尾 |

### 淘汰理由（重要，别再踩）

- **DEEPFOLD-SOLVER**（411★，容易误判为热门好项目）：README 末行写 "© DEEPFOLD — All rights reserved"，"source is published **for transparency**"，使用需 **Google OAuth 登录 + DEEPFOLD PRO 会员**。是伪装成开源的商品。
- **robopoker**（MIT + Pluribus 路线，看起来最对口）：系统需求 **16 vCPU / 120GB RAM / PostgreSQL**，river abstraction 3.02GB，训练 40 小时；而官方对 Slumbot 的基准**最好成绩是 −22.8 bb/100**（负=输钱）。研究工具，不是可用的策略源。
- **GTOpen**：本轮复核 `license: None` 且上游仍无 LICENSE 文件。此前实测 7-max 92–164s / 8-max 499s、gap 0.021–0.048 未收敛。
- **ZippySolver**：只支持 WSL 编译，README 第 7 步直接是 "TODO"，未完工。

---

## 三、首选方案细节：`b-inary/postflop-solver`

**为什么是它**

- **唯一的 bunching effect 实现**，官方明确 "supports up to four folded players (6-max game)"——为我们的场景而生
- Discounted CFR（γ=3.0，策略在 4 的幂次迭代重置）+ 多线程 + 作者手工审汇编确保 SIMD；作者称性能**超过 PioSOLVER 和 GTO+**
- 32-bit 浮点 / 可选 16-bit 压缩，内存可控
- Isomorphism：同构的转牌河牌合并计算（花色单调面收益 3-7×）

**API 形态**（`examples/basic.rs` 已核实）

```
CardConfig { range: [oop_range, ip_range], flop, turn, river }   // 注意：数组只有 2 项 = heads-up
TreeConfig { starting_pot, effective_stack, flop/turn/river_bet_sizes: [OOP, IP], ... }
solve(&mut game, max_iterations, target_exploitability, print_progress) -> exploitability
game.equity(player) / game.expected_values(player) / game.strategy()
game.play(action) / game.back_to_root()   // 树导航
```

**风险**

- ⚠️ 2023-10 作者转为商业开发，**停止维护**。但代码已冻结成熟，对我们反而是好事（无 API 动荡）
- ⚠️ **AGPL-3.0**：见下方合规边界

**次选 `TexasSolver` 的理由**：直接下载 release 里的 `console_solver.exe`，命令行喂一个 txt 配置即可，输出 `output_result.json`。**零编译**，本周就能验证。官方 FAQ 明确写："**for personal users, the solver is completely opensourced and free**" + "If you integrate the release package (binary) into your software, **Yes**, you can do that"。

---

## 四、合规边界（即使自用也要注意）

PokerSense 是 **public 仓库**，这点决定了落地方式：

| 做法 | 是否触发 AGPL 义务 | 结论 |
|---|---|---|
| 把 solver 源码/二进制提交进 PokerSense | ❌ 触发（且 TexasSolver FAQ Q2 明令禁止分发二进制） | **禁止** |
| 放在 `.upstream/`（`.gitignore:57` 已忽略），运行时以**独立进程 / HTTP** 调用 | ✅ 不触发 | **推荐** |
| 把自己的代码与 solver 源码**静态链接** | ❌ 触发传染 | **禁止** |
| 自用、不对外提供网络服务 | ✅ AGPL §13 网络条款不触发 | **安全** |

> 沿用 GTOpen 的模式：二进制与源码都在 `.upstream/`，不进 Git。

---

## 五、建议的接入路径（按投入排序）

| 步骤 | 内容 | 产出 |
|---|---|---|
| **S1 · 验证可行性** | 下载 TexasSolver release 二进制，命令行跑一个 6-max 常见翻后局面（如 BTN vs BB 单挑底池），实测**真实耗时与 exploitability** | 拿到本机真实延迟数字，判断哪些街可以实时 |
| **S2 · 河牌先行** | 河牌最快（0.06s 级）。先做一个 `RiverStrategyProvider`，只覆盖河牌单挑 | 最快见效的翻后覆盖 |
| **S3 · 转牌 / 翻牌** | 转牌评估实时可行性；翻牌走**离线预计算成资产**（需 owner 确认许可立场） | 逐步扩街 |
| **S4 · 交叉验证** | 用 `noambrown/poker_solver`（河牌专项）或 GTOpen 反向校验我们的结果 | 提高可信度 |

**不要做**：多人翻后求解——没有现成开源可用，自研成本等同于复现 Pluribus。

---

## 六、待 owner 拍板的问题

1. **翻后走哪条路**：TexasSolver 二进制（省事、AGPL、有 Windows 版）还是 postflop-solver（Rust 库、有 bunching、需自己编译+封装）？
2. **翻牌圈是否接受"离线预计算成资产"**？这涉及用 AGPL 软件生成数据——自用立场下我认为可接受，但需要你确认。
3. **多人底池（3+ 人看到翻牌）怎么处理**？建议直接 ABSTAIN 并给出明确原因（符合 fail-closed），而不是用单挑结果近似。
