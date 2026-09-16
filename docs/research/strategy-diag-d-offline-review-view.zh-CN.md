# STRATEGY-DIAG D 阶段：现有离线复盘的最小接入

任务：STRATEGY-DIAG（Issue #23）· 检查点来源 = 评论 5693166971 + C 审查 `pullrequestreview-5219296503`
run_id：`STRATEGY-DIAG-D-20260916T0645Z`
**范围：只有 D。A/B/C 未重做、未改协议参数、未删历史。** 未改视觉规范、未开采集、未新增算法或自动化。

## 1. 页面入口与点击步骤（非程序员可直接照做）

| 步骤 | 操作 |
| --- | --- |
| 1 | 启动本机 AA 观察台（`python -m poker_engine.desktop.aa_server --profile <识别资源> --records-dir <记录目录>`），浏览器打开 `http://127.0.0.1:8771/` |
| 2 | 顶部导航点 **「设置与分析」** |
| 3 | 页面下半部展开 **「研究结果示例与已保存记录（合成 · 非本桌）」** |
| 4 | 在「研究示例」下拉里选 **对手范围敏感性对照（合成研究示例）**，点 **「载入示例（只读预览）」** |
| 5 | 查看：来源与适用范围、固定策略本、唯一变化对象、三组条件净 EV 与相对基线差、可展开的精确分数与主要终局路径 |
| 6 | 点 **「保存为独立研究记录」** → 记录编号出现在提示里，记录下拉多一条（状态「已保存 · 当前版本」） |
| 7 | 在「已保存的研究记录」下拉切到**另一条**记录，点 **「重开结果」** |
| 8 | 需要核对原始字段时展开 **「完整来源、身份与原始字段（只读）」** |

真实截图（本机实测，未提交仓库以免与「不提交媒体」规则冲突）：

| 文件 | 内容 | 大小 / sha256 前 16 |
| --- | --- | --- |
| `d-preview.png` | 示例只读预览：合成标签、来源、固定策略本、变化对象、EV 对照表 | 71,543 B / `491248be112d2c59` |
| `d-saved.png` | 保存后的独立记录：编号、状态「已保存 · 当前版本」、记录下拉「共 3 条」 | 74,578 B / `a7f1f4f2afdc1082` |
| `d-details.png` | 精确分数表（`737057752/20650813` 等）与主要终局路径贡献差（5 条非零） | 40,453 B / `80419f97e0453ae5` |

## 2. 最小接入做了什么

| 层 | 文件 | 内容 |
| --- | --- | --- |
| 后端 | `src/poker_engine/desktop/aa_study_records.py`（新） | 受控示例注册表 + 报告校验 + 视图模型 + 独立记录存储 |
| 后端 | `src/poker_engine/desktop/aa_server.py` | 5 条只读/受控路由 + 服务 `/study.js`（沿用既有同源与 `X-AA-Live` 中间件） |
| 前端 | `ui/aa-live/study.js`（新） | 仅用 `createElement`/`textContent` 的安全渲染、token 迟到响应保护、错误态 |
| 前端 | `ui/aa-live/index.html` | 在「设置与分析」里加一个 `<details>` 面板 + 一条 script |

**五条路由**：`GET /api/study/examples`、`GET /api/study/examples/{id}/view`（只读预览，不落盘）、
`POST /api/study/records`（保存）、`GET /api/study/records`（列表）、`GET /api/study/records/{record_id}`（重开）。

## 3. 合成示例与真实牌局的隔离（硬约束）

- 示例只经**固定注册表**载入：`example_id → 仓内固定文件名 + 期望 sha256`；
  报告里的 `protocol.path` / `input.path` 字符串**只做一致性比对，绝不用来打开文件**（有专门回归测试证明）。
- 保存的记录是**独立 sidecar**：`records_dir/study-records/<record_id>/record.json`，
  与「复查记录」的 issue 命名空间分离；记录里 `bound_record_id = null` 并写明不绑定任何观测牌局。
- 视图模型显式标注 `source_type = SYNTHETIC_STUDY_EXAMPLE` 与「合成研究示例 · 非本桌 · 非真实对局」，
  并给出 4 条**缺失项**（未绑定真实牌局、未观测范围、真实抽水/封顶/摊牌 UNKNOWN、未评估跨世界总体优劣），
  界面不允许用示例参数补齐真实牌局。
- `strategy_eligible` / `advice_emitted` / `live_advice` 恒为 `false`；产品流程不调用视觉 API、不开采集设备、
  不做任何牌桌操作（有 `RecordingSession` 回归：整个流程只触发应用关闭时的 `stop`）。

## 4. 数值展示与校验

- EV 以 **chips** 为单位给出易读小数（4 位，仅展示）+ **可展开的精确分数**；
  对账只用精确 Fraction，`0.1607` 这类小数不参与任何比较。
- 保存与打开时都会**重算**：每套策略概率和 = 1、贡献和 = 该策略 EV、终局路径贡献和 = 台账贡献和、
  同本对账 `Σ(逐路径差) = 报告 EV 差`、跨本对账 `Σ(逐路径差) = selected − manual_reference`、
  逐因子读数与策略指标一致、`Σ 后验 = 1`。任何一项不符即拒绝。
- **`policy_unchanged` 不接受「两个空字段相等」**：要求策略本哈希是完整 64 位十六进制、
  世界内 `policy_hash_before == policy_hash_after == book_sha256`、`unchanged_across_worlds is True`
  且与冻结策略本台账一一对应；空值或缺失直接判无效。
- 拒绝清单（都返回 400 且界面不渲染结果）：基准未复现（`matches_original_reading != true`）、
  世界 BLOCKED、未知 schema / 多余字段、路径贡献或概率被篡改、不可达路径带条件 EV、
  控制台行数/路径数与计数不自洽、空身份、非法路径、示例源文件哈希不符、未登记示例、
  外部记录编号、超大小记录文件、重复键 JSON。
- **输入修订变更后旧结果标历史**：登记文件在保存后被替换时，重开该记录返回
  `content_status = SUPERSEDED_SOURCE`（界面显示「历史记录 · 源文件已变更」），内容与标签仍按保存时原样展示，
  不再作为当前结果。
- 迟到响应：`studyToken` 单调令牌 + `studyAcceptance()`，旧响应到达时既不渲染也不改动 `#study-content`。

## 5. 实际验证（本轮实跑）

| 检查 | 命令 / 方式 | 结果 |
| --- | --- | --- |
| 后端与 API 回归 | `pytest tests/desktop/test_aa_study_records.py` | **24 passed** |
| 真实 JS 交互（无浏览器） | `pytest tests/desktop/test_aa_study_ui.py`（起真 uvicorn + 跑 node 驱动**原样 study.js**） | **2 passed / 26 项检查全绿** |
| 页面交互与布局 | agent-browser（Chromium）打开 `http://localhost:8791/` 并按步骤点击 | **BROWSER_VISUAL_RUN**：3 张真实截图（见 §1） |
| 全部 JS 检查 | `node tests/js/study_flow_dom_stub_test.mjs <base>` | `verdict ok=true, passed=26, failed=0, requests=13` |
| 全仓回归 / lint | `pytest -v` / `flake8 src tests tools` | 见 PR 正文的当轮实测行 |

JS 交互测试覆盖：示例列表与 SYNTHETIC 标记、只读预览的数值/路径/缺失项/固定说明、保存为独立记录、
记录列表、重开内容与身份一致、**迟到响应不覆盖新记录**、未知示例与外部记录编号被拒、
无 `innerHTML` 写入、全部请求只落在 `/api/study/*`。

**浏览器限制（如实记录）**：agent-browser 的页面会话在**单条命令内**稳定，跨命令会丢失页面
（表现为快照 `(empty page)`、ref 失效）；因此点击流程都放在同一条命令链里完成，
`http://127.0.0.1:8791/` 渲染空白而 `http://localhost:8791/` 正常。截图与自动化断言分开表述。

## 6. 未完成项与边界

- 列表上限 30 条、单记录/报告上限 2 MB；不做导出、不做图表、不做多示例并列（本轮明确不做美化）。
- 未接入真实牌局：示例与观测记录**没有**任何自动关联路径，也不打算有。
- 未改视觉规范（复用既有 `panel` / `footnote` / `tag` / `review-picker` / `analysis-table` 样式）。
- 未新增常驻服务、队列或通用导入平台；未改观察、人工复查与 AI 候选数据。
