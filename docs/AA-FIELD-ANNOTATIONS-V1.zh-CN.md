# AA-FIELD-ANNOTATIONS-V1 · 来源绑定的开发字段标注与不可逆曝光历史

> 状态：**P0 已实现（PR pending review）**。本模块是 #19 的**选择性迁移**结果，不是 #19 的原样合并。
> 依据：Issue #27 `issuecomment-5724976158` · `PR19_FINAL_ARCHITECTURE_DECISION_READY`（裁决 `MIGRATE_SELECTED_THEN_CLOSE`）。
> 实现：`tools/aa_annotation_records.py`；测试：`tests/tools/test_aa_annotation_records.py`。

---

## 1. 这是什么，不是什么

**是**：AA 视觉开发的**数据治理记录**——把「某来源帧的某个字段，人工标注成什么、改过几次、被训练消费过哪一版」做成可追溯的事实。

**不是**：

- 不是truth source。标注 ≠ holdout gold、confirmed poker facts、合法行动历史、PHH confirmed actions、策略资格、验收结论。
- 不是训练平台。不训练、不导出训练样本、不生成 crop、不调用模型、不发 HTTP、不做 UI。
- 不是第二套准入体系。用途资格**只读派生**，且必须绑定当前 main 既有的 reservation-first 治理（`aa_data_separation` / `aa_holdout_reservations_v1.json` / freeze exposures）。
- 不是 P1。金额/glyph 导出、UI 字段编辑区、更多字段词汇、历史标签导入均**未迁**。

---

## 2. P0 保留的三类价值

| 类别 | 内容 |
| --- | --- |
| **A. 来源绑定的逐字段开发标注** | 同一 `(source identity, field_id, seat)` 上的 KNOWN / UNKNOWN / NOT_APPLICABLE 标注 |
| **B. append-only 修订历史** | 更正只能追加新 revision，`supersedes` 指向旧值，`expected_revision` 不匹配即拒；旧 revision 永久可读 |
| **C. 不可逆 training exposure history** | 来源级使用事件单调累积；修订、改名、换 replay/目录/模型、撤回标签都**不能**把它变回 untouched holdout |

已**明确不迁**（见 §8）：逐事件 hash chain、旧 9 槽/1801 帧 UI 与 HTTP、`zone==development` 旧分区布尔、`strategy_ready`/`strategy_eligible`、训练标签导出。

---

## 3. 首批支持的字段

| field_id | 含义 | 值类型 | unit | 备注 |
| --- | --- | --- | --- | --- |
| `pot_display` | 显示的底池数额 | `int`（>=0，KNOWN 时必须给值） | 必填，现仅 `chips` | 只表示**显示底池**；`current_street_wager` / `call_price` / `cumulative_commitment` 等 amount role 一律拒绝 |
| `visible_action_glyph` | 可见的动作字样 | `str` ∈ {fold, check, call, bet, raise, all_in, muck, none} | 不允许 | 只表示**可见字样**；携带 `full_actions` 一律拒绝 |

**未知 `field_id` 一律 fail closed**：

- `seat_presence` / `current_bet` / `street_wager` / `actor` / `special_mode` → `UnsupportedFieldError`（P1 候选，需先定义可见语义与 UNKNOWN 规则）
- `full_actions` / `acceptance` / `strategy_eligible` / `canonical_verified` / `confirmed_facts` / `holdout_gold` / `phh_confirmed_actions` / `strategy_ready` → `ForbiddenTruthFieldError`（**永不开放**：标注不是这些事实的归属）

---

## 4. Source identity（长期身份）

`source_ref` 必须携带，且**拒绝** `path` / `current_path` / `replay_name` / `replay_file` / `ui_index` / `display_index` 等可变定位与 UI 行号作为身份。

| 字段 | 含义 |
| --- | --- |
| `source_audit_ref` + `source_audit_sha256` | 不可变的来源/审计/清单引用 |
| `media_id` + `media_sha256` | 媒体身份 |
| `frame_index` + `frame_sha256` | 帧身份 |
| `layout_id` / `slot_count` / `slot_mapping` | 布局与槽位映射；**只支持 8 物理槽位**，不做 9→8 猜映射 |
| `observation_snapshot_digest` | observation / replay 快照摘要 |
| `implementation_revision`（可选） | 实现 / 模型版本（对照物，不参与身份键） |
| `parent_frame_ref`（可选） | 派生 crop 的父帧身份； exposures 在父子之间双向可追踪 |

身份等级：

- `COMPLETE` — 全部就位且槽位映射是 0..7 的双射 ⇒ 可用于 `assess_use`
- `INCOMPLETE` — 核心身份在，但布局/摘要/槽位不全 ⇒ **只能 `ANNOTATION_ONLY`**，训练与 gold 桥关闭
- `UNBOUND` — 核心身份缺失 ⇒ 连稳定身份键都算不出来，不可记录 exposure

稳定身份键 = SHA-256(`source_audit_ref | source_audit_sha256 | media_id | media_sha256 | frame_index | frame_sha256`)。
**换模型版本不改变身份键**，因此修订/重建模型不会洗掉历史曝光。

---

## 5. 标注记录 schema（append-only JSONL）

每条 revision 含：`schema_version` / `event_id` / `revision_id` / `recorded_at` / `actor` /
`source_key` / `parent_source_key` / `identity_class` / `identity_gaps` / `source_ref` /
`field_id` / `seat` / `status` / `value` / `unit` / `reason` / `supersedes` / `expected_revision` /
`model_output` / `model_reason` / `model_output_seen` / `implementation_revision` / `observation_snapshot_digest`。

**UNKNOWN 是一等状态**：

- `status=UNKNOWN` ⇒ `value` 必须为 `None` 且必须有 `reason`；**绝不**折叠成 0 / false / none
- `status=KNOWN` 且 `value=0` 是合法标注，与 UNKNOWN 严格区分
- `status=KNOWN` 且 `value="none"`（可见字样为无）与 UNKNOWN 严格区分
- `status=NOT_APPLICABLE` ⇒ 必须有 `reason`（上下文依据），且不得携带值

`model_output` / `model_reason` / `model_output_seen`（YES / NO / UNKNOWN）是**对照材料**：
绑定当时的 observation 快照与实现版本，不随新模型或新 replay 回填；human value 不因模型输出而改变。
看过模型预测的标注保留 `model_output_seen=YES`，未被策略显式允许（`allow_model_assisted_labels`）时不能进 Independent 用途。

**当前值只按 `supersedes` / revision 折叠，不按时间戳猜赢家**；同组出现多个 head 记为 `conflicts` 并 fail closed。

---

## 6. Exposure invariant（长期不变量）

> 一旦某来源帧或其派生样本进入 **training / tuning / template_selection**，任何后续修订、撤回标签、replay 重建、改名换目录、crop 重建、模型变化、新建 store，**都不能把这次使用从历史中抹掉，也不能据此恢复成 untouched holdout**。

- exposure event 记：`purpose` / `evidence_kind` / `annotation_revision_ids` / `source_key` / `parent_source_key` / `run_or_manifest_ref` / `artifact_ref` / `role` / `policy_ref` / `policy_violation`
- `evidence_kind` 四态严格区分：`RESERVED`（预留意图）/ `RELEASED`（释放意图）/ `POSSIBLY_USED` / `ACTUALLY_CONSUMED`
- **状态单调**：`RESERVED → RELEASED` 仍是 `EXPOSED_POSSIBLY_USED`（释放不能清除保守污染）；出现过 `ACTUALLY_CONSUMED` 则永久 `EXPOSED_ACTUALLY_CONSUMED`
- **没有记录 ≠ never_trained**：历史覆盖未声明/不可读时状态为 `EXPOSURE_UNKNOWN` 并 **BLOCKED**；只有在**全部已知 store 都被声明并成功读取**且零事件时，才是 `NO_EXPOSURE_RECORDED_UNDER_DECLARED_COVERAGE`（注意名字里没有 "clean"）
- **违规事实必须记录**：`policy_violation=True` 的事件照常落盘并保留，"本不该训练" 不能成为删除历史的理由；它使后续独立验收失败
- 训练过的**开发数据**依策略可以再次训练（`allow_reuse_of_exposed_development`）；永久禁止的是把它**伪装成独立 holdout**
- 记录持久化失败 ⇒ 消费者**拿不到** ALLOWED（`reserve=True` 时回落到 `consumption_record_not_durable` + BLOCKED）

---

## 7. 最小 API

| 函数 | 职责 |
| --- | --- |
| `append_annotation(store, source_ref, field_id, seat, label, expected_revision, actor, ...)` | 追加一个明确修订，返回 `revision_id`；不写原 observation |
| `get_annotations(store, source_ref, include_history=False)` | 只读当前标注与历史；按 `supersedes` 折叠 |
| `assess_use(store, source_ref, purpose, policy_bundle, *, reserve=False)` | 只读派生用途结论；客户端不能提交 `training_eligible` |
| `record_exposure(store, source_ref, annotation_revision_ids, purpose, run_or_manifest_ref, evidence_kind, ...)` | 追加使用/释放/消费/违规事实 |
| `exposure_projection(store, scope)` | 供现有 freeze.exposures 绑定的只读投影 |

### `assess_use` fail-closed 清单

以下任一命中即为 `BLOCKED`（`usable_scope = ANNOTATION_ONLY`）：

`policy_bundle_missing` · `policy_ref_missing` · `purpose_unsupported` · `source_identity_incomplete:*` ·
`reservations_missing` · `frame_reserved_or_invalid` · `role_unknown` · `role_conflicts_reservation` ·
`exploration_not_development` · `required_annotation_missing:*` · `required_annotation_not_known:*` ·
`annotation_history_conflict` · `model_assisted_label_without_policy` ·
`exposure_history_coverage_undeclared` · `declared_store_missing:*` · `declared_store_unreadable:*` ·
`own_store_not_declared_in_history_coverage` · `exposure_history_unknown` ·
`possibly_used_reuse_not_declared` · `exposed_development_reuse_not_declared` ·
`known_policy_violation_exposure` · `consumption_record_not_durable`

**「不在 reserved」绝不等于「自动可训练」**：仅调用 `assert_training_frames` 通过，而缺少 policy 绑定、角色、完整历史覆盖时仍然 BLOCKED。

---

## 8. 与现有体系的关系（桥接方向）

| 现有对象 | 本 P0 的关系 |
| --- | --- |
| `aa_data_separation` / `aa_holdout_reservations_v1.json` | **复用**：`assess_use` 调用 `assert_training_frames` 做 reserved 排除，并在此基础上叠加 reserved-first 的完整性要求。**未放宽**任何旧门禁 |
| `freeze.exposures` | **只读投影**：`exposure_projection()` 输出带 `used_for` / `role` / `artifact_sha256` / `history_coverage` / `freeze_bindable` 的行，供冻结清单绑定。**未修改** `aa8_holdout_plan.validate_freeze` |
| AAReviewDesk | **无桥**。截图 verdict/note 不解析成字段值，不回写 observation |
| `aa8_gold_registry_v2` | **无导入桥**。gold 仍走既有独立审阅流程 |
| `aa_analysis_records` / `aa_study_records` / `aa_hand_input` | **无桥**。分析、合成研究、手工假设均不自动变成标注 |
| `aa_semantics` / PHH / 策略路由 | 无任何写入路径 |
| UI / HTTP / export | **未迁**（P1）。未改动 `ui/`、`tools/aa_replay_viewer.py`、任何 HTTP endpoint |

---

## 9. 明确未迁（P0 边界）

- **逐事件 hash chain**（`previous_sha256` + `entry_sha256`）：**未迁**。保留 source / frame / artifact / snapshot 摘要作为身份锚点，但没有任何链式依赖；不引入签名、Merkle、数据库、事件总线或外部锚。
- **UI / HTTP 标注入口**：未迁。
- **训练标签导出**（money/glyph）：未迁。
- **旧真实台账自动导入**：未做，也未检查历史标签文件是否存在。
- **旧 9 槽 / 1801 帧固定 replay**：未迁，`slot_count != 8` 直接判定身份不完整。

---

## 10. 持久化与完整性

单命名空间 `annotations.jsonl` + `exposures.jsonl`，append-only，写入 `fsync`，附简易独占文件锁。
读取时**不静默修复**：空白行、坏 JSON、未知 `schema_version`、错误 `event_type`、重复 `event_id` 一律 `StoreIntegrityError`，并在 `assess_use` / `exposure_projection` 中回落为 UNKNOWN / BLOCKED。

---

## 11. 状态边界

`REAL_HAND_ACCEPTANCE_PENDING` 与 `NOT_ASSESSED` **继续保留**。
本模块不宣称：真实牌局验收通过、PHH 可导出真实手牌、策略/盈利已验证、GTO 已完成、holdout 已合格、既有训练入口已全部接入拦截器。
P0 只证明：**这个新入口的标注 / 修订 / 曝光契约成立**。
