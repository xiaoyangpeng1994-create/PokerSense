# 五个关键感知字段的有限证据验证

`tools/aa_critical_perception_score.py` 比较冻结机器候选和单独的来源审核记录，
作用域固定为 `aa-critical-perception-five-fields-v1`。它不解码图片、录像或打开设备，
不训练模型，不调用 solver，不写真值或确认收据。文件哈希检查读取声明文件的字节；
不会按 sources 行中的帧标识去打开或核验源图像。

结果 `CRITICAL_PERCEPTION_FINITE_PASS` 只表示这批登记行在下面的声明范围内通过；
`full_visual_acceptance`、`strategy_eligible`、`advice_emitted`、
`real_hand_confirmed`、`independence_verified` 始终为 false。它不能代替原有
10/13 字段视觉验收、TARGET-S 真手确认或独立盲测签收。

## 输入与最小契约

CLI 只读输入，向标准输出写 JSON，通过返回 0，PARTIAL/BLOCKED 返回 2：

```powershell
$env:PYTHONPATH = 'src'
$env:PYTHONUTF8 = '1'
python -m tools.aa_critical_perception_score --manifest <local-manifest.json>
```

所有 JSON 对象拒绝重复键。契约对象拒绝未知键；版本号必须是整数 1，不能是布尔值。
引用格式统一为 `{path, sha256}`，路径相对于 manifest，也可使用绝对路径。
SHA256 必须是 64 位小写十六进制。所有被引用文件实际核对字节哈希。

| 文件 | 必须字段 |
| --- | --- |
| manifest | `schema_version`, `scope`, `cohort`, `declared_at_utc`, `freeze`, `sources`, `predictions`, `reviews` |
| sources | `schema_version`, `cohort`, `exposure`, `registered_at_utc`, `rows` |
| predictions | `schema_version`, `started_at_utc`, `source_manifest_sha256`, `producer_sha256`, `rows` |
| reviews | `schema_version`, `started_at_utc`, `source_manifest_sha256`, `reviewer_id`, `method`, `rows` |

`freeze` 是 `{frozen_at_utc, files: [{path, sha256}, ...]}`。它至少冻结评分器、
`aa_critical_perception.py`、`aa_reader.py`、新旧 `aa_live_context`、
`aa8_critical_fields_v3.py`、`aa8_state_adapter_v2.py`、`aa8_cards_v2.py` 和 sources。
具体必需路径由评分器的 `REQUIRED_IMPLEMENTATION_PATHS` 列出。真实运行的模型、
参数、上游实现和训练来源还应加入该清单；评分器不自动推断完整训练依赖闭包。
predictions/reviews 是冻结后的输出，不能伪装成冻结前输入。

声明时序必须满足 `declared <= registered <= frozen < prediction_started`，且
`frozen < review_started`，均为非未来 UTC 时间。时间先后只是输入声明的一致性检查，
不能认证真实操作发生时间。

`cohort/exposure` 只能是 `development/development_exposed`、
`verification/unexposed_declared` 或 `synthetic/synthetic`。改变一个标志不产生
独立性；即使声明 verification，通过结果也不会返回“已独立验证”。

sources 每行格式是 `{binding, required_fields}`。`binding` 精确包含：

```text
source_id: 非空来源 ID
epoch: 非空手牌 epoch
frame: 非负整数处理帧号
source_frame: 非负整数源帧号
pts_seconds: 非负有限数字
source_sha256: 来源帧声明哈希
```

同一来源的处理帧号和源帧号必须分别连续递增，PTS 严格递增；帧不得重复。
`required_fields` 按下面固定字段顺序列出该 checkpoint 必需自动正确的字段，可为空
以保存 preroll/转换/遮挡等上下文。这个选择随 sources 在看预测和审核输出前冻结；
不能事后删除失败行或把失败项移成 context。原视频是否完整、区间之外是否被选择性
删除、图片是否真与声明哈希对应，仍需外部来源审计。本评分器只核对登记的元数据。

predictions.rows 保存完整原始 runtime row，包括 `critical_perception_v1`。
评分器先调用 `checked_view`，再用 `CriticalPerceptionBoundary` 按顺序重放每行，
要求缓存候选与重新推导结果精确一致。只补一个成功标志、手改候选或造 temporal
window 都不能过。所有 window witness 必须属于登记来源且绑定完全一致。
`producer_sha256` 必须匹配冻结的候选模块。显式人工修正标记会被拒绝；完整伪造
原始输入的真实性仍不是哈希或结构检查能证明的事。

reviews.rows 每行是 `{binding, fields}`，来源与 predictions/sources 一一对应。
`method` 只接受 `source_only`；审核者 ID 必填，不接受 `independent=true` 或
`predictions_used_for_labels` 等自封权限/泄漏元数据。源图审核和预测隔离必须由实际
工作流程保证；`source_only` 字符串本身不是隔离证明。

reviews.fields 必须恰好包含全部五字段，每项是 `{status, value, reason}`。
`status` 为 `KNOWN`、`UNREADABLE`、`UNREVIEWED`；后两者 value 必须 null。
reason 必须非空。机器的 `CONFLICT` 按拒识处理，不可计为正确。

| 字段顺序 | KNOWN 值 | 必需正向覆盖格 |
| --- | --- | --- |
| `board` | 五张无重复、顺序一致的具体牌，如 rank+suit | 至少一个 required 正确实例 |
| `hero_participation` | `ACTIVE` / `FOLDED` / `ALL_IN` | required ACTIVE、FOLDED 各至少一个 |
| `participation` | 固定八座 `0`…`7` 的 ACTIVE/FOLDED/ALL_IN/EMPTY/WAITING 字典 | required 正确向量覆盖 ACTIVE 和 FOLDED |
| `all_in_seats` | 有序、唯一、0…7 整数列表 | required 空列表及非空列表各至少一个 |
| `river_first_actor` | 0…7 整数座位 | required Hero=4 首位及其他座位首位各至少一个；另需一个 source review 为 UNREADABLE 的安全 UNKNOWN |

Hero 固定为 AA8 seat 4；gold 的 Hero 状态和 ALL-IN 列表必须与它自己的完整参与
向量一致。first actor 是河牌首位行动者，不能用当前 actor 代填。

## 评分与停止条件

每个字段报告 `total`、`required_total`、`correct_automatic`、`correct_required`、
`wrong_known`、`unknown_required`、`guarded_unknown`、`guarded_unreadable`、
`unreadable`、`unreviewed`、`conflict`，并保留逐行比较。所有自动 KNOWN 都要核对，
即使在非 required 上下文中；
审核不可读或未审也不能让一个 KNOWN 绕过核对。

通过要求所有 required 项自动正确，所有正向覆盖格均有匹配实例，所有 KNOWN
错误/未获审核支持数量为零，所有未审字段数量为零。required 项缺审核、不可读或
机器 UNKNOWN/CONFLICT 都是 PARTIAL。全 UNKNOWN、全 context、只有 ALL-IN 负例
都无法通过。非 required 且已审核的 UNKNOWN 可计 `guarded_unknown`，不计正确覆盖。
河牌首位 actor 还必须有非 required 的自动 UNKNOWN，并且 source-only review 明确为
UNREADABLE，才贡献 `guarded_unreadable`；KNOWN 上下文、UNREVIEWED 或 CONFLICT
不能满足这格。缺 Hero-first、other-first 或这格拒识证据均为 PARTIAL。
人工把 gold 改成正确值不能把原机器错误改记为自动成功。

输入缺失、重复、额外行、重排、来源/版本/字节哈希不符、冻结漂移、伪造候选或
不合契约的标志返回 BLOCKED；正确性、可用性或覆盖不足返回 PARTIAL。停止本批晋级，
保留失败记录。修改实现/模型/阈值后重新冻结；已经看过结果的材料永久保留暴露事实。

输出 `checked_hashes` / `checked_frozen_artifacts` 是实际检查；`declarations` 是
时间线、exposure、reviewer/method 声明。有限测试不产生总体准确率或盲测独立性结论。
`source_media_verified=false`、`media_decoded=false` 明示像素真实性未在本工具验证。
专项通过也不自动确认任何真实手，更不能绕过 TARGET-S 的显式证据和确认门。

## 当前已暴露材料与后续真实验证

PHASE8E 28 手始终是已暴露 development，全部保留原 CLASS-X。它们的旧 collage
每手只有 10 帧，不足以重建连续 turn→river 首位 actor 或完整当前 pipeline；其中
5 条记录缺河牌 anchor 元数据，不能自动纳入“完整验证”登记。

本轮现有 28 手单个当前帧诊断使用原冻结 card heads，没有时序重放：native 与
gaussian 各匹配已暴露视觉 board 26/28，gaussian 接受了一个 native 拒绝的错误
牌身份。这只是有已暴露人工结论的开发诊断，不是独立 board 准确率。新增
AA8CardReaderV2 要求 native/current 一致，否则拒识；没有训练或更改阈值。
实际逐手开发诊断保存在私有证据报告，不生成本评分器的假验证输入。

阶段出口仍需另行授权的新验证集、可信来源登记和真实逐帧审核，以及至少一手满足
TARGET-S 全部事实/确认要求的真实闭环。工程夹具、28 手开发回归或本评分器的有限
PASS 都不能替代它；缺新材料或缺合格真手时保留真实验收未完成。
