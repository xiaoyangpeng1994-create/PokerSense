# Capture Replay 合同与发布证据

`poker_engine.replay` 用于注册和重放经授权、脱敏的真实牌桌采集数据。它解决的是
Capture/Recognition → Stable Observation → State/Event 的可复验性，不等同于从事件流重建
整手牌的 Event Sourcing Replay。

## 两种 Replay stage

| stage | 输入 | 用途 | 可作为 R6 发布证据 |
|---|---|---|---|
| `stable_observation` | 内嵌、已序列化的 `RawObservation` | 合同、边界和 Synthetic 回归 | 否 |
| `raw_frame` | 哈希固定的原始帧文件；执行时调用真实 recognizer | 真实 Capture/Recognition/State 回归 | 满足全部证据门后可以 |

`stable_observation` 即使标记为 `real_capture` 也会得到
`replay_does_not_start_from_raw_frame`，不能提升为发布证据。`raw_frame` 不允许内嵌预先计算的
Observation，防止绕过当前 recognizer。

## v1 artifact 必填内容

- `replay_id`、`evidence_kind`、`replay_stage`。
- `platform`：`platform_id`、`layout_id` 和平台配置文件 path/SHA-256。
- `source`：授权、隐私审查、采集时间和 recognizer/source revision。
- `calibrations`：每个识别字段的文件 path/SHA-256/sample count；文件内部必须声明同一
  platform/layout、字段名和相同样本数。
- `seat_mapping`：版本化 stack/action/actor/dealer visual slot→canonical seat 映射。
- `initial_state`：Core schema v1 的 `PokerState`。
- `frames[]`：严格递增 frame sequence、非递减 aware timestamp、原始帧引用或内嵌
  Observation，以及逐帧期望 status/state version/event types/reasons。

artifact 自身也必须由调用方提供预期 SHA-256。所有引用路径必须相对显式 `asset_root`，解析后
不得越界。v1 对未知字段 fail closed；扩展字段必须升级 schema，而不是被旧实现静默忽略。

## 执行和证据门

```python
from poker_engine.replay import load_capture_replay, run_capture_replay

replay = load_capture_replay(
    "replays/wepoker-hand-001.json",
    expected_sha256="<64 hex>",
    asset_root=".",
)
report = run_capture_replay(
    replay,
    raw_frame_recognizer=production_recognizer,
)
quality_report = report.to_dict()
```

`production_recognizer(path, frame_seq, timestamp)` 必须返回身份一致的 `RawObservation`。每个
`VALID` 且非空字段的 evidence 必须包含当前原始帧 `frame_sha256` 和 artifact 声明的
`recognizer_revision`。执行器逐帧比较：

- `EXACT / NO_ACTION / AMBIGUOUS / INVALID`；
- canonical `state_version`；
- `StateEvent` 类型；
- fail-closed reasons。

质量报告包含状态计数、字段质量计数、全部 mismatch、完整性原因和 `release_eligible`。只有以下
条件同时满足时才可为 `true`：

1. artifact/平台配置/校准/每帧文件的 SHA-256 全部匹配；
2. `evidence_kind=real_capture` 且 `replay_stage=raw_frame`；
3. 授权和隐私审查均明确为 `true`；
4. recognizer 已实际执行，输出身份、帧 hash 和 revision 全部匹配；
5. 每个实际使用字段都有非空、平台匹配的校准证据；
6. 所有逐帧预期均匹配。

合同和故障测试见 `tests/strategy/test_capture_replay.py`。仓库当前没有提交真实 WePoker
stack/action/actor/dealer 原始帧，因此该模块完成的是可验证接入路径，不代表 R6 已通过。
