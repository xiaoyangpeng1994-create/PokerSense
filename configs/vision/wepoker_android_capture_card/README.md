# wepoker_android_capture_card — 未标定（UNCALIBRATED）

`platform_id = wepoker_android_capture_card`（采集卡：手机 → 视频适配器 → UVC 采集卡 → PC）。

## 状态：NOT CALIBRATED

本目录目前只有一份**状态清单** `calibration.json`（`status: "uncalibrated"`）。它
**不含任何**字段测量（ROI、阈值、样本数、置信度），也**不得**继承
`wepoker_android`（LDPlayer）或 `wepoker`（H5）的任何结论。

## 已就绪（代码层）

- `src/poker_engine/perceptual/capture/capture_card_backend.py`：UVC 实时采集后端
  （`CaptureCardBackend`），MSMF / DirectShow / YUY2，断流黑屏检测。
- `src/poker_engine/perceptual/capture/normalization.py`：阶段 C 的画面归一化
  （旋转 → 镜像 → 裁剪 → 尺寸验证）。
- `tools/capture_card_calibration/`：硬件无关的标定工具链，外加阶段 A/B 的
  **真机录制工具**（`probe` 探设备、`record` 录会话）。运行方式：
  `python -m tools.capture_card_calibration.cli --help`。

## 已就绪（几何层，2026-09-03 更新）

- `hero_slot_layout.json`：hero 2 槽，实测。
- `board_slot_layout.json`：board 5 槽，实测（83 个 RIVER 帧、跨两个 session
  中位数稳定 ±1px）；配套平台配置的 `board_cards` ROI 已拓宽到完整 5 牌条带。
- ⚠️ 识别标定仍未完成：H5（wepoker）角标模板在采集卡分辨率下**未通过**
  花色验证（黑桃/梅花混淆且 raw score 高，无法用阈值门控），故
  `template_source: wepoker` 不得用于花色。`calibration.json` 保持
  `uncalibrated`，card 字段必须继续读出 UNKNOWN。

## 用录制工具做阶段 A/B

```bash
# 1. 建数据集骨架（含 device_and_capture.json 模板）
python -m tools.capture_card_calibration.cli init --root capture_card_calibration_YYYYMMDD

# 2. 填好 source/device_and_capture.json 里的手机/适配器/采集卡字段（消除 REPLACE_ME）

# 3. 探测采集卡实际参数（回填 uvc 字段）
python -m tools.capture_card_calibration.cli probe --device 0 --api MSMF

# 4. 录制一个会话（Ctrl+C 结束；断流/黑屏事件自动记进日志）
python -m tools.capture_card_calibration.cli record --root capture_card_calibration_YYYYMMDD --session session_001 --update-manifest
```

`record` 会写到 `source/raw/session_001.mkv`（**未归一化原画**，归一化留到阶段 C 离线做），
并在 `source/probe/` 输出逐会话的断流/黑屏/重连事件日志。

## 待完成（需真机采集，见 `docs/capture-card-calibration-guide.zh-CN.md` 阶段 A–L）

1. **阶段 A**：冻结硬件与采集环境，写 `device_and_capture.json`。
2. **阶段 B**：录制 45–90 分钟真实对局素材。
3. **阶段 C**：确定 `normalization.json`（旋转/裁剪/输出尺寸）。
4. **阶段 D–G**：抽帧、ROI 测量、逐帧真值标签、最低覆盖。
5. **阶段 H–I**：数据划分、模板与阈值标定。
6. **阶段 J–K**：座位映射、Replay 与性能证据。
7. **阶段 L**：把通过验收的 `configs/platform/wepoker_android_capture_card__<layout_id>.json`
   、`_seat_mapping.json` 以及 `board/dealer/empty/hero_slot_layout.json`、
   隐私安全模板落到本仓库，并接入 `live.py` 管线。

在阶段 L 完成前，任何基于本平台的识别都**必须**读出 `UNKNOWN`，不得宣称端到端可用。
