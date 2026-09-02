# 手机采集卡输入完整标定执行规范

> 本文是一份可直接交给 AI 编程代理执行的任务书。目标是把“安卓手机 → 视频输出适配器 → USB 采集卡 → 电脑”形成的微扑克牌桌画面，制作成可复现、可审计、可验收的 PokerSense 独立识别标定。

交给执行 AI 时请直接要求：**先阅读项目根目录 `AGENTS.md` 和本文，然后从阶段 A 依次执行到阶段 L；实际检查素材、编写必要工具、生成配置和测试，不要只复述方案。每完成一个阶段都保存产物与哈希，缺少真值时列出精确补录清单并继续完成不受影响的部分。**

## 1. 任务目标与边界

完成后应得到一套新的采集卡平台 Profile，包括：

- 固定且可复现的采集参数和画面归一化规则；
- Hero 手牌、公共牌、街道、总底池的 ROI 与置信度标定；
- 8 个视觉座位的占用、筹码、Dealer、已完成动作 ROI 与标定；
- Hero 当前行动回合的识别标定；
- 视觉座位到规范座位/位置的映射；
- 连续帧、动画、遮挡、菜单、断流和重连的负样本；
- 按手牌和采集会话隔离的数据划分；
- 阈值证据、锁定验证集结果、文件哈希和 Replay 草案；
- 一份明确列出“通过、未通过、尚未实现”的验收报告。

本任务只负责采集卡输入的标定数据、配置和验证。它**不等于** PokerSense 已经实现 UVC/采集卡实时捕获后端，也不允许在没有验证的情况下宣称该输入方式已发布。

## 2. 强制规则

执行代理必须遵守以下规则：

1. 采集卡是新平台，建议使用 `platform_id = wepoker_android_capture_card`。不得复用 LDPlayer 或 H5 的 ROI、坐标、阈值、样本计数或置信度结论。
2. 可以复用平台无关的识别算法；牌面素材完全一致且独立验证通过后，可以复用牌面模板。几何和阈值证据仍必须重新生成。
3. 正样本必须来自未来实际运行时使用的采集卡视频流，不得用手机截图、微信压缩图或雷电模拟器截图代替。
4. 原始视频、完整画面、昵称和头像属于私有标定材料，不进入 Git、不进入安装包、不公开上传。Git 最多保留少量经审核和脱敏的紧裁剪回归样本。
5. 不可见、被遮挡、动画中或无法确认的字段标记为 `UNKNOWN`；证据矛盾标记为 `CONFLICT`。不得根据扑克常识补猜。
6. 阈值和模板只能使用训练/标定集调整；锁定验证集只能在方案冻结后运行。
7. 同一手牌及相邻连续帧必须属于同一数据划分，避免相邻帧泄漏造成虚假高分。
8. 所有进入结果的数据、配置、模板和报告都必须记录 SHA-256。
9. 任何真实字段进入生产状态前，仍需连续两帧完全一致的确认；单帧结果不得绕过现有安全门。
10. 如果画面方向、镜像、裁剪、分辨率、缩放或内容边界变化，必须新建 `layout_id` 或停止标定，不得动态移动 ROI 迁就输入。

## 3. 推荐交付目录

```text
capture_card_calibration_YYYYMMDD/
├── README.md
├── source/
│   ├── device_and_capture.json
│   ├── probe/session_001.ffprobe.json
│   └── raw/session_001.mkv
├── normalization/normalization.json
├── normalized/
│   ├── frames/
│   └── manifest.json
├── labels/
│   ├── frames.jsonl
│   └── roi_measurements.csv
├── geometry/
│   ├── table_map.draft.json
│   ├── board_slot_layout.draft.json
│   ├── dealer_slot_layout.draft.json
│   ├── empty_slot_layout.draft.json
│   └── seat_mapping.draft.json
├── templates/{cards,stacks,actions,dealer,occupancy,actor}/
├── splits/{train,calibration,validation,negative,temporal}.txt
├── evidence/
│   ├── field_metrics.json
│   ├── review_report.md
│   └── contact_sheets/
├── replay/replay.draft.json
└── SHA256SUMS
```

原始视频可以很大，交付时允许单独放在私有存储中，但清单和 Replay 必须通过哈希准确引用它。

## 4. 阶段 A：冻结硬件与采集环境

首先创建 `source/device_and_capture.json`，至少记录：

- 手机型号、Android 版本、屏幕物理分辨率；
- 系统显示大小、字体大小、DPI、深色/浅色模式；
- 微扑克版本、语言、牌桌主题、屏幕方向；
- 视频输出适配器型号、输出分辨率和刷新率；
- 采集卡型号、固件、USB 连接方式；
- 主机操作系统、采集软件及版本；
- UVC 输入分辨率、帧率、像素格式、色彩空间和色彩范围；
- 录制容器、编码器、码率、关键帧间隔；
- 是否旋转、镜像、裁剪、缩放，以及所有滤镜；
- 每次采集的日期、会话 ID 和操作者。

示例：

```json
{
  "schema_version": 1,
  "phone": {
    "model": "REPLACE_ME",
    "android_version": "REPLACE_ME",
    "display_size": [1080, 2400],
    "font_scale": 1.0,
    "display_scale": 1.0
  },
  "app": {
    "name": "WePoker",
    "version": "REPLACE_ME",
    "language": "zh-CN",
    "orientation": "portrait"
  },
  "video_adapter": {"model": "REPLACE_ME"},
  "capture_card": {"model": "REPLACE_ME", "connection": "USB 3.x"},
  "uvc": {
    "frame_size": [1920, 1080],
    "fps": 30,
    "pixel_format": "REPLACE_ME",
    "color_space": "REPLACE_ME",
    "color_range": "REPLACE_ME"
  },
  "recording": {
    "container": "mkv",
    "codec": "REPLACE_ME",
    "bitrate": "REPLACE_ME",
    "filters": []
  }
}
```

保存原始媒体探测结果：

```bash
ffprobe -v error -show_streams -show_format -of json \
  source/raw/session_001.mkv \
  > source/probe/session_001.ffprobe.json
```

### 采集设置要求

- 使用采集卡可持续稳定输出的最高原生分辨率，至少 30 fps；
- 优先 MKV 和高码率 H.264/H.265，条件允许时使用无损或帧内编码；
- 禁用 OBS 画布缩放、锐化、美颜、降噪、色键、动态裁剪和叠加层；
- 不要把桌面、鼠标或采集软件 UI 录进视频；
- 不要通过微信传输原视频；使用网盘、局域网或移动硬盘，并核对哈希；
- 手机和采集端关闭自动旋转、自动缩放和可能改变画面的通知浮窗。

## 5. 阶段 B：录制完整素材

至少录制 3 个互相独立的会话，总时长建议 45–90 分钟。会话之间至少执行一次断开和重连采集卡，以验证内容边界是否稳定。

必须覆盖：

- 2 人、3–5 人、6–8 人桌面或座位占用状态；
- Preflop、Flop、Turn、River；
- 发牌、翻牌、弃牌、过牌、跟注、下注、加注、All-in；
- Hero 有牌、弃牌、摊牌、结算和下一手发牌；
- 每个视觉座位出现筹码、Dealer 和动作文字；
- 玩家加入、离开、空座、暂离、坐下或占位变化；
- 菜单、弹窗、聊天、表情、牌局结束、战绩或其他遮挡；
- 画面切后台/回前台、采集卡断流/重连、黑屏或花屏；
- 动作发生前、动画中、动作稳定后的连续过程。

如某类状态实战中很难出现，不要伪造标签。把它列入缺口报告，后续定向补录。

## 6. 阶段 C：画面归一化

识别输入必须是固定的“游戏画布”，不能直接把不确定的 UVC 外框当作牌桌坐标。创建 `normalization/normalization.json`：

```json
{
  "schema_version": 1,
  "source_size": [1920, 1080],
  "rotate_degrees": 90,
  "mirror_horizontal": false,
  "crop_after_rotation": [0, 0, 1080, 1920],
  "output_size": [1080, 1920],
  "color_transform": "none",
  "version": "capture-card-normalization-v1"
}
```

处理顺序固定为：

```text
解码 → 旋转 → 镜像 → 裁剪 → 尺寸和内容边界验证 → PNG
```

要求：

- 优先只旋转和裁剪，不缩放；
- 如硬件必然产生固定缩放，只允许一种确定性插值和一个固定输出尺寸，并把它写入版本；
- 从每个会话的开头、中间、结尾和重连后各抽帧，测量游戏内容边界；
- 同一 `layout_id` 下边界漂移不得超过 2 个输出像素；超过则拆分 layout 或判定采集链路不合格；
- 归一化输出必须保留未来实时后端实际会看到的像素，不得为了标注而二次增强。

建议 `layout_id`：

```text
phone_<手机型号>__card_<采集卡型号>__uvc_<宽>x<高>_<fps>__canvas_<宽>x<高>__v1
```

名称需要转换为小写 ASCII 和下划线。任何影响像素坐标的设置变化都必须产生新的 `layout_id`。

## 7. 阶段 D：抽帧、去重与命名

稳定状态按 500–1000ms 抽取一帧，再做感知去重。去重只用于减少几乎完全相同的稳定帧，不能删掉关键时序。每个关键事件保留至少三帧：

```text
事件前稳定帧 → 动画/变化帧 → 事件后稳定帧
```

发牌、街道变化、动作、结算、下一手、玩家加入/离开、断流/重连均需保留时序帧。

文件名：

```text
<session_id>__t_<毫秒>__f_<源帧号>__<sha256前12位>.png
```

`normalized/manifest.json` 中每帧至少记录：文件名和 SHA-256、原视频 ID、时间戳、源帧号、normalization 版本、`stable`、场景类别、hand/group ID 和抽帧原因。

## 8. 阶段 E：座位编号与 ROI 测量

视觉座位编号固定如下，不随 Dealer 或人数变化：

```text
                         slot 4（顶部）
             slot 3                         slot 5

       slot 2                                     slot 6

             slot 1                         slot 7
                         slot 0（Hero）
```

所有 ROI 先在归一化画布上记录整数像素半开区间 `[x0, y0, x1, y1)`，再转换为归一化坐标：

```text
x      = x0 / canvas_width
y      = y0 / canvas_height
width  = (x1 - x0) / canvas_width
height = (y1 - y0) / canvas_height
```

`labels/roi_measurements.csv` 使用：

```csv
field,slot_id,x0,y0,x1,y1,source_frame,notes
```

必须独立测量：

- `hero_cards`：Hero 两张底牌整体区域；
- `board_cards`：五张公共牌整体区域，并另测五个卡槽；
- `pot`：总底池数字区域；
- `stack[0..7]`：每个视觉座位的剩余筹码；
- `action[0..7]`：每个视觉座位已完成并显示在座位旁的动作文字；
- `dealer_search[0..7]`：每个座位附近 Dealer 标志搜索窗；
- `empty_slot[0..7]`：空座“+”或等价视觉证据；
- `hero_actor`：Hero 当前可操作按钮/行动回合证据。

注意：

- `action` 是已完成动作文字，不是 Hero 底部可点击按钮；
- Dealer 输出首先是视觉 slot，之后才通过映射推导位置；
- 座位占用必须有独立证据，不能仅凭筹码 OCR 成功与否判断；
- 对手倒计时可以采集为未来字段，但当前不得冒充已经支持的 actor 识别；
- 每个 ROI 至少用 5 个不同时间、不同人数/街道的稳定帧复核，ROI 不能随帧移动。

TableMap 草案结构：

```json
{
  "schema_version": 1,
  "platform_id": "wepoker_android_capture_card",
  "layout_id": "REPLACE_ME",
  "reference_size": [1080, 1920],
  "aspect_tolerance": 0.01,
  "rois": [
    {"kind": "hero_cards", "slot_id": null, "x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0},
    {"kind": "board_cards", "slot_id": null, "x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0},
    {"kind": "pot", "slot_id": null, "x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0},
    {"kind": "stack", "slot_id": 0, "x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0},
    {"kind": "action", "slot_id": 0, "x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
  ]
}
```

八个 slot 必须全部列出。公共牌卡槽、Dealer、空座证据使用独立布局文件，不要强塞进一个 ROI。

## 9. 阶段 F：逐帧人工真值标签

`labels/frames.jsonl` 每行一个 JSON 对象。标签只能根据该帧实际可见像素填写。

```json
{
  "schema_version": 1,
  "frame": "session_001__t_001234__f_000037__abcdef123456.png",
  "sha256": "REPLACE_FULL_SHA256",
  "session_id": "session_001",
  "hand_id": "session_001_hand_0001",
  "timestamp_ms": 1234,
  "stable": true,
  "scene": "table",
  "hero_cards": {"status": "VALID", "value": ["TS", "JD"]},
  "board_cards": {"status": "VALID", "value": ["4H", "KC", "TS"]},
  "street": {"status": "VALID", "value": "FLOP"},
  "pot": {"status": "VALID", "value": 176},
  "slots": [
    {
      "slot_id": 0,
      "occupancy": {"status": "VALID", "value": "OCCUPIED"},
      "stack": {"status": "VALID", "value": 254},
      "dealer": {"status": "VALID", "value": false},
      "completed_action": {"status": "UNKNOWN", "value": null},
      "current_actor": {"status": "VALID", "value": "HERO"}
    }
  ],
  "review": {"reviewer": "REPLACE_ME", "method": "manual_source_pixels", "notes": ""}
}
```

枚举规则：

- 状态：`VALID`、`UNKNOWN`、`CONFLICT`；
- 牌：点数 `2..9,T,J,Q,K,A` 加花色 `C,D,H,S`，例如 `TS`、`QD`；
- 街道：`PRE_FLOP`、`FLOP`、`TURN`、`RIVER`；
- 动作：`FOLD`、`CHECK`、`CALL`、`BET`、`RAISE`、`ALL_IN`；
- 占用：`OCCUPIED`、`EMPTY`；
- scene：`table`、`deal_transition`、`action_transition`、`result`、`menu`、`overlay`、`signal_loss`、`reconnect`。

动作文字只记录屏幕上可见的“已完成动作”。动作金额不能从文字猜测；后续必须用该玩家筹码变化和底池变化重建并校验。

## 10. 阶段 G：最低覆盖要求

以下是进入完整标定验收前的最低数据量，不是“截图总数”，而是人工确认的有效、分散样本：

| 字段 | 最低正样本 | 最低负样本/额外要求 |
|---|---:|---|
| Hero 手牌 | 40 个不同手牌、至少 80 个稳定帧 | ≥40 个牌背、弃牌、遮挡、发牌动画帧；所有点数和四种花色均出现 |
| 公共牌/街道 | Flop ≥20、Turn ≥15、River ≥15 个不同稳定牌面 | Prefl 空牌面 ≥20；翻牌/发牌/遮挡负样本 ≥40 |
| 总底池 | ≥40 个不同数值 | ≥60 个非底池数字、动画、遮挡负样本；数字 0–9 均覆盖 |
| 座位筹码 | 每个 slot ≥12 个可读值，共 ≥96 | 每个 slot ≥10 个空座、遮挡、动画或相似数字负样本，共 ≥80；数字 0–9 均覆盖 |
| Dealer | 每个 slot ≥6 次，共 ≥48 | ≥50 个无标志、隐藏、动画或多候选负样本 |
| 座位占用 | ≥40 个稳定桌面状态，共 ≥320 个 slot 标签 | 覆盖 2 人、3–5 人、6–8 人以及加入/离开 |
| 已完成动作 | FOLD/CHECK/CALL/BET/RAISE 各 ≥10，ALL_IN ≥6 | ≥80 个 Hero 控件、昵称、头像、牌面、菜单和遮挡硬负样本；动作需分散到不同 slot |
| Hero 当前行动 | ≥40 个正样本 | ≥160 个非 Hero 回合、结果、菜单和动画负样本 |
| 时序 | 发牌 ≥20 组、动作 ≥30 组、换街 ≥10 组、手牌结束 ≥10 组 | 每组保留前/中/后三帧；断流/重连 ≥5 组 |
| 全局异常场景 | 不适用 | 菜单、遮挡、黑屏、花屏、断流等合计 ≥50 帧 |

至少来自 3 个独立采集会话。某字段没有达到覆盖要求时，报告为缺口，不得用相邻重复帧凑数。

## 11. 阶段 H：数据划分

按 `session_id + hand_id` 分组，建议：

- `train`：60%，用于模板和算法开发；
- `calibration`：20%，用于选择每个字段的阈值；
- `validation`：20%，冻结后一次性验收。

规则：

- 同一手牌、同一动作前后帧、感知近重复帧必须在同一 split；
- 三个 split 都必须包含稳定正样本和真实硬负样本；
- 验证集在 ROI、模板、算法和阈值冻结前不得用于调参；
- 如验证失败，记录失败原因，修改版本后重新建立新的锁定验证集，不能偷偷把失败帧移走。

## 12. 阶段 I：模板、阈值与置信度标定

每个字段独立生成证据，不允许一个全局阈值覆盖所有内容。

1. 在训练集上确定 ROI、图像预处理和模板；
2. 在 calibration 集上统计每个正样本和硬负样本的分数分布；
3. 为 Hero、Board、Pot、每类 Stack/Dealer/Action/Occupancy/Actor 分别选择阈值；
4. 冻结代码、配置、模板及其哈希；
5. 在锁定 validation 集运行一次生产识别器；
6. 输出所有误识别和 abstain，不得只报平均准确率。

模板必须是紧裁剪、隐私安全的牌面/字形/掩码，不得包含昵称、头像或完整桌面。

阈值原则：

- 锁定验证集必须保持零个错误 `VALID`；
- 正样本最低分与硬负样本最高分应存在可解释间隔；
- 分数重叠时应改进 ROI、模板或识别方法，并让困难帧 abstain，不能通过降低阈值掩盖问题；
- recall 必须单独报告，不能把大量 `UNKNOWN` 当作准确；
- `UNKNOWN` 和 `CONFLICT` 不得被序列化成默认值；
- 两帧确认、帧序号中断、断流和重连后的候选清空必须通过时序测试。

`evidence/field_metrics.json` 每个字段至少记录：

```json
{
  "field": "pot",
  "algorithm_version": "REPLACE_ME",
  "threshold": 0.0,
  "train_samples": 0,
  "calibration_positive_samples": 0,
  "calibration_negative_samples": 0,
  "validation_positive_samples": 0,
  "validation_negative_samples": 0,
  "correct_valid": 0,
  "false_valid": 0,
  "unknown_on_positive": 0,
  "conflict": 0,
  "lowest_accepted_positive": 0.0,
  "highest_rejected_negative": 0.0,
  "source_sessions": [],
  "code_sha256": "REPLACE_ME",
  "config_sha256": "REPLACE_ME",
  "template_sha256": "REPLACE_ME"
}
```

## 13. 阶段 J：座位映射与状态一致性

创建独立的 `seat_mapping.draft.json`，不得修改或覆盖现有 LDPlayer 映射。映射验收必须覆盖：

- 2–8 人不同占座组合；
- Dealer 在每个视觉 slot；
- Hero 固定为 slot 0 时的规范位置推导；
- 空座加入和离开；
- 无 Dealer、多个 Dealer、占用冲突时 fail closed；
- stack/action/dealer/occupancy 各自使用独立几何证据；
- 已完成动作只在动作、演员、筹码差和底池差一致时生成规范事件；
- 缺少金额、多个玩家同时变化或边池无法解释时不得猜动作金额。

对手当前行动者、复杂漏帧动作链、主池/边池如果尚无充分证据，应继续输出 `UNKNOWN` 或让策略 `ABSTAIN`。

## 14. 阶段 K：Replay 与性能证据

Replay 草案应引用：

- 源视频或原始帧哈希；
- normalization、TableMap、字段 calibration 和 seat mapping 的版本及哈希；
- 生产识别器代码版本；
- 每帧预期字段状态、状态版本、事件和拒绝原因；
- 数据授权和隐私审核结果；
- 是否使用真实采集卡输入。

至少报告：UVC 解码帧率和丢帧率、归一化加识别的 p50/p95 延迟、连续运行 30 分钟的断流/内存/恢复情况、拔插采集卡与切后台后的行为，以及多个采集设备存在时的设备选择规则。

性能数字只如实记录。没有项目负责人批准的阈值时，不得自行把“看起来够快”写成发布通过。

## 15. 阶段 L：落地到 PokerSense 仓库

标定证据通过后，执行 AI 需要把可公开、隐私安全的生产配置落到仓库；原始视频和完整帧仍留在私有标定目录。

建议新增而不是覆盖：

```text
configs/platform/wepoker_android_capture_card__<layout_id>.json
configs/platform/wepoker_android_capture_card__<layout_id>_seat_mapping.json
configs/vision/wepoker_android_capture_card/calibration.json
configs/vision/wepoker_android_capture_card/board_slot_layout.json
configs/vision/wepoker_android_capture_card/dealer_slot_layout.json
configs/vision/wepoker_android_capture_card/empty_slot_layout.json
configs/vision/wepoker_android_capture_card/hero_slot_layout.json
configs/vision/wepoker_android_capture_card/<隐私安全模板>
```

同时完成：

- 为新 platform/layout 增加资源加载和序列化测试；
- 用锁定验证集增加字段级回归测试和硬负样本测试；
- 增加归一化、尺寸不匹配、镜像/旋转错误、断流和两帧确认测试；
- 增加 2–8 人视觉 slot 映射、Dealer 冲突和空座变化测试；
- 增加动作文字去重、筹码/底池不守恒时拒绝事件的测试；
- 确认 PyInstaller/安装包只包含生产配置和隐私安全模板，不包含原视频、完整截图、标签工作目录或联系人信息；
- 根据实际能力更新 `docs/capture-and-table-mapping.md`、`docs/capture-replay.md`、中英文 README 和 `AGENTS.md`；
- 运行项目约定的 pytest、Flake8、差异检查和本地打包检查。

如果 UVC 实时 CaptureBackend 尚未实现，标定完成后仍应明确写“配置已准备，实时后端未接入”，不得把数据通过等同于端到端产品通过。

## 16. 验收门槛

只有同时满足以下条件，报告才可写“标定通过”：

- 硬件和采集参数完整、可复现；
- 归一化内容边界在全部会话和重连后稳定；
- ROI 经跨会话、跨人数和跨街道复核；
- 每个生产字段达到最低覆盖量；
- 数据按手牌/会话隔离，无相邻帧泄漏；
- 所有配置、模板、数据和代码有 SHA-256；
- 锁定验证集没有错误 `VALID`，abstain 和漏检已完整报告；
- 两帧确认、动画负样本、断流和重连测试通过；
- 视觉 slot 到规范座位映射无歧义；
- 动作事件满足筹码和底池一致性，否则拒绝生成；
- 私有原图未进入 Git 或安装包；
- Replay 明确绑定真实采集卡来源、识别器版本和人工真值；
- 未实现能力仍明确标为未实现或 `UNKNOWN`。

只要任一关键字段、映射、来源或验证证据缺失，最终状态必须写为 `PARTIAL` 或 `BLOCKED`，不得宣称完整通过。

## 17. 最终交付报告模板

`evidence/review_report.md` 必须按以下结构输出：

```markdown
# Capture Card Calibration Review

## 结论
- 状态：PASS / PARTIAL / BLOCKED
- platform_id：
- layout_id：
- normalization version：
- source sessions：

## 硬件与采集参数
...

## 画面稳定性
- 原始尺寸：
- 归一化尺寸：
- 最大内容边界漂移：
- 旋转/镜像/裁剪：

## 字段结果
| 字段 | 正样本 | 负样本 | false VALID | UNKNOWN | 结论 |
|---|---:|---:|---:|---:|---|

## 时序与重连
...

## 座位映射与动作一致性
...

## 性能
...

## 未解决问题
...

## 需要补录的精确清单
...

## 文件和版本哈希
...
```

## 18. 哈希与私有交付

macOS/Linux 可在交付根目录执行：

```bash
find . -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 shasum -a 256 \
  > SHA256SUMS
```

交付前后随机抽查文件，并核对压缩包 SHA-256。压缩包通过私有通道传输；不要上传公开 GitHub，不要让聊天软件二次压缩视频或图片。

## 19. 执行代理的停止条件

遇到下列情况必须停止当前标定、保留证据并向负责人报告，不得自行猜测：

- 采集画面会随机裁剪、缩放、镜像或改变方向；
- UVC 分辨率或像素格式在会话中变化；
- 游戏内容边界漂移超过 2 个归一化像素；
- 无法确定 slot 0 或座位布局发生变化；
- 正负分数分布重叠且会产生错误 `VALID`；
- 原始素材缺少授权或隐私审核；
- 标签只能靠牌局逻辑推断而不是肉眼确认；
- 验证集被提前用于调参或发生手牌级数据泄漏；
- 需要改动现有 LDPlayer/H5 标定才能让采集卡数据“通过”。

停止后只输出：已经完成的内容、失败证据、准确缺口和下一次补录清单。不要扩大能力声明。
