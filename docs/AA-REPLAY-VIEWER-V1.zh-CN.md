# AA 冻结回放与人工确认台账

PR #18 经所有者授权，以 merge commit `fb703e2` 合并；人工确认从该 main 继续。
只接受 SHA `7f86f4cd631dfae997b2fe7dc8ce42fda819f6b52d8b0127f56e6173e18161bc`
的 observations.jsonl，连续 0–1800 共 1801 帧、九个视觉槽位。
不读取原视频，不执行识别器、训练、采集、输入控制或策略建议。

## 启动和确认

使用已安装项目 desktop/dev 依赖的 Python，在仓库根目录运行：

```powershell
$env:PYTHONPATH='src;.'
$env:PYTHONUTF8='1'
python -m tools.aa_replay_viewer --observations '<冻结日志完整路径>' --ledger '<新的私有目录>/confirmations.jsonl' --port 8766
```

打开 `http://127.0.0.1:8766`。不传 `--ledger` 时仍可只读查看。
支持前后帧、方向键、滑块、跳帧和播放暂停；输入获得焦点时暂停播放。
金额使用非负整数字符串；actor 使用 0–8 槽号；枚举见输入提示。
special_mode 输入 critical_hit/squid/insurance 三个对象，各有布尔型
enabled/triggered。false 表示人工确认未启用，null 表示未知。

点击“确认”后，先 append、flush、fsync 成功，界面才显示“人工确认”与值。
同帧同字段更正另加一行；最新确认用于显示，历史不删，其他帧不继承。
自动输出、缺失声明和确认 SHA 保留。不要将合成测试确认冒充真实人工真值。

## 缺失声明修正（总控规格第六节）

未实现集合直接来自每帧 `incomplete_fields`，页面原样展示；硬编码只校验
声明名称，未知名称报 `INCOMPLETE_FIELDS_CONTRACT_MISMATCH`。
已声明缺失却有完整已知源值时，显示值并报 `INCOMPLETE_FIELDS_VALUE_CONFLICT`，
不隐藏新数据。未来 pot 有值且从声明移除后正常显示。current_actor 未声明
且为空时显示“未知”。participation 和 acceptance 都有明确字段行。
可见动作字样与缺少完整行动历史可同时成立，两者分开显示。

**无逐字段数值置信度；逐帧缺失由 `incomplete_fields` 声明。**
未声明的空字段仅说明未知，不编造原因。源重置/不支持场景明确显示。

## 台账、分区与曝光

JSONL 每行含 schema_version、session_id、frame、seat、field、model_output、
model_reason、human_value、source_media_sha256、recorded_at_utc、frame_zone、
training_eligible、exposure；另含 replay_sha256、policy_sha256、sequence、event、
previous_sha256、entry_sha256。全局字段 seat=null。field 仅允许
seat_presence/current_bet/pot/actor/action/special_mode。
动作人工值是可见字样标签，muck 不会变成 fold，也不代表完整法律行动历史。

每次访问验证源绑定与哈希链；截断/改值/分区变更不静默恢复。写入有线程锁及
跨进程文件锁。台账及 `.lock` 只放私有目录，不提交真实确认或媒体。
默认所有帧为 unknown，training_eligible=false。可显式传 `--zone-policy`：

```json
{
  "session_id": "operator-reviewed-session",
  "replay_sha256": "7f86f4cd631dfae997b2fe7dc8ce42fda819f6b52d8b0127f56e6173e18161bc",
  "zones": [
    {"first": 0, "last": 100, "zone": "development"},
    {"first": 101, "last": 1800, "zone": "reserved"}
  ]
}
```

上例只是格式，**不是实际分区事实**。必须使用既有授权/登记的分区。
区间闭合、排序、不重叠；未覆盖帧为 unknown。只有 development 为 true，
reserved/exploration/unknown 全部 false。浏览器不能修改分区或训练资格。
已有台账的 policy SHA 不允许更改，不能通过新建台账洗掉历史曝光。

`POST /api/exposure` 接收 `{"entry_sha256":"待用于训练的确认行SHA"}`，追加
training_exposure 事件；训练调用方必须在消费样本前登记（保守先标记再训练）。
该帧后续确认永远为 trained；历史行保留原 exposure 与 SHA。
报表/导出另有 `trained_frames` 表示当前曝光，消费者必须合并读取。
导出 GET 本身不训练，不把下载伪称已训练。已曝光帧永不能作独立验收。
此约束依赖台账保留，无法阻止操作者在项目外删除台账或绕过入口训练。

## 门禁与复盘

strategy_eligible 仅表示离线必需字段完整性，无 Advice/live 权限含义。
要求场景正常、无重置/声明冲突；源 acceptance=true、Hero 两张牌、完整
board/participation/full_actions；有效 pot/actor/三个特殊模式和所有座位
presence；参与/all_in 座位还需 stack/current_bet。actor 必须在参与座位。
自动/人工值均检查类型与结构。人工字样不补全历史，也不能设置 acceptance。
本冻结回放仍缺这些源字段，资格保持 false。

`GET /api/report` 与“只读复盘报表”统计最新确认，每字段给出一致、不一致、
可比分母、未知/拒识、确认分母。金额 100 与 "00100" 等值，原始值不变。
更正不膨胀分母，另列全部 confirmation_events。by_frame_seat_field 保留明细。
没有权威手号，hand=UNKNOWN；没有完整机会真值，误报率/漏记率为 null 并附
原因，不能用稀疏确认宣称独立准确率。报表含台账字节 SHA 和规范 JSON SHA。

## 导出接口

`GET /api/export/all|money|glyph` 只输出 training_eligible=true 的最新确认，
含来源/帧/确认 SHA、ledger_sha256、export_sha256、trained_frames。
金额 inputs.value 对应 `aa8_money_bank_v2.labelled_glyphs(value, patch)`；
字样 label/slot 对应 `GlyphSupplementV3.references[label]=(image,slot)`，
只选 fold/muck/all_in。其他字段在 all 中作为未来数据集标签保留。
这些是已有函数所需的标签参数，不是已生成的 bank.npz 或模板图。
冻结日志没有源图像/裁剪 SHA；后续调用方仍须提供源帧绑定核验的图像、
合适的八槽/九槽适配及需要的竞争模板。不会伪造 first/second pool、八槽位置
或图片哈希，也不直接塞进旧八槽 build 命令。实际增量训练属于后续工作。

所有 POST 需要同源 JSON 与 `X-AA-Confirmation: 1`。仅绑定 127.0.0.1，
Host 白名单限制第三方域名写入；用户输入按文本渲染。

## 本次验证与证据

Windows Python 3.12，环境 `PYTHONPATH=src;.`、`PYTHONUTF8=1`：

```powershell
$py = 'C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe'
& $py -m pytest -o addopts= -q tests/tools/test_aa_replay_viewer.py tests/tools/test_aa_confirmation_ledger.py
# exit 0: 27 passed, 1 dependency warning
& $py -m pytest -o addopts= -q
# exit 0: 3421 passed, 7 skipped, 1 dependency warning (35.04 s)
& $py -m flake8 src tests tools
# exit 0
git diff --check
# exit 0
```

只读真实日志检查：1801 帧 × 57 行、声明冲突 0、strategy_eligible=true 为 0。
合成真实 HTTP 验证：3 次确认持久落盘，含 1 条 reserved（训练资格 false）；
金额和字样各导出 1 条 development。重启加载、并发写入、篡改/截断拒绝、
更正与曝光单调性、类型等值比较、必需结构校验均有 focused 回归。
独立代码复审及 focused 重跑不等于独立视觉准确率验收。
浏览器可视检查 **BLOCKED**：连接两次返回 `nodeRepl.fetch request failed`。

以下均为**合成确认工程证据**，不是真实人工金标；私有目录
`aa-human-confirmation-ledger-v1-20260915`，旧证据未覆盖：

- 台账字节 SHA：`433064d07a3649522144765ace564e873e7f0d5d6a0603de9ff48c4104837816`
- 报表规范内容 SHA：`5e76af824fc4430f7b0b73f44914c99f1160f8778efaeea38a6d4a134a6f760c`
- 工程证据 manifest SHA：`1b9c2c67bba514a0b388a70f499a0beea13aa762f495548b8a357391575e028c`
