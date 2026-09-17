# PokerSense 离线试用入口（USABLE-001 U2）

版本 **`aa-trial-u2.0`** · 记录实现 **`aa-analysis-record-v1`** · 随仓库提交，独立于用户既有安装。

## 怎么启动

双击 `START-TRIAL.cmd`，或在命令行运行：

```
python launch\u2-trial\trial_launcher.py                # 首选端口 8791
python launch\u2-trial\trial_launcher.py --port 8800    # 指定端口
python launch\u2-trial\trial_launcher.py --self-check    # 只检查资源/端口/目录，不启动
python launch\u2-trial\trial_launcher.py --version
```

它用项目**已有的本机 Python**，不装新依赖、不改全局环境或代理、不覆盖任何旧程序。
端口被占用时**不会结束任何进程**，只报告并改用明确空闲端口；服务**就绪后**才打开浏览器。

## 这一步能做什么

1. 在「设置与分析 → 本桌规则」保存一份完整桌规。
2. 在「录入一手已结束牌局」填一手人工牌局并「核对输入」。
3. 「计算条件收益」→ 得到本次输入在本次假设与本次桌规下的各动作条件净 EV。
4. 在「保存本次分析并重开」里点「保存本次分析」→ 冻结本次事实与来源、对手假设、
   核对时的桌规版本、规范输入与服务端实际完成结果。
5. 关掉程序再启动 → 「刷新记录」→「打开已保存分析」读回同一条记录，**不会重新计算**，
   也**不会**把当前桌规套到历史结果上。

## 必须知道的未验收项

- **真实牌局输入尚未验证**（`REAL_HAND_ACCEPTANCE_PENDING`）：现有授权牌局不满足内核前提。
- **策略强度未评估**（`NOT_ASSESSED`）：这里只有条件净 EV，不是 GTO，也不是实战胜率。
- 只支持**河牌**、恰好**三名活跃玩家**、**单底池**；不支持边池、全下跨越、翻牌/转牌。
- 对手范围与响应权重都是**人工假设**，换一个假设数字就会变。
- 本入口**不打开采集设备**、不回放媒体、不出实战提示。

## 目录与隔离

| 路径 | 内容 |
| --- | --- |
| `launch/u2-trial/state/profile.json` | 试用专用配置（首次启动时创建） |
| `launch/u2-trial/state/table-rules.json` | 试用专用桌规 |
| `launch/u2-trial/state/records/analysis-records/<记录号>/record.json` | 已保存的分析记录 |

这些都**只写在试用目录里**：既有的 `%LOCALAPPDATA%` 程序、桌面快捷方式、用户自己的规则与记录都不会被读取、覆盖或删除。
升级试用版本时先复制一份 `state` 目录即可保留记录。
