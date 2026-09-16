# 标准 Git 同步：诊断结论与可复用入口（GITHUB-001）

本文件说明 **PokerSense 在 Windows 本机上让标准 `git push` 可用**所需的配置、根因证据和使用方式。
它记录的是已验证事实，不是方案设想。

> 安全：本文件全部使用占位符（`$REPO`、`$PYTHON`、`$PROGRESS`），**不含**用户名目录、私有证据路径或任何令牌。

---

## 1. 症状与根因

| 项 | 事实 |
| --- | --- |
| 症状 | `git push` 挂起：一次约 3 分钟无输出；一次在 240s 外部超时后被终止（rc=124） |
| 误判一 | 「没有登录」——**错**。`gh auth status` 显示已登录（凭据存 keyring，scopes `gist, read:org, repo`） |
| 误判二 | 「代理不通」——**错**。`git fetch` / `git ls-remote` 均 1s 内成功；本机代理端口在监听 |
| **真实根因** | **凭据助手顺序**。本机 `credential.helper` 有两条：系统级 `helper-selector`（PortableGit）与用户级 `git-credential-manager.exe`。`-c credential.helper='!gh auth git-credential'` 只是**追加**到列表，**不移除**已有 helper；于是 GCM 仍先执行并阻塞（无人值守时等不到交互）。 |

### 修法（已验证）

```bash
# 关键：先 RESET 列表（空值），再添加 gh helper
git -c credential.helper= \
    -c credential.helper='!gh auth git-credential' \
    push origin refs/heads/"$BRANCH":refs/heads/"$BRANCH"
```

另需 `GIT_TERMINAL_PROMPT=0` 与 `GCM_INTERACTIVE=never`，确保任何情况下都不进入交互等待。

### 前后对比（实测）

| 项 | 修复前 | 修复后 |
| --- | --- | --- |
| `git push` | 挂起 / 240s 超时 rc=124 | **成功 2.8s，exit 0** |
| 远端分支 | 不存在 | 已创建 |
| 三方 SHA | 无从核对 | `local HEAD = remote ref = PR head` 全等 |

---

## 2. 可复用入口

薄封装：`tools/git_sync.py`（**只调用真正的 `git`/`gh`，不重写 Git 实现，不新建发布平台**）。

```bash
# 查看生效配置（脱敏：不含令牌）
$PYTHON tools/git_sync.py --repo $REPO --branch $BRANCH show-config

# 标准 push + 三方 SHA 校验（本地 HEAD / 远端 ref / PR head）
$PYTHON tools/git_sync.py --repo $REPO --branch $BRANCH \
    --progress $PROGRESS --timeout 120 --pr $PR push

# 只做校验（不推送）
$PYTHON tools/git_sync.py --repo $REPO --branch $BRANCH --pr $PR verify
```

### 它保证什么

- **凭据顺序**：helper 列表先重置、再加 gh helper（顺序有单测固定）。
- **真实退出码**：直接执行，`exit_code` 来自进程本身；**绝不用 `git push | tail` 取管道尾状态**。
- **外部计时器**：每条命令有超时；超时只 `taskkill /F /T` **自己这一条命令的子进程树**，不碰其他进程。
- **进度轨迹**：每 20 秒向 `$PROGRESS` 追加一条 JSON（`command_id`/`label`/`phase`/`elapsed_s`/`at_utc`）。
  这是**命令进度**，不是「研究有进展」的心跳。
- **无管道死锁**：stdout/stderr 由独立 reader 线程排空。
- **拒绝假成功**：三方 SHA 不全等即视为未通过；空值不算相等。
- **幂等发布**：`publish()` 只在树变化时才有提交；同分支已存在 PR 则复用，不重复开 PR。

### 它不做（硬边界）

不 `--force` / 不 `reset` / 不 `rebase` / 不 `clean`（有单测对其断言）；
不改仓库可见性、CI 配置、账号权限；不读写 secrets。

---

## 3. 元数据与 Git 的分工

| 用途 | 工具 |
| --- | --- |
| 版本对象（提交、分支、同步） | **标准 git**（经上述封装） |
| Issue / PR / review / Actions 元数据 | `gh` / 官方 API |

时间预算按任务要求：元数据/API 诊断 60s 外部超时；小规模 fetch/push 120s 外部超时。

---

## 4. 未修复 / 未验证（如实）

- 用户级 `git config` 的 GCM helper **未被修改**：本轮以**命令级作用域**解决，不改全局配置。
- 是否改用 `gh auth setup-git` 写入用户级 helper：**未执行**（需先备份并取得所有者确认）。
- 传输层强制 HTTP/1.1、`http.postBuffer` 调整：**未尝试**（根因不在传输层，无需盲改）。
