# MiniAgent — 轻量级多模型 Agent 框架

本项目核心是将agent工具调用框架以API形式打包，以直接适配前端的 **MiniAgent**（API请求 → 调模型 → 执行工具 → 返回结果），上层通过管线引擎编排多 Agent 协作。并尝试解决基于状态机的多模型分工框架，用于长线扮演内容生成。

## 扮演核心矛盾

小说/长内容生成涉及世界观、人设、人物关系、事件链、规则、文风、输出格式等多个维度。Transformer 注意力随上下文增长而消散——单一模型同时承担意图分析、状态管理、内容生成、质量校验，必然崩溃。

**解决路径**：小模型分工承担意图分析和状态管理，每轮为生成模型装载当前场景元信息并卸载离场内容。明确告诉多个模型各自做什么，比让一个模型在多状态间纠结更合理。

## 架构

```
L0 基石      paths.py  timeutil.py  fsutil.py  http_utils.py  prompt.py
L1 编解码    codec/        owui.py  events.py
L2 基础设施  infra/        settings.py  templates.py  logger.py  debug.py
L3 存储      store/        ua.py
L4 领域      agent/        core.py  safety.py
             drivers/      base.py  anthropic.py  openai.py
L5 编排      engine/       execution.py  pipeline.py  plugins.py
L6 入口      entry/        server.py  admin.py  admin.html
```

## 快速开始

```bash
pip install anthropic httpx
python core/server.py
```

启动后访问 `http://localhost:18789/admin` 进入管理面板。

### 最小配置

1. 打开 Admin → 路由 → 添加路由（API key + base URL + 模型）
2. Admin → 链路 → 添加链路（绑定路由 + system prompt）
3. 在 OpenWebUI 中添加 OpenAI 兼容连接指向 `http://localhost:18789/v1` （本项目暂时仅兼容OpenWebUI 0.9.6+的前端）

## 核心能力

| 能力 | 说明 |
|------|------|
| **双协议Agent适配** | Anthropic SDK + OpenAI 兼容协议，多 Provider 路由 |
| **轻量化工具系统** | Read / Write / Edit / Bash / Glob / Grep / WebSearch / WebFetch / TaskCreate / SubAgent / DescribeImage / 后台任务 |
| **多模型分工管线** | 旁路 Agent（状态机维护上下文）+ 主路 Agent（写作），独立 conversation |
| **Admin 面板** | Dashboard、路由/链路/插件可视化配置、日志查看、文件管理器、Debug 模式 |
| **UA Store** | 适配OpenWebui前端不返回think块的多哈希回退缓存，跨请求恢复工具上下文 |
| **插件系统** | 用户注入（消息前后追加）+ 条件事件（轮数/字符数触发 LLM 任务） |
| **文件变更追踪** | Git-like MD5 清单，Write/Edit/Delete/Bash 操作前差异备份 |
| **工作区隔离** | 每条链路/路由独立 workspace 子目录，共享区读、隔离区写 |

## 工具清单

| 工具 | 用途 |
|------|------|
| Read | 读取文件 |
| Write | 创建/覆盖文件（隔离区内） |
| Edit | 精确字符串替换（隔离区内） |
| Bash | 执行命令（路径穿越防护 + 危险命令拦截） |
| Glob | 文件名模糊搜索 |
| Grep | 文件内容正则搜索 |
| WebSearch | 网络搜索 |
| WebFetch | 抓取网页内容 |
| TaskCreate | 创建结构化任务 |
| SubAgent | 独立线程子 Agent（防递归、30min 超时） |
| DescribeImage | 视觉模型识图 |
