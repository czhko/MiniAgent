# MiniAgent —— 轻量级多模型 Agent 框架

本项目核心是将 Agent 工具调用框架以 API 形式打包，以直接适配前端的 **MiniAgent**（API 请求 → 调模型 → 执行工具 → 返回结果），上层通过管线引擎编排多 Agent 协作。并尝试解决基于状态机的多模型分工框架，用于长线扮演内容生成。

## 架构

```
L0 基石      paths.py  timeutil.py  fsutil.py  http_utils.py  prompt.py
L1 编解码    codec/        owui.py
L2 基础设施  infra/        settings.py  templates.py  logger.py  debug.py
L3 存储      store/        ua.py
L4 领域      agent/        core.py
             drivers/      base.py  anthropic.py  openai.py
L5 编排      engine/       execution.py  pipeline.py  plugins.py
L6 入口      entry/        server.py  admin.py  admin.html
```

## 快速开始

```bash
pip install anthropic httpx
python core/server.py
```

启动后访问 `http://localhost:18787/admin` 进入管理面板。

### 最小配置

1. 打开 Admin → 路由 → 添加路由（API key + base URL + 模型）
2. Admin → 链路 → 添加链路（绑定路由 + system prompt）
3. 在 OpenWebUI 中添加 OpenAI 兼容连接指向 `http://localhost:18787/v1`

## 核心能力

| 能力 | 说明 |
|------|------|
| **双协议 Agent** | Anthropic SDK + OpenAI 兼容协议，按路由切换 |
| **工具系统** | Read / Write / Edit / Bash / Glob / Grep / WebSearch / WebFetch / TavilyExtract / TavilyCrawl / TavilyMap / TaskCreate / TaskGet / TaskList / TaskUpdate / SubAgent / DescribeImage |
| **多模型管线** | 旁路 Agent（上下文维护）+ 主路 Agent（内容生成），独立 conversation |
| **Admin 面板** | Dashboard、路由/链路/插件可视化配置、日志查看、文件管理器、Debug 模式 |
| **Tavily 集成** | AI 专用搜索 API，WebSearch 优先走 Tavily，失败自动退化 DuckDuckGo |
| **视觉模型链** | 表格化配置多端点降级，DescribeImage 依次尝试直到成功 |
| **UA Store** | 适配 OWUI 的对话缓存，多哈希回退，跨请求恢复工具上下文 |
| **插件系统** | 用户注入 + 条件事件触发 |
| **文件追踪** | MD5 清单差异备份，Write/Edit/Delete/Bash 操作前自动备份 |
| **工作区隔离** | 每条链路/路由独立 workspace 子目录 |
