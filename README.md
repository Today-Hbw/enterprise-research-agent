# Enterprise Research Agent

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个轻量、可追踪、可扩展的企业研究 Agent。它把 LLM、企业知识库、公开网络、只读数据库、受限计算、浏览器和 MCP 工具统一到同一套运行时中，并为每次研究任务保留计划、来源、执行轨迹、用量与成本信息。

项目默认使用确定性离线模式：无需 API Key、不会访问外部服务，安装后即可体验完整的 Agent 调度与界面。需要真实能力时，可以逐项启用 OpenAI、豆包、Qdrant、RAG Platform、Brave Search、PostgreSQL、Playwright、Redis 或 MCP，而无需重写 Agent 核心。

> An open-source, traceable enterprise research agent that runs fully offline by default and supports opt-in production integrations.

## 核心特性

- **开箱即用**：默认离线 Planner 与 Stub 工具不依赖外部账号，适合开发、演示和测试。
- **全程可追踪**：同步与 SSE 流式 API 都会返回计划、来源、工具轨迹、延迟、Token 和成本估算。
- **知识检索可控**：支持文本摄入、URL 导入、确定性分块、授权过滤、锚定引用、混合检索和重排。
- **工具边界清晰**：统一 JSON Schema、权限等级、超时、并发限制、重复调用保护和错误边界。
- **真实集成按需启用**：所有外部能力默认关闭，通过服务端配置、域名白名单或只读权限显式开启。
- **内置评估与界面**：提供离线 Agent/Retrieval 评估，以及会话、计划、来源、轨迹和指标三栏 UI。

## 能力矩阵

| 能力 | 默认模式 | 可选真实后端 |
|---|---|---|
| LLM | 确定性离线 Planner | OpenAI、豆包 Responses API |
| 知识库 | 内存存储 | Qdrant、RAG Platform HTTP API |
| 检索 | 语义检索 | RRF 混合检索、Token overlap 重排 |
| Web | 固定响应 Stub | Brave Search、安全 HTTP Fetch |
| 数据分析 | SQL / Python Stub | 只读 PostgreSQL、受限表达式计算 |
| 扩展工具 | Browser / MCP Stub | Playwright、服务端配置的 HTTPS MCP |
| 状态与事件 | 进程内存 | PostgreSQL、Redis |

## 工作原理

```mermaid
flowchart LR
    Client[Web UI / API] --> API[FastAPI + SSE]
    API --> Runtime[Agent Runtime]
    Runtime --> LLM[Deterministic / OpenAI / Doubao]
    Runtime --> Registry[Tool Registry]
    Registry --> Tools[Knowledge / Web / SQL / Python / Browser / MCP]
    Tools --> Backends[Memory / Qdrant / RAG Platform / PostgreSQL]
    Runtime --> Result[Plan + Sources + Trace + Metrics]
```

运行时限制最大步骤数、总时长、单工具时长、并发数和重复调用；工具权限上限、Token 预算与成本预算均由服务端控制。高层计划随执行实时更新，但不会暴露模型的隐藏思维过程。

## 快速开始

要求 Python 3.12 或 3.13。

```bash
git clone https://github.com/Today-Hbw/enterprise-research-agent.git
cd enterprise-research-agent
python -m venv .venv
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

```bash
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

- Web UI：<http://localhost:8000>
- OpenAPI：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/api/health>

此时使用完全离线的默认配置，可直接观察计划、并行工具调用、来源引用和执行轨迹。

### Docker Compose

```bash
docker compose up --build
```

Compose 会在本机启动 Agent 与 Qdrant。回退密钥仅用于本地开发；对外部署前必须设置强随机的 `KNOWLEDGE_ADMIN_TOKEN` 和 `QDRANT_API_KEY`。

## 配置真实能力

先复制示例配置；不要提交含密钥的 `.env`：

```bash
cp .env.example .env
```

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

常用配置示例：

```dotenv
# OpenAI；也可使用 LLM_PROVIDER=doubao 与对应 DOUBAO_* 配置
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-5-mini

# Qdrant
KNOWLEDGE_BACKEND=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key

# RAG Platform 需要 hybrid 检索与 knowledge_base_id
# KNOWLEDGE_BACKEND=rag-platform
# KNOWLEDGE_RANKING=hybrid
# RAG_PLATFORM_BASE_URL=http://localhost:8001
# RAG_PLATFORM_API_KEY=your_api_key

# Brave Search
WEB_SEARCH_BACKEND=brave
BRAVE_SEARCH_API_KEY=your_brave_subscription_token

# 只读 PostgreSQL
SQL_BACKEND=postgres
POSTGRES_DSN=postgresql://research_readonly:password@localhost:5432/research
POSTGRES_ALLOWED_SCHEMAS=public,analytics

# 可选受限计算与浏览器
PYTHON_BACKEND=isolated
BROWSER_BACKEND=playwright
BROWSER_ALLOWED_HOSTS=app.example.com,.approved.example
```

启用真实 LLM 后，缺少 API Key 会直接启动失败，不会静默降级。数据库账号本身也必须只拥有允许 Schema 的读取权限。

### 受控知识摄入

设置 `KNOWLEDGE_ADMIN_TOKEN` 后才能写入知识库：

```bash
curl -X POST http://localhost:8000/api/knowledge/documents \
  -H "Content-Type: application/json" \
  -H "X-Knowledge-Admin-Token: replace-with-a-long-random-secret" \
  -d '{"title":"Supplier policy","content":"Quarterly review is required.","knowledge_base_id":"policy","allowed_principal_ids":["demo-user"]}'
```

URL 导入还需设置 `HTTP_FETCH_BACKEND=safe` 和 `HTTP_ALLOWED_HOSTS`。支持 UTF-8 文本、Markdown、HTML、CSV、JSON/JSON-LD 和可提取文本的 PDF；加密 PDF、纯扫描 PDF 与 Office 文档暂不支持。

更多配置见 [`.env.example`](.env.example) 和[部署配置](docs/部署配置.md)。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/health` | 服务状态与当前模式 |
| `GET` | `/api/tools` | 工具目录、Schema 与权限 |
| `POST` | `/api/knowledge/documents` | 摄入文本知识 |
| `POST` | `/api/knowledge/import-url` | 安全下载、解析并导入 URL |
| `POST` | `/api/chat` | 同步研究任务 |
| `POST` | `/api/chat/stream` | SSE 流式研究任务 |
| `GET` | `/api/conversations` | 查询会话列表 |
| `GET` | `/api/conversations/{id}` | 查询单个会话 |
| `GET` | `/api/runs` | 查询 Run 汇总与指标 |
| `GET` | `/api/runs/{id}` | 查询单个 Run 与完整轨迹 |
| `GET` | `/api/runs/{id}/events` | 重放 Run 事件 |

```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"调研市场并结合采购数据做分析"}'
```

## 测试与评估

```bash
pytest
ruff check .
python -m app.evaluation --dataset evals/demo.json --output output/evaluation/demo-report.json
```

评估会运行真实 Agent Runtime 与知识检索链路，输出 JSON 报告，并在阈值未达标时返回退出码 `1`。它覆盖工具路由、引用、Recall@K、MRR、Hit Rate、调用量与延迟，不等同于生产答案质量评测。

## 安全边界

- 默认配置不访问外部网络、数据库或浏览器。
- 知识访问过滤由服务端 `AccessContext` 生成；身份头仅应在可信网关后启用。
- HTTP Fetch 与 Browser 使用域名白名单，并拒绝 URL 凭据、IP 字面量和非公网地址。
- SQL 只允许单条只读查询，并配合 Schema 白名单、超时和行数限制。
- 隔离 Python 仅执行 AST 白名单约束的表达式，不是通用代码沙箱。
- 项目尚未内置完整认证、租户目录、生产限流与持久化审计，公开部署前需在外围补齐。

更多内容见[安全设计](docs/安全设计.md)。

## 文档

- [项目总览](docs/项目总览.md)
- [架构详解](docs/架构详解.md)
- [工具系统](docs/工具系统.md)
- [API 接口](docs/API接口.md)
- [安全设计](docs/安全设计.md)
- [部署配置](docs/部署配置.md)

## 参与贡献

欢迎提交 Issue 与 Pull Request。请保持默认离线模式可运行，为行为变更补充测试，并在提交前运行 `pytest` 和 `ruff check .`。请勿提交 `.env`、API Key、数据库凭据、客户数据或内部文档。

## 开源许可

本项目基于 [MIT License](LICENSE) 开源。你可以自由使用、复制、修改、合并、发布和分发本项目，但须保留原始版权与许可声明。
