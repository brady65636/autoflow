# AutoFlow scheduling MVP

本目录现在包含 SQLite + SQLAlchemy 的调度与工单 MVP 后端闭环，兼容当前 Python 3.11.5；后续可升级到 Python 3.12。包含调度核心、事务工单服务和 FastAPI API；暂不包含 Docker、Agent 编排或 CP-SAT。

## 执行路径

1. 上游把标准作业转换成 `TaskRequirement`：品牌、单门店、车辆、执行时长、时间窗口、所需能力、工位类型和稀缺设备类型。
2. `capabilities` 定义能力，`technician_capabilities` 保存工程师与能力的多对多关系及有效期；调度侧转换为 `EffectiveAbility`，以二元有效能力匹配，不计算熟练度。
3. `FirstFitPlanner.plan()` 从最早时间开始按分钟尝试，依次过滤门店/能力/技师要求、可用技师、工位和设备。
4. 每个候选区间联合检查车辆、技师、工位、设备的既有 `ResourceReservation`。所有任务不可中断；等待不进入调度。
5. `plan()` 找到第一个完整分配就返回 `SchedulingResult(status="FEASIBLE", plan=...)`；找不到时返回 `SchedulingResult(status="INFEASIBLE", reasons=...)`。原因在同一次搜索中记录。区间采用左闭右开语义，相邻区间不冲突。

一个执行任务恰好分配一个技师和一个工位，设备可以为 0 个或多个；同车任务通过车辆预留实现互斥。规划上下文明确限定单品牌、单门店。

## 数据库闭环

默认使用 SQLite 文件 `autoflow.db`；测试使用内存 SQLite。数据库层位于：

- `database.py`：SQLite engine、Session 和表初始化
- `db_models.py`：ORM 表；工程师与能力通过 `technician_capabilities` 多对多关联
- `repository.py`：基础 CRUD
- `seed.py`：可重复执行的演示数据
- `db_planner.py`：数据库记录到 First-Fit 调度器的适配
- `work_order_service.py`：确认后的 CandidatePlan 落单、资源锁定、取消、改约和门店执行状态
- `work_order_api.py`：Customer/Store 工单 API 适配层
- `work_order.py`：四状态工单状态机

初始化基础演示数据：

```bash
python -c "from autoflow_scheduling.database import create_session_factory; from autoflow_scheduling.seed import seed_demo; f=create_session_factory(); s=f(); seed_demo(s); s.close()"
```

这个命令会创建 `autoflow.db`，并插入基础技师、车辆、工位和共享设备。`seed_demo` 已做幂等保护，重复执行不会重复插入。

生成标准单品牌 4S 店数据：

```bash
python -c "from autoflow_scheduling.database import create_session_factory; from autoflow_scheduling.seed import seed_standard_4s; f=create_session_factory(); s=f(); seed_standard_4s(s); s.close()"
```

`seed_standard_4s` 通过 Repository CRUD 写入 `vw-4s-store-001`，包含8名技师、8个工位、6台共享设备，以及17条工程师能力关联。客户车辆实例不在本 seed 中插入；调度直接使用 `VehicleProfile(category, powertrain)`，示例覆盖轿车、SUV/吉普、两厢车，以及燃油、混动和纯电。

## 不可行原因

```python
result = planner.plan(task)
```

`result.status` 为 `INFEASIBLE` 时，`result.reasons` 会返回结构化原因：

```text
SCOPE_MISMATCH
NO_QUALIFIED_TECHNICIAN
NO_COMPATIBLE_WORKSTATION
NO_REQUIRED_EQUIPMENT
WINDOW_TOO_SHORT
RESOURCE_CONFLICT
NO_AVAILABLE_TIME
```

现在不再通过第二次调用查询原因；一次 `plan()` 调用同时返回计划或结构化不可行原因。

## 运行

启动工单 API：

```bash
uv run uvicorn autoflow_scheduling.app:app --reload
```

启动 API 后，可以使用 CLI 直接体验 Agent：

```bash
uv run autoflow-agent
```

CLI 会在内存中保存本次登录得到的 Bearer Token，并将每条消息发送到
`POST /api/agent/chat`。也可以发送单条消息后退出：

```bash
uv run autoflow-agent --message "查询我当前 CONFIRMED 状态的工单"
```

API 地址默认是 `http://127.0.0.1:8000`，也可以通过环境变量或参数覆盖：

```bash
AUTOFLOW_API_BASE_URL=http://127.0.0.1:8000 uv run autoflow-agent
```

在 `backend/` 下执行测试：

```bash
uv sync --extra dev
uv run --extra dev pytest -q
uv run --extra dev ruff check .
```

依赖只有 Pydantic 2；pytest 和 ruff 位于开发依赖。每次运行命令都显式使用 `--extra dev`，避免默认同步时移除开发工具。

MVP 标准作业配置位于 `src/autoflow_scheduling/catalog.py`，包含到店诊断、常规保养、发动机故障灯诊断、电气诊断、制动检查、更换刹车片、四轮定位、竣工质检。配置中的 OBD 扫描仪、厂家专用诊断仪和四轮定位仪作为稀缺共享设备示例；万用表等普通工具不进入调度。真实门店应按实际设备数量和共享规则配置。

## RAG 检索接口

完整本地模型链路接口，独立于工单领域路由。路由定义位于 `src/autoflow_scheduling/knowledge_api.py`，应用只在 `work_order_api.create_app()` 中组装该 router；检索实现位于 `knowledge/retrieval_service.py`：

```text
POST /api/knowledge/retrieve
```

请求体：

```json
{
  "query": "What is the purpose of the EA211 high-pressure fuel system?",
  "question_type": "specification"
}
```

接口执行：已存 SQLite embedding → Dense top-50 + BM25 top-50 → RRF Dense:BM25=20:1 → 15 chunks → 本地 Qwen3-Reranker → 业务规则融合 → 去重 section。返回 `chunks`、`sections`、每个 chunk 的 reranker/business-rule/final score，以及实际算法配置。需要 `CUSTOMER_AGENT` 或 `SERVICE_ADVISOR` 身份；模型路径通过 `AUTOFLOW_EMBEDDING_MODEL` 和 `AUTOFLOW_RERANKER_MODEL` 配置。

## RAG 解析产物导入 SQLite

知识库使用与工单相同的 SQLAlchemy/SQLite 数据库，包含
`knowledge_documents`、`knowledge_sections` 和 `knowledge_chunks` 三张表。导入会先校验
ID 与引用关系，再在单个事务中按 `document_id` 整体替换，重复执行不会追加重复记录。

导入当前 v4 的 10 文档语料：

```bash
uv run autoflow-knowledge-import \
  ../research/automotive-pdfs/pipeline-10pdf-v4-ocr-run/combined_sections.json \
  ../research/automotive-pdfs/pipeline-10pdf-v4-ocr-run/combined_chunks.json \
  --documents ../research/automotive-pdfs/pipeline-10pdf-run/selection.json \
  --pipeline-version 4 \
  --database-url sqlite:///./autoflow.db
```

当前 SQLite 保存解析后的 section 正文和 chunk 的 `text`、`index_text` 及结构/质量元数据。chunk 的 embedding 以 little-endian float32 BLOB 保存在 `knowledge_chunks.embedding`，并通过 `embedding_model`、`embedding_dimension` 记录模型和维度；正文 hash 校验不包含派生向量字段。
尚未建立向量或 FTS5 索引。导入时会验证原 PDF `source_sha256` 与 manifest 一致性，并保存
每个 section/chunk 的 `content_sha256` 和文档级 `artifact_sha256`。只有输入 hash 与数据库读回
重算结果都一致时，重复导入才会跳过该文档。

可随时对数据库中的全部知识内容执行完整性校验：

```bash
uv run autoflow-knowledge-verify --database-url sqlite:///./autoflow.db
```

为已导入的 chunk 生成并保存 embedding：

```bash
uv run autoflow-knowledge-embeddings \
  ../research/automotive-pdfs/pipeline-20pdf-run/combined_chunks.json \
  D:/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/<snapshot> \
  --database-url sqlite:///./autoflow.db \
  --model-name Qwen3-Embedding-0.6B
```

写入后应确认 `knowledge_chunks` 的 `embedding` 非空、维度一致，且模型名与本次索引配置一致。

发现任意正文、结构元数据、计数或 hash 被修改时，命令会返回非零退出码并指出首个不一致记录。

## 并行 PDF 批量摄取

从 manifest 选择 20 份 PDF，以独立进程运行摄取、合并产物并在全部验证通过后导入 SQLite：

```bash
uv run autoflow-knowledge-batch \
  ../research/automotive-pdfs/pdf_manifest.json \
  ../research/automotive-pdfs \
  ../research/automotive-pdfs/pipeline-20pdf-run \
  --limit 20 --max-workers 20 \
  --database-url sqlite:///./autoflow.db
```

每份文档写入独立目录并复用 ingestion fingerprint；中断后重跑时，已完成文档会跳过。
批级 `combined_sections.json`、`combined_chunks.json` 以及 SQLite 导入只在全部文档完成且
跨文档 ID/引用验证通过后发布。20 路适合压力测试；本机约 16GB 内存的实测表明 4–5 路完成
吞吐更合理，20 路会因资源争用显著变慢。

## LangSmith Trace（可选）

后端使用 LangSmith Python SDK 记录 ingestion 与 query/evaluation 的嵌套 run。未同时设置
`LANGSMITH_TRACING=true` 和 `LANGSMITH_API_KEY` 时完全 no-op；SDK 初始化、上报或 flush
失败不会影响业务流程。短命 ingestion/evaluation CLI 会在结束时调用 `Client.flush()`。

请将密钥放入本机环境变量，不要写入仓库：

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<new-key>
LANGSMITH_PROJECT=autoflow-rag
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

如果账户位于 EU/APAC 等区域，按 LangSmith 控制台提供的地址修改 `LANGSMITH_ENDPOINT`。
Trace 只发送阶段耗时、状态、数量、候选 ID/排名等有界元数据，不发送 PDF、section、Prompt
或完整查询正文。适配层位于 `src/autoflow_scheduling/observability/langsmith_tracing.py`，业务代码
不直接依赖 SDK API。

当前 ingestion pipeline version 为 **4**。解析顺序为：PyMuPDF4LLM Markdown；乱码率超过
`--replacement-ratio-threshold` 时回退 PyMuPDF plain text；无可提取文本时使用 Tesseract OCR。
默认 OCR 参数为 `--ocr-dpi 200 --ocr-language eng+chi_sim`。成功 fallback 页面会标记 warning，
OCR/解析仍失败的页面会进入 quarantine，且不会生成可索引 section。Docker 镜像已安装
`tesseract-ocr`、`tesseract-ocr-eng` 和 `tesseract-ocr-chi-sim`。

Windows 本地需要安装 Tesseract 5 和对应 `.traineddata`，并设置：

```powershell
$env:AUTOFLOW_TESSDATA_PREFIX = "C:\path\to\tessdata"
```

管道会依次查找 `AUTOFLOW_TESSDATA_PREFIX`、`TESSDATA_PREFIX`、用户级
`autoflow-ocr/share/tessdata`、系统 Tesseract 和 Debian 标准路径；缺少请求的语言数据时会把
页面标记为 parser failed，而不是静默返回空文本。

Section/Chunk ID 使用稳定的文档命名空间：

```text
PDF-006:s0001
PDF-006:s0001:c001
```

PDF metadata title 仅用于展示，不参与 ID。`document_id` 必须匹配
`[A-Za-z0-9][A-Za-z0-9._-]{0,127}`；持久化前会验证 section/chunk 唯一性、所属文档和引用关系。
从旧 pipeline 升级时必须重建 raw/clean pages、sections、chunks、BM25 和向量索引，不能把旧产物与新产物混用。
单独执行 sectioning CLI 时必须传 `--document-id`。

每个根 Trace 会自动附带以下可比较维度：`environment`、`release_version`、
`pipeline_version`、`corpus_version`、`index_version`、`embedding_model`、
`reranker_mode`、`reranker_model`、`prompt_version`。通过对应的 `AUTOFLOW_*` 环境变量设置；
`AUTOFLOW_CORPUS_MANIFEST` 还可以在未显式设置语料版本时生成内容哈希。请求入口应额外传入
`question_type`、`language` 和 `document_type`。

## 长期质量监控与回归闭环

### LangSmith Dataset 与 Experiment

将 EA211 Golden Set 幂等同步为 Dataset，并把现有报告的逐题结果上传为 Experiment：

```bash
uv run autoflow-langsmith-eval \
  ../research/automotive-pdfs/llm-test/ea211/evaluation_cases.json \
  ../research/automotive-pdfs/llm-test/ea211/e2e_run/evaluation_report.json \
  autoflow-ea211-golden --stage reranker
```

命令上传每题的 Hit@1/3/5/10 和 MRR。默认 target 重放本次评测报告；生产检索 target 可通过
`--target package.module:function` 接入。实验名可通过 LangSmith `evaluate()` 支持的参数在代码调用
`run_experiment(..., experiment_prefix="release-0.3.0")` 时指定。

### Bad Case 队列

从报告自动识别空召回、expected rank > 5、reranker fallback、正确结果误降和安全失败：

```bash
uv run autoflow-quality scan evaluation_report.json data/bad_cases.json --release 0.3.0
```

人工归因和关闭可通过 CLI（`key` 可从 `bad_cases.json` 获得）：

```bash
uv run autoflow-quality classify data/bad_cases.json '<key>' RERANKER_ERROR
uv run autoflow-quality resolve data/bad_cases.json '<key>' --fixed-by 0.3.1
uv run autoflow-quality promote data/bad_cases.json evaluation_cases.json
```

只有完成归因的案例才能关闭。确认后的案例必须进入 Golden Set。

写入按 `case_id` 幂等，防止同一回归案例重复追加。

### Health、Dashboard 与告警

线上检索入口用 `RuntimeSampleRecorder` 写入不含正文的 JSONL 样本。字段包括状态、统一错误码、
总耗时、各阶段耗时、候选数、fallback 以及版本/问题维度。生成 Dashboard 数据：

```bash
uv run autoflow-quality dashboard data/runtime.jsonl data/bad_cases.json \
  --baseline-samples data/baseline-7d.jsonl \
  --evaluation-report evaluation_report.json \
  --output data/dashboard.json
```

输出包含 Health、P50/P95/P99、阶段延迟、错误分布、按发布/问题类型/语言/文档类型拆分、
Hit@K/MRR、reranker 正向提升率/误降率、Bad Case 和基线告警。先采集一周
`baseline-7d.jsonl`；默认请求数不足 20 时不告警，之后按 P95 1.5 倍、失败/空召回/fallback
2 倍基线触发。阈值可通过 `AlertPolicy` 调整。

### 发布硬门禁

```bash
uv run pytest -q
uv run ruff check .
uv run autoflow-quality gate candidate-report.json baseline-report.json
```

门禁默认要求 Hit@5 和 MRR 不下降、P95 不超过基线 1.2 倍、没有新增严重 Bad Case，失败返回
非零退出码，可直接接入 CI。chunk、embedding、RRF、reranker、语料或过滤逻辑变化时，应先生成
candidate report，再执行 Dataset Experiment 和此门禁。

### 答案与引用质量

当前仓库尚无答案生成链路，因此没有伪造线上答案指标。已提供
`record_answer_quality()` 接入点：生成器完成 claim 切分和引用支持关系验证后，传入 claims、
citations、refused、evidence_available、tokens、cost，即可计算并记录 Citation Coverage、
Citation Correctness、Groundedness、无证据回答、证据不足拒答、Token 和成本。
