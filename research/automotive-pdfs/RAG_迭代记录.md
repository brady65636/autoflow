# RAG 数据与算法迭代记录

> 用途：记录每次 RAG 数据处理、索引构建、检索测试、reranker 测试和验收的完整输入、算法、配置、结果与结论。
>
> 规则：只追加，不覆盖历史记录。每个结果必须能对应到明确的数据文件、代码版本和产物路径。

## 记录模板

### RAG-ITER-XXXX

- **时间：** YYYY-MM-DD HH:mm（时区）
- **任务/目的：**
- **运行命令：**
- **运行环境：**
  - Python/依赖：
  - CPU/GPU：
  - 模型及路径：

#### 数据版本

- 原始资料集/manifest：
- 文档数/页数：
- 数据 hash 或 manifest 版本：
- 测评集：
- 样本数：
- 正例/无答案/跨文档/安全拒答：
- 中文/英文：

#### 算法与配置

- Parser/OCR：
- 清洗与 section：
- Chunk：
- Embedding：
- BM25/词法检索：
- RRF：
- Reranker：
- Pipeline version：
- 关键参数：

#### 结果

| 范围 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|
| section | 未测量 | 未测量 | 未测量 | 未测量 | 未测量 |
| document | 未测量 | 未测量 | 未测量 | 未测量 | 未测量 |

- 中文/英文分组：
- 负例误答率：
- 安全越权率：
- 延迟/吞吐：

#### 质量与验收

- replacement character：
- parser failed pages：
- fallback/OCR pages：
- section/chunk 数量：
- 重复/悬空引用：
- SQLite 校验：
- release gate：

#### 产物

- 结果报告：
- 逐条明细：
- 原始/清洗/section/chunk 数据：
- 日志：

#### 结论与下一步

- 本轮变化：
- 已知问题：
- 下一轮计划：

---

## RAG-ITER-0001：20 PDF 数据管线与 lexical baseline

- **时间：** 2026-08-24（文件时间；具体命令时间未完整记录）
- **任务/目的：** 完成 20 份 PDF 的 RAG 摄取、section/chunk 生成、SQLite 导入，并建立 80 条候选 Golden Set 和 lexical retrieval baseline。
- **运行命令：** 历史运行；原始完整命令未保存。测评结果已使用当前代码复现。
- **运行环境：**
  - Python/依赖：后端 `.venv`；相关单元/组件测试通过
  - CPU/GPU：未记录
  - 模型及路径：lexical baseline 不使用 embedding 模型；最终 hybrid/reranker 本轮未运行

### 数据版本

- 原始资料集/manifest：`research/automotive-pdfs/pdf_manifest.json`
- 资料集：109 份清单内 PDF；清单内校验通过
- 额外文件：`research/automotive-pdfs/llm-test/ea211/watermark_redacted_EA211.pdf` 未纳入 manifest，属于实验文件
- RAG 处理范围：PDF-001 至 PDF-020，共 20 份、1125 页
- 测评集：`research/automotive-pdfs/pipeline-20pdf-run/evaluation_cases_80.json`
- 测评集规模：80 条
  - 正例：60
  - 无答案：5
  - 跨文档混淆：5
  - 权威/安全拒答：10
  - 中文：40；英文：40
- 正例覆盖：PDF-001 至 PDF-020，每份 3 条
- 测评集状态：机械校验通过；80/80 AI 复核通过；人工双人标注仍为 pending

### 算法与配置

- Parser：PyMuPDF4LLM Markdown 为主；PyMuPDF plain-text fallback；Tesseract OCR fallback
- 清洗：`backend/src/autoflow_scheduling/knowledge/post_processor.py`
- Section：`backend/src/autoflow_scheduling/knowledge/sectioning.py`
- Chunk：`backend/src/autoflow_scheduling/knowledge/chunking.py`
- Retrieval：`backend/src/autoflow_scheduling/knowledge/retrieval_baseline.py`
- 本轮算法：词法 TF-IDF-like scorer + section expansion；没有使用 embedding、RRF 或 reranker
- Pipeline version：4
- Chunk size：1800
- Chunk overlap：200
- replacement ratio threshold：0.10
- OCR：200 DPI，`eng+chi_sim`
- 评测 top-k：10 个 section
- 每个 section 保留最多 3 个匹配 chunk

### 结果

结果文件：`research/automotive-pdfs/pipeline-20pdf-run/retrieval_60_positive_baseline.json`

本轮只评测 60 条正例，未评测负例和安全拒答。

| 范围 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|
| section | 0.2333 | 0.3667 | 0.4333 | 0.4833 | 0.3076 |
| document | 0.4167 | 0.5833 | 0.6500 | 0.6667 | 0.5124 |

分语言结果：

- section / 中文：Hit@1 0.0000，Hit@5 0.1290，Hit@10 0.2258，MRR 0.0513
- section / 英文：Hit@1 0.4828，Hit@5 0.7586，Hit@10 0.7586，MRR 0.5816
- document / 中文：Hit@1 0.1935，Hit@5 0.4194，Hit@10 0.4194，MRR 0.2903
- document / 英文：Hit@1 0.6552，Hit@5 0.8966，Hit@10 0.9310，MRR 0.7497

指标定义：

- Hit@K：目标 section/document 的排名不大于 K 即命中
- MRR：命中样本的 `1/rank` 平均值，前 10 名外按 0 计
- section rank：目标 section 的排名
- document rank：目标文档首次出现的 section 排名

### 质量与验收

- 结构化产物：1689 sections，1760 chunks
- 重复 section ID：0
- 重复 chunk ID：0
- broken chunk-section refs：0
- empty sections/chunks：0
- SQLite：20 份文档、1689 sections、1760 chunks；hash verified 20/20
- replacement character：12 个 chunk
  - PDF-002：1
  - PDF-017：10
  - PDF-018：1
- parser failed pages：PDF-015 第 41 页 1 页
- fallback/OCR：PDF-008 40 页 OCR；PDF-009 31 页 plain-text fallback；PDF-015 第 41 页 OCR fallback 失败
- release gate：blocked
- 阻断原因：12 个 replacement-character chunks + 1 个 parser failed page

### 产物

- 总体验收：`research/automotive-pdfs/pipeline-20pdf-run/acceptance_report.json`
- 批处理报告：`research/automotive-pdfs/pipeline-20pdf-run/batch_report.json`
- 测评集：`research/automotive-pdfs/pipeline-20pdf-run/evaluation_cases_80.json`
- 正例输入：`research/automotive-pdfs/pipeline-20pdf-run/retrieval_positive_cases.json`
- 检索逐条明细：`research/automotive-pdfs/pipeline-20pdf-run/retrieval_60_positive_baseline.json`
- 结构校验：`research/automotive-pdfs/pipeline-20pdf-run/validation.json`
- 异常清单：`research/automotive-pdfs/pipeline-20pdf-run/extraction_bad_cases.json`
- 每文档产物：`research/automotive-pdfs/pipeline-20pdf-run/PDF-001/` 至 `PDF-020/`

### 结论与下一步

- 20 份文档的摄取、切分、引用和 SQLite 持久化链路已完成。
- 测评集结构已完成，但人工双标和仲裁未完成。
- lexical baseline 明显未达到计划门槛，中文 section 检索尤其弱；该结果不能代表最终 hybrid/reranker 能力。
- 需要先处理 replacement character 和 PDF-015 第 41 页，再运行 embedding + BM25 + RRF + reranker 评测。
- 下一轮必须保存完整运行命令、环境/模型信息，并同时评测 80 条全量数据，包括无答案、跨文档和安全拒答子集。

---

## RAG-ITER-0002：Qwen embedding + BM25 + RRF，最终保留 15 个 chunk

- **时间：** 2026-08-24（本轮实际运行）
- **任务/目的：** 使用本机已有 embedding 环境，验证 dense embedding + BM25 混合召回；严格先保留 15 个 chunk，再由这些 chunk 映射 section。
- **运行环境：**
  - Python：`E:\\python\\python.exe`
  - PyTorch：`2.12.0+cu126`
  - CUDA：可用，12.6
  - GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`
  - Transformers：`4.57.3`
  - rank-bm25：已安装
  - Hugging Face cache：`D:\\hf_cache`
- **运行命令：**

```bash
PYTHONPATH=backend/src HF_HOME=D:/hf_cache TRANSFORMERS_CACHE=D:/hf_cache \
/e/python/python.exe -m autoflow_scheduling.knowledge.hybrid_evaluation \
  research/automotive-pdfs/pipeline-20pdf-run/combined_chunks.json \
  research/automotive-pdfs/pipeline-20pdf-run/combined_sections.json \
  research/automotive-pdfs/pipeline-20pdf-run/retrieval_positive_cases.json \
  research/automotive-pdfs/pipeline-20pdf-run/retrieval_hybrid_embed_bm25_top15.json \
  D:/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 \
  --reranker-mode none --dense-limit 50 --bm25-limit 50 --final-chunk-limit 15
```

### 数据与算法

- 数据：与 RAG-ITER-0001 相同的 1760 chunks、1689 sections、60 条正例
- Embedding：Qwen3-Embedding-0.6B，向量维度 1024
- Dense candidate：top 50
- BM25 candidate：top 50
- RRF：`k=60`
- 最终 chunk：RRF 后严格保留 15 个
- Reranker：未启用
- Section：只从最终 15 个 chunk 映射得到，未先聚合全库 section
- 代码：`backend/src/autoflow_scheduling/knowledge/hybrid_evaluation.py`

### 结果

结果文件：`research/automotive-pdfs/pipeline-20pdf-run/retrieval_hybrid_embed_bm25_top15.json`

| 阶段 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|
| dense | 0.7500 | 0.9167 | 0.9500 | 1.0000 | 0.8481 |
| BM25 | 0.3333 | 0.4833 | 0.5500 | 0.6167 | 0.4279 |
| RRF + 15 chunks | 0.5667 | 0.8333 | 0.8833 | 0.9167 | 0.7008 |

- 测评条数：60 条正例
- 每条最终候选 chunk：15 个
- 每条输出 section：最多 3 个去重 section
- Reranker fallback：0；本轮未启用 reranker
- 负例误答率：未测量
- 安全越权率：未测量
- 延迟：embedding 55.4857 秒；混合检索 0.9852 秒；模型加载 5.8801 秒

### 验证与结论

- 窄测试：`3 passed`
- 报告参数确认：`final_chunk_limit=15`
- 报告候选数量：60/60 条均为 15 个 chunk
- 报告最终阶段：60/60 条为 `rrf`
- 本轮证明 embedding 召回明显优于旧 lexical baseline；RRF 在当前参数下低于 dense 单路，但仍显著高于 BM25 单路。
- 本轮只覆盖 60 条正例，不能作为完整 80 条 Golden Set 发布验收结果；负例、安全拒答和人工双标仍未完成。
- 语言分组详细报告：`research/automotive-pdfs/pipeline-20pdf-run/retrieval_hybrid_embed_bm25_top15_by_language.md`
- 语言分组 JSON 明细：`research/automotive-pdfs/pipeline-20pdf-run/retrieval_hybrid_embed_bm25_top15_by_language.json`
- 中文结果：dense MRR 0.8215，BM25 MRR 0.1401，RRF MRR 0.5850；英文结果：dense MRR 0.8764，BM25 MRR 0.7355，RRF MRR 0.8245。

---

## RAG-ITER-0003：全英文测评集上的 RRF 权重对比

- **时间：** 2026-08-24（本轮实际运行）
- **任务/目的：** 在全英文 60 条正例上比较 Dense:BM25 三组 RRF 权重。
- **固定配置：** Dense top-50、BM25 top-50、RRF 后保留 15 个 chunk、未启用 reranker、Qwen3-Embedding-0.6B、RRF `k=60`。
- **新增算法参数：** `dense_weight`、`bm25_weight`，写入 `hybrid_evaluation.py` 和每份 JSON 报告。

### 对比结果

| Dense:BM25 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|
| 1:1 | 0.7167 | 0.9667 | 0.9833 | 0.9833 | 0.8352 |
| 2:1 | 0.7833 | 0.9667 | 0.9833 | 1.0000 | 0.8719 |
| 3:1 | 0.7833 | 0.9833 | 1.0000 | 1.0000 | 0.8783 |

### 产物

- 1:1：`research/automotive-pdfs/pipeline-20pdf-run/retrieval_hybrid_rrf_1to1_top15_all_english.json`
- 2:1：`research/automotive-pdfs/pipeline-20pdf-run/retrieval_hybrid_rrf_2to1_top15_all_english.json`
- 3:1：`research/automotive-pdfs/pipeline-20pdf-run/retrieval_hybrid_rrf_3to1_top15_all_english.json`
- 对比 JSON：`research/automotive-pdfs/pipeline-20pdf-run/retrieval_hybrid_rrf_weight_comparison.json`
- 对比 Markdown：`research/automotive-pdfs/pipeline-20pdf-run/retrieval_hybrid_rrf_weight_comparison.md`

### 结论

- 三组中 `3:1` 最好：MRR 0.8783，Hit@3/5/10 为 0.9833/1.0000/1.0000。
- `3:1` 相比 `1:1`：Hit@1 +0.0666，Hit@5 +0.0167，Hit@10 +0.0167，MRR +0.0431。
- 本结论只覆盖正例召回；负例、安全拒答和 reranker 尚未纳入。

### RAG-ITER-0004：更高 Dense 权重对比

- **固定配置：** 全英文 60 条正例、Dense top-50、BM25 top-50、RRF 后 15 chunks、`k=60`、未启用 reranker。

| Dense:BM25 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|
| 5:1 | 0.7667 | 0.9833 | 1.0000 | 1.0000 | 0.8756 |
| 10:1 | 0.7833 | 0.9667 | 1.0000 | 1.0000 | 0.8825 |
| 20:1 | 0.8000 | 0.9667 | 1.0000 | 1.0000 | 0.8881 |

- 5:1 报告：`research/automotive-pdfs/pipeline-20pdf-run/retrieval_hybrid_rrf_5to1_top15_all_english.json`
- 10:1 报告：`research/automotive-pdfs/pipeline-20pdf-run/retrieval_hybrid_rrf_10to1_top15_all_english.json`
- 20:1 报告：`research/automotive-pdfs/pipeline-20pdf-run/retrieval_hybrid_rrf_20to1_top15_all_english.json`
- 六组汇总：`research/automotive-pdfs/pipeline-20pdf-run/retrieval_hybrid_rrf_weight_comparison.md`
- 当前六组中 `20:1` 的 MRR 最高，为 0.8881；继续增大 Dense 权重的效果尚未验证。

### RAG-ITER-0005：RRF 20:1 + Qwen reranker + 业务规则

- **固定召回：** Dense:BM25 = 20:1，Dense/BM25 各 top-50，RRF 后保留 15 个 chunk。
- **Reranker：** `Qwen3-Reranker-0.6B`，15 个候选全部进入精排。
- **业务规则：** 复用 `retrieval_profile.py` 的 `QuestionType × DocumentContentType` 兼容矩阵；`PRIMARY=1.0`、`SUPPORTING=0.5`、`NONE=0.0`，再结合 metadata confidence 和 rag text quality。
- **融合公式：** `final = 0.8 × model_score + 0.2 × business_rule_score`。
- **结果文件：** `research/automotive-pdfs/pipeline-20pdf-run/retrieval_rrf20_rerank_business_top15_all_english.json`

| 阶段 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|
| Dense | 0.7833 | 0.9500 | 0.9833 | 1.0000 | 0.8769 |
| BM25 | 0.6500 | 0.7833 | 0.9167 | 0.9667 | 0.7536 |
| RRF 20:1 | 0.8000 | 0.9667 | 1.0000 | 1.0000 | 0.8881 |
| RRF + reranker + rules | **0.8333** | **0.9833** | 1.0000 | 1.0000 | **0.9125** |

- Reranker errors/fallback：0/60
- 最终候选 chunk：60/60 条均为 15 个
- 最终阶段：60/60 条为 `reranker`
- 相比 RRF 20:1：Hit@1 +0.0333，Hit@3 +0.0166，MRR +0.0244。

### RAG-ITER-0006：持久化 chunk embedding 到 SQLite

- **数据库：** `backend/autoflow.db`
- **输入：** `research/automotive-pdfs/pipeline-20pdf-run/combined_chunks.json`
- **模型：** `Qwen3-Embedding-0.6B`
- **存储格式：** little-endian float32 BLOB
- **写入结果：** 1760/1760 chunks，1024 维，单向量 4096 bytes
- **字段：** `knowledge_chunks.embedding`、`embedding_model`、`embedding_dimension`
- **验证结果：** 所有向量非空、维度统一、模型名统一；后端测试 `103 passed`
- **实现：** `backend/src/autoflow_scheduling/knowledge/embedding_store.py`

### RAG-ITER-0007：Agent RAG smoke dataset 与真实检索验证

- **时间：** 2026-08-25 10:27（本机时间）
- **任务/目的：** 验证现有 60 条正例是否可以复用为 Agent 的 `retrieve_knowledge` 测试，并补充无答案、跨文档混淆和安全拒答样例。
- **数据集：** `research/automotive-pdfs/pipeline-20pdf-run/evaluation_cases_80.json`
- **新增 Agent smoke 集：** `research/automotive-pdfs/pipeline-20pdf-run/agent_rag_smoke_cases.json`
- **样本数：** 8 条，其中正例 5 条、无答案 1 条、跨文档混淆 1 条、安全拒答 1 条。
- **语言：** 英文 8/8。
- **复用规则：** 正例的 `query`、`question_type`、`expected_document_id`、`expected_section_id` 直接复用；负例的 `expected_behavior` 用于 Agent 最终回答级别的拒答/安全判断。

#### 算法与环境

- **检索实现：** `backend/src/autoflow_scheduling/knowledge/retrieval_service.py`
- **Embedding：** Qwen3-Embedding-0.6B，本地 CUDA 推理。
- **BM25：** rank-bm25。
- **融合：** Dense top-50 + BM25 top-50，RRF，Dense:BM25=20:1，`k=60`。
- **Reranker：** Qwen3-Reranker-0.6B，候选 15 条。
- **业务规则：** `business_rule_weight=0.2`。
- **运行环境：** `E:/python/python.exe`，PyTorch 2.12.0+cu126，NVIDIA GeForce RTX 4060 Laptop GPU。

#### 结果

- **正例 document Hit@15：** 1.0000（5/5）
- **正例 section Hit@15：** 1.0000（5/5）
- **正例 section Hit@1：** 0.8000（4/5）
- **负例误答率：** 未测量；本轮只完成检索层，未运行 Agent 最终回答拒答评测。
- **安全越权率：** 未测量。
- **延迟/吞吐：** 未测量。

#### 观察与结论

- 现有 60 条正例可以直接复用为 Agent RAG 的检索测试基线；Agent 工具额外只需要把结果压缩成证据和来源。
- 5 条 smoke 正例全部召回目标文档和目标 section；其中 1 条目标 section 排名第 2，不能只用 Hit@1 判断。
- 部分查询会带出相关但非目标文档，Agent 必须依据证据内容和文档置信度回答，不能只看到“有检索结果”就作答。
- 负向/安全用例需要接入 DeepSeek Agent Loop，检查拒答和不跨文档推断；不能用纯检索命中率代替。

#### 产物

- 测试集：`research/automotive-pdfs/pipeline-20pdf-run/agent_rag_smoke_cases.json`
- 检索报告：`research/automotive-pdfs/pipeline-20pdf-run/agent_rag_smoke_report.json`
- 代码：`backend/src/autoflow_scheduling/knowledge/retrieval_service.py`、`backend/src/autoflow_scheduling/agent/tools.py`

### RAG-ITER-0008：DeepSeek Agent RAG-调度-工单端到端验证

- **时间：** 2026-08-25（本机时间）
- **任务/目的：** 在临时 SQLite 数据库中模拟真实用户，验证“咨询、资料检索、原因辅助判断、profile-only 调度、用户确认、工单创建、状态查询”的完整链路。
- **运行环境：** `E:/python/python.exe`，PyTorch 2.12.0+cu126，NVIDIA GeForce RTX 4060 Laptop GPU。
- **模型：** DeepSeek V4 Flash，thinking disabled；Qwen3-Embedding-0.6B；Qwen3-Reranker-0.6B。
- **数据库：** 从 `backend/autoflow.db` 复制的临时 SQLite；正式数据库未写入。

#### 测试流程与结果

- 咨询：发动机故障灯亮、怠速抖动。
- RAG：成功返回失火、MIL 闪烁/常亮和故障码读取相关证据；模型第一次传入中文问题类型和错误参数，工具返回结构化校验错误后模型自行修正。
- 调度：使用 `store_id + brand + vehicle_profile + ISO 时间窗口`，最终返回 `FEASIBLE`，时间为 `2026-08-29 16:00-17:00`。
- 确认：用户明确提供确认令牌后，成功创建 ConfirmedPlan。
- 工单：成功创建 `CONFIRMED` 工单，并写入临时用户的 `customer_user_id`。
- 查询：Agent 使用 `status=CONFIRMED`，按用户 ID 查询到刚创建的工单。

#### 验收状态

- **业务链路：** 通过。
- **正式数据库污染：** 未发生。
- **RAG Hit@K：** 未测量；本轮是 Agent 端到端 smoke，不是批量检索评测。
- **负例误答率：** 未测量。
- **安全越权率：** 未测量。
- **已知问题：** DeepSeek 初次工具参数不稳定，依赖 Pydantic 工具错误反馈进行自修正；RAG 某次 reranker 调用出现本地模型 meta tensor 临时错误，后续调用成功。
- **迁移修复：** 发现并修复旧 SQLite `work_orders.vehicle_id NOT NULL` 外键字段导致的 profile-only 工单创建失败，并补充旧资源记录 NULL 时间的迁移兼容。
