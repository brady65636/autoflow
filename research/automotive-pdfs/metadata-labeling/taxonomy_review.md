# 汽车售后 RAG 文档分类与排序依据：独立审查

## 结论摘要

1. 当前清单能证明“文件存在、可读、与汽车维修主题相关”，不能证明“文件对某一车型/年款/故障适用”，更不能证明“可作为当前维修操作依据”。`document_type`、`has_tables`、`is_double_column` 等字段主要是 PyMuPDF 版式启发式初筛；不应直接充当语义分类或权威等级。
2. `applies_to` 必须与 `incidental_mentions` 分离。标题、目录或背景段提到“汽车/发动机/新能源”不等于该段证据适用于用户车辆、系统、故障或维修任务。适用性应是带范围和证据定位的关系，而不是一个无条件布尔值。
3. 不建议全局固定 `ranking_tier`。应保留稳定的来源/能力标签，但在 query intent 下用动态 capability/authority 矩阵排序；同一 SSP 对“工作原理”可很强，对“当前车型扭矩/软件/维修步骤”应被降权或硬排除。
4. SSP 的准确定位是“厂商服务培训、自学课程的设计与功能说明（截至编制时）”，不是当前车型维修手册、TPI、诊断树、扭矩表或软件版本资料。厂商署名提高来源真实性，不自动把其所有内容提升为当前维修权威。
5. 所有影响答案的判断都应记录：证据片段、PDF 页码/章节、证据类型、范围、时间/版本、来源链、置信度和反证/限制。

## 审查范围与抽查证据

已读取 `pdf_manifest.csv`、`README_100_PDF资料集.md`、`VERIFICATION_REPORT.md`，并抽查以下 16 份不同内容/形态文件（页码为 PDF 页码；“文本抽取”不能替代视觉复核）：

| PDF | 类型与观察 | 对分类/排序的含义 |
|---|---|---|
| PDF-001 p1 | VW Newsroom `Media Information`，MQB 十周年新闻稿；正文是企业事实/宣传叙述 | 官方发布的权威范围是企业声明、平台历史和产品叙述，不是维修程序 |
| PDF-002 p1-2 | VW `Service Training / Self-Study Programme 511`；明确写“content will not be updated”，测试/设置/维修应查 service literature | SSP 可支持设计原理，维修意图必须降权/排除 |
| PDF-003 p1-2 | Audi eSelf-Study Program 920243，注明 2013/12、应检查更新的技术公告和最新维修资料 | 厂商/日期/修订号应单独建模，不能只给一个 tier |
| PDF-004 p2 | SSP 374，明确“not a workshop manual”，数值仅为当时软件状态的理解辅助 | 是最直接的 applies_to 边界证据 |
| PDF-006 p2 | Touran 电气 SSP，说明网络、控制单元、诊断系统的设计/功能 | “原理适用”与“现车维修适用”应拆开 |
| PDF-007 p2 | Jetta 2019 SSP，说明实际测试、调整、维修应看 After-Sales Service documentation | 车型命中也不能绕过文档用途限制 |
| PDF-026 p1-2 | SSP 202 VAS 5051，描述诊断设备和引导式故障查找 | 可排“诊断设备/原理”，不能据此声称给出现车型诊断流程 |
| PDF-101 p1-2 | 长治技能大赛汽车维修技术文件；依据国家职业技能标准，含理论/操作竞赛与计分 | 对竞赛评分/能力要求适用，不是 OEM 车型维修证据 |
| PDF-102 p1-2 | 汽车运用与维修（含智能新能源汽车）培训/考核站设备工具清单，目录有多类实训设备表 | 对设备配置/培训设施适用；设备“出现于清单”不等于维修步骤或车型适用 |
| PDF-103 p108、215 | 文件名标为 auto repair training plan，但正文抽取显示“机电一体化技术专业人才培养方案”、代码 560301 | 标题与正文疑似错配；`applies_to` 只能低置信或待复核，不能因文件名自动纳入汽车维修答案 |
| PDF-104 p1 | 福建理工学校 2025 汽车运用与维修专业人才培养方案，含课程、学时、实习与毕业标准 | 对课程/培养目标适用，非具体车辆维修权威；时间和机构范围很重要 |
| PDF-105 p3、11、20 | 新疆汽车维修竞赛技术文件；以国家职业技能标准和竞赛模块为依据，含评分点与安全要求 | 对竞赛任务、安全/考核规则适用；不能泛化为所有车型工艺 |
| PDF-106 p86 | 《农机使用与维修》文章，标题为汽车发动机故障诊断技巧与维修经验，含 DOI、作者和摘要 | 二手经验/方法论；需按文献身份和车型缺失降权 |
| PDF-107 p94-95 | 同刊汽车电气系统典型故障诊断文章，讨论直观观察、设备检测等通用方法 | 可作背景或候选诊断思路，不可替代厂商流程和实测数据 |
| PDF-108 p173-174 | 新能源汽车维修与故障诊断技术文章，谈三电、高压和检测设备 | 可作新能源通识/安全提示；需核验出版物、日期、标准引用，不能当车型规范 |
| PDF-109 p85 | 电子诊断技术新能源汽车研究，作者/机构/期刊信息可抽取但来源和版本仍需核验 | 学术/行业观点与可执行维修证据必须分层 |

上述抽查也支持验证报告的限制：109 份路径、哈希和页数验证通过，但补充 9 份主要完成首屏相关性核验；其深层版式、OCR 和语义范围不能假定已人工确认。

## `applies_to` 与 `incidental_mentions` 的分离

建议把文档（或 chunk）与实体/任务的关系至少建成两类，禁止用关键词命中替代关系判断：

- `applies_to`：证据明确针对该对象或任务。例如 PDF-002 的 EA211 设计/功能、PDF-007 的 Jetta 2019 车型介绍、PDF-105 的竞赛模块。必须带 `scope_type`（车型/年款/动力/系统/故障/设备/课程/竞赛）、`scope_value`、`evidence_locator`、`valid_time` 和 `confidence`。
- `incidental_mentions`：只在背景、对比、目录、引用、宣传或设备举例中出现；例如 PDF-001 提到 MQB/MEB/SSP，不能因此对某辆车的维修适用；PDF-102 的设备类别出现发动机/底盘，也不能推出某车型故障步骤。该关系可帮助召回，但默认不得贡献 authority，也不得单独触发答案。

同一文档可以同时有两种关系（例如一篇通用文章实际讨论发动机故障，同时在引言泛提新能源）。建议 chunk 级标注而非仅文档级标注；最终使用 `applicability = direct | contextual | incidental | contradicted | unknown`，而不是只保留一个“相关/不相关”。标题与正文冲突时，以正文证据和人工复核为准，并记录冲突；PDF-103 是必须进入冲突队列的样例。

## 哪些维度硬过滤，哪些只做 rerank

### 可硬过滤（满足明确条件才进入候选集）

- **安全与任务禁用**：用户要求当前维修步骤、扭矩、间隙、软件编码、召回/TPI 或高压操作时，缺少相应 OEM/官方售后资料的 SSP、培养方案、竞赛文件、泛学术文章不得作为执行依据；可作为背景但应隔离。
- **实体范围**：用户明确给出品牌、车型、年款、动力、系统或市场时，`applies_to` 不匹配、仅 `incidental_mentions`、或范围未知的资料硬过滤出操作性答案候选；跨车型/跨代资料不能因零件名相同而放行。
- **文档用途**：`document_purpose` 与 intent 不相容时过滤。例如设备清单不能回答故障排除步骤，培养方案不能回答扭矩，竞赛评分表不能回答 OEM 维修时序，新闻稿不能回答诊断值。
- **证据完整性**：文件打不开、页码/片段无法定位、OCR 质量不足以支持关键数值、来源/版本不可追溯，不能进入高风险答案集合。
- **时间/版本/替代关系**：用户问“当前/最新”时，过期 SSP 或未核验发布日期的资料硬排除，除非明确标为历史背景；若资料声明“content will not be updated”，不得宣称当前有效。
- **语言/辖区/版权与访问政策**：若任务限定中国标准、某市场或可公开引用，辖区不符、版权许可不明或无法展示证据的资料应过滤或转为仅内部检索。

### 只宜 rerank 后软排序

- 来源机构、文档形式、作者资历、是否有 DOI、页面数量、表格/图片/双栏、文本抽取质量、术语匹配、发布日期新旧（在没有明确“当前”要求时）都不应单独硬过滤。
- `has_tables` 可提高“规格/设备/评分表”候选的排序，但不能证明表格正确；`is_scanned` 是文字层检测，不是 OCR 正确率。
- 语义相似度只能产生候选，不能压过范围不匹配、用途冲突或 SSP 的明确免责声明。
- 多个独立来源一致可加分，但“都来自同一转载链”不能算独立印证；应记录 source lineage。

## 取消全局 `ranking_tier`：采用动态 capability/authority 矩阵

建议取消一个跨 query 固定的 `ranking_tier`（或把它降级为不可用于最终排序的静态来源画像）。原因是“权威”不是单轴属性，而是“对哪个问题、哪个范围、用于什么动作”的条件属性：

| query intent | 首选 capability | authority 判断 | SSP 的位置 |
|---|---|---|---|
| 设计/工作原理/系统关系 | OEM training、官方技术说明 | 厂商署名、对象/版本匹配 | 高 capability；仍标历史时点 |
| 当前车型维修步骤/扭矩/调整 | 当前 OEM service literature、TPI、官方维修资料 | 车型年款、修订、市场和程序直接匹配 | 通常排除或仅背景 |
| 故障诊断流程/测量值/软件 | 当前官方诊断/维修资料 | 流程、DTC、软件状态、测量条件直接证据 | 只能提供原理，不能替代 |
| 竞赛任务/评分/职业能力 | 竞赛组委会/国家职业标准 | 赛事、年份、模块范围匹配 | 低 capability |
| 培养目标/课程/学时 | 学校或教育主管机构方案 | 学校、专业代码、年份 | 低维修 authority，高课程 capability |
| 设备采购/实训站配置 | 设备清单/评价组织 | 清单版本、适用专业、配置表 | 非维修 capability |
| 通用背景/研究综述 | 有可核验出版信息的学术/行业文章 | 作者、期刊、DOI、方法和引用质量 | 视内容而定，不能升级为 OEM 规范 |
| 新闻/平台历史/产品事实 | 官方新闻稿 | 新闻事实的发布者权威 | 非维修 capability |

实现上可保留 `source_profile`（如 `manufacturer_training`, `official_news`, `competition_standard`, `curriculum`, `equipment_list`, `academic_article`）和静态 `capability` 向量；查询先分类为 intent，再按矩阵选择硬约束和权重。例如：

```text
eligible = scope_match AND purpose_allowed AND evidence_sufficient
score = semantic_relevance
      + intent_capability(intent, source_profile)
      + authority_for(intent, scope, time)
      + evidence_quality
      - stale_or_disclaimer_penalty
      - incidental_only_penalty
```

若产品仍需输出 tier，建议改名为 query-conditioned `retrieval_band`，在每次检索结果中记录 `intent_used`、矩阵版本和降权原因，避免把“tier 2”误读成全局真理。排序也应保留多样性：在直接证据不足时，可展示“原理背景”和“当前维修资料缺口”，而不是用低权威文档填满结果。

## 证据与置信度记录规范

每个会影响过滤或排序的标签建议使用如下最小记录：

```yaml
claim: "该段适用于 EA211 设计与功能，不是当前维修步骤"
relation: applies_to | incidental_mentions | contradicted | unknown
scope: { brand: VW, model: null, year: null, system: engine, task: design_function }
evidence: [{ pdf_id: PDF-002, page: 2, section: "Important note", quote: "..." }]
source_profile: manufacturer_training
published_or_revision: unknown_or_value
validity: historical_at_creation | current_verified | unknown
confidence: high | medium | low
confidence_basis: direct_text | metadata_only | OCR_or_layout_uncertain | title_body_conflict
reviewed_by: human | rule | model
reviewed_at: YYYY-MM-DD
limitations: "当前测试、设置和维修应查最新 service literature"
```

置信度不是“权威等级”的替代品：`high` 表示该关系/文字在指定页码上确实存在，不表示内容对当前车辆一定正确。建议规则：直接、清晰、可定位的正文范围和用途声明为 high；仅标题/目录或自动抽取为 medium/low；扫描、空文本、版式错乱、标题正文冲突必须降级并进入人工复核。关键安全/数值答案要求两层证据：范围证据 + 具体操作/数值证据，不能只凭摘要。

## 现有受控枚举的漏洞与建议

当前 `document_type` 的大量值为“普通文本”“图片较多”“中文维修/培训补充”等，混合了版式、语言、来源和语义；`validation` 只表示技术可读性，`relevance_status` 只表示主题相关。缺少以下受控维度：

- `source_profile`：OEM service / OEM training / official_news / government_standard / competition_rule / curriculum / equipment_list / academic_article / commercial_secondary / unknown；并单独记录 `issuer`、`source_url`、`source_lineage`。
- `document_purpose`：repair_procedure、diagnosis、design_function、product_fact、competition_assessment、curriculum、equipment_spec、research_review 等。
- `applicability_relation` 与 `scope_type/value`：禁止把 `relevance_status=确认相关` 当作适用性。
- `incidental_mentions`：实体/术语出现但非主张对象的可审计关系。
- `authority_by_intent` 或 capability 向量，而不是单一 `ranking_tier`。
- `valid_time`、`publication_date`、`revision`、`superseded_by`、`software_status`、`market/jurisdiction`。
- `evidence_locator`、`evidence_kind`（正文/表格/图/目录/元数据）、`quote_hash`、`extraction_quality`、`human_review_scope`、`confidence`。
- `safety_class`（普通背景/维修操作/高压或安全关键）与 `rights/access_policy`。
- 质量字段应拆分：文字层、OCR 质量、表格抽取质量、版式标签置信度；不能以 `is_scanned=否` 推导文本正确。

迁移时不必立刻删除旧字段，但应把旧 `document_type` 标为 legacy/layout_hint，并禁止其直接决定权威排序。首先为 109 份补齐 `source_profile`、`document_purpose`、`applicability_relation`、`evidence/confidence`；优先复核 PDF-103 的标题/正文冲突，以及所有需要支持当前维修和高压安全的资料。

## 最终判定

该资料集适合作为汽车售后 RAG 的**分层候选语料和评估集**，不适合作为未经条件过滤的统一维修知识库。最重要的控制点不是给每份 PDF 排一个永久 tier，而是：先按 query intent、实体范围、用途和安全等级硬过滤，再在 capability/authority 矩阵内软排序，并在回答中显式区分“直接适用证据”“背景提及”和“当前资料缺口”。
