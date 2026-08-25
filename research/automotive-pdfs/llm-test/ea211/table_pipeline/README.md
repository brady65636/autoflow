# EA211 Table Pipeline 测试

输入：64 页 EA211 PDF。

PyMuPDF-only baseline:

```text
pages=64
pages_with_tables=20
candidate_count=20
accepted_tables=2
warning_tables=18
failed_tables=0
camelot_available=false
```

Camelot fallback run (`table_pipeline_camelot/`):

```text
pages=64
pages_with_tables=61
candidate_count=153
accepted_tables=2
warning_tables=59
failed_tables=0
camelot_available=true
camelot_candidates=133
camelot_selected=54
```

Camelot increased candidate coverage, but did not increase accepted tables on this
watermarked SSP. Its `accuracy` can be high while cells still contain rotated
Volkswagen copyright fragments, so the project quality gate correctly keeps those
results at `warning` instead of sending them directly to RAG.

当前策略：

1. `pymupdf_lines`：优先处理有边框/线框表格；
2. `pymupdf_text`：处理无边框文字布局表格；
3. 对普通正文造成的误检使用行数、列数、分数和覆盖率过滤；
4. 对保留下来的候选计算一致性、空单元格比例、短碎片比例、原始词覆盖率和水印污染率；
5. `good` 进入统一表格对象；`warning` 保留但不直接作为高置信度 RAG 内容；`failed` 进入 quarantine；
6. Camelot 作为可选 fallback，目前环境未安装，因此报告 `camelot_available=false`。

输出：

- `tables.json`：PyMuPDF-only 选中的表格对象；
- `table_quality_report.json`：PyMuPDF-only 每页候选、分数、解析器和质量状态；
- `table_pipeline_camelot/`：启用 Camelot fallback 后的对比结果；
- `table_pipeline_camelot/table_quality_report.json`：包含 Camelot parser report；
- `table_pipeline_summary.json`：命令摘要。

EA211 这种带大量旋转版权水印和图文混排的 SSP，表格候选多数被标为 `warning`，这是预期的保守行为：宁可隔离不可靠表格，也不把污染数据直接送入 RAG。
