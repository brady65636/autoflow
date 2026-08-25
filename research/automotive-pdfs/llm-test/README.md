# PyMuPDF4LLM 测试记录

## 环境

- 项目：`backend`
- `pymupdf4llm==1.28.2`
- `pymupdf-layout==1.28.2`
- `PyMuPDF==1.28.2`

## 输入

```text
research/automotive-pdfs/framework/01_SSP_511_EA211_petrol_engine.pdf
```

64 页，大众 EA211 发动机 Self-Study Program。

## 运行方式

```python
import pymupdf4llm
chunks = pymupdf4llm.to_markdown(
    pdf,
    page_chunks=True,
    write_images=True,
    image_path=output_dir,
    dpi=120,
)
```

## 实际结果

### Raw 解析

- 成功处理 64 页；
- 生成 64 个 page chunks；
- Markdown 约 151,641 字符；
- 输出 189 个页面图片/图形资源；
- 能提取正文、章节标题、目录和部分技术表格；
- `table_output="html"` 能输出 HTML 表格。

### Post Processing

输出目录：`post_processed/`

- `clean_pages.json`：清洗后的 page chunks；
- `clean_document.md`：清洗后的 Markdown；
- `quality_report.json`：逐页和文档级质量报告；
- 64 页全部处理成功；
- 9 页无 warning，55 页有 warning；
- 标记并移除 2765 个重复/旋转水印碎片；
- 过滤 32 个图片内部短碎片；
- 识别 5 个表格候选，其中 3 个标记为低质量；
- raw JSON 保留在本目录，未被覆盖。

## 观察到的问题

1. 普通正文提取效果较好，例如能正确提取 MQB、EA211、发动机排量和技术说明。
2. 技术规格表可以识别为表格，但页面上的竖排版权水印会混入单元格，造成噪声。
3. 复杂图片页会产生 `Start of picture text`，其中可能包含图片 OCR 或旋转版权文字，部分内容顺序混乱。
4. 图像资源能够导出，但默认 Markdown 使用本地相对图片路径；导入知识库前需要统一资源目录和 URL。
5. 对这种“图文混排 + 水印 + 表格 + 复杂布局”的 SSP，不能直接把全部 Markdown 当作高质量知识，需要清理水印、过滤图片 OCR 噪声并对表格做质量检查。

## 结论

PyMuPDF4LLM 适合做第一阶段 PDF 摄取器：正文、标题、页码、图片和部分表格都能得到结构化输出。对于 AutoFlow 的知识库，建议先使用：

```text
PyMuPDF4LLM → 清理旋转水印/图片 OCR 噪声 → 表格质量检查 → Chunk → Embedding
```

不建议直接把原始输出交给 Agent。
