# 汽车售后/维修 PDF 验证报告

## 运行位置

本报告和 PDF 均位于当前项目目录：

```text
research/automotive-pdfs/
```

## 自动验证结果

执行命令：

```bash
uv run --with pymupdf python research/automotive-pdfs/validate_corpus.py
```

结果：

```text
manifest_rows=109
actual_pdf_count=109
unique_sha256=109
errors=0
```

验证内容：

- 清单包含 109 行（原始 100 份 + 9 份中文补充资料）；
- 项目目录实际找到 109 个 `.pdf` 文件；
- 每个清单路径都存在；
- 每个文件以 `%PDF-` 开头；
- SHA-256 与清单一致；
- SHA-256 全部唯一；
- PyMuPDF 能打开每个文件，且页数与清单一致；
- CSV 使用 UTF-8 BOM，Excel 和 Python 均可读取；
- 每行均有 `pdf_id / filename / relative_path / source / page_count / sha256 / validation` 等字段。

## 人工抽查

使用 PyMuPDF 渲染了以下样本的第 1 页和中间页，并人工查看：

| PDF | 人工结果 |
|---|---|
| PDF-001 | 官方 MQB PDF；普通单栏正文，右侧有联系信息侧栏和图片，未见表格或双栏正文 |
| PDF-002 | EA211 SSP；文字层可提取，含技术表格、曲线图和发动机图片，不是扫描件 |
| PDF-008 | SSP 147；正文为可搜索文字加插图，不是扫描件，未见表格、双栏或独立侧栏；已修正自动标签 |
| PDF-009 | SSP 153；普通单栏技术正文，含示意图 |
| PDF-020 | SSP 195；车型/发动机技术资料，文字层正常，图片较多 |
| PDF-050 | SSP 226；机械剖面图为主，文字层正常，图片较多 |
| PDF-080 | SSP 258；车型介绍和图片排版，文字层正常 |
| PDF-100 | SSP 278；车身技术资料，图纸和页眉模块明显，文字层正常 |

抽查图像保存在：

```text
research/automotive-pdfs/inspection/
```

清单新增字段：

```text
manual_review_status
manual_review_notes
```

原始 VAG/VW 100 份已查看首屏联系表和内页联系表；9 份中文补充资料已查看首屏联系表。所有 109 份均在 `relevance_audit.csv` 中标记为“确认相关”，并记录逐份核验依据。未完成深度版式人工核验的补充资料，其版式字段仍明确标记为“待人工抽查”，没有把自动分类伪装成人工确认。

## 分类说明

`document_type / has_tables / is_double_column / has_sidebar` 是版式初筛标签，基于文本块、图片、绘图线和坐标启发式生成；人工抽查结果单独记录。原始 100 份的首屏和内页联系表已人工查看，9 份中文补充资料的首屏已人工查看。当前资料仍以 VAG/VW Self-Study Program 为主体，补充资料来自政府、职业教育、维修技能竞赛和汽车维修技术出版物，因此不能声称覆盖所有汽车厂家或所有版式类型。

## 内容边界

资料覆盖大众/VAG 的平台、发动机、变速箱、诊断、CAN、ESP/ABS、制动、转向、悬架、空调、车身电气和车型技术介绍。SSP 是公开技术培训/原理资料，不等于当前车型的完整维修手册、TPI、诊断树、扭矩表或软件版本资料。导入 Agent 前仍需按车型、年款、动力版本、来源等级和版权许可过滤。
