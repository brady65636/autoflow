# 汽车售后/维修 PDF 资料集

## 验收结果

- PDF 数量：109（原始 100 份 + 9 份中文汽车维修补充资料）
- 原始 100 份总页数：5263；补充资料也已纳入清单和校验
- 可读取文件：109/109
- 每份文件均有来源 URL、SHA-256、页数和相对路径
- 清单：`pdf_manifest.csv`、`pdf_manifest.json`
- PDF 文件目录：`pdfs/`、`framework/` 和 `supplemental/`
- 逐份相关性清单：`relevance_audit.csv`
- 人工视觉核验联系表：`manual-audit/`

## 清单字段

```text
pdf_id
filename
relative_path
title
source
bytes
sha256
validation
page_count
document_type
is_scanned
has_images
has_tables
is_double_column
has_sidebar
notes
```

`document_type`、表格、双栏和侧边栏字段由 PyMuPDF 版式启发式分析生成，适合作为初筛标签；正式做 OCR/解析评估时建议再人工抽查。

## 内容覆盖

- 大众 MQB/MEB 平台
- EA211、EA888 发动机
- DSG/手动/自动变速箱
- OBD、VAS 诊断、CAN 总线
- ESP、ABS、制动辅助、胎压监测
- 电子转向、悬架、四驱/Haldex
- 空调、热泵、冷却系统
- 车身电气、蓄电池、车载网络
- Polo、Golf、Passat、Audi、Touran、Phaeton、Jetta 等车型/系统
- 中国汽车维修技术信息公开目录

## 来源与版权边界

资料主要来自大众/大众中国官方公开页面，以及公开提供 VAG Self-Study Program 的资料站点。SSP 主要用于结构和工作原理学习，不是当前车型维修手册；不能替代厂家的维修工艺、TPI、诊断树、扭矩和软件版本资料。后续导入 Agent 前，应按来源等级、车型年款和版权许可进行筛选。
