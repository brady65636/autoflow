# Table Candidate Ground Truth

## Scope

人工查看了 `redacted_gated_tables/table_quality_report.json` 中全部 27 个 candidate crop。

- 疑似表格页：21
- candidate 区域：27
- 人工确认真实表格 candidate：11
- 人工确认误检：16
- candidate-level precision：11/27 = 40.7%
- 人工确认真实表格所在页：9 页

## 重要统计修正

之前的：

```text
good=2
warning=19
failed=0
```

是按 21 个候选页的 selected result 统计，不是 27 个 candidate 的逐个统计，因此不能与 27 相加。

本次清单区分：

```text
Page level：21 个疑似表格页
Candidate level：27 个表格候选区域
Ground truth：11 个真实表格候选，16 个 false positive
```

## 误检类型

16 个误检主要来自：

- 图文混排页面；
- 发动机示意图和曲线图；
- 图片 caption/标签；
- 普通正文的多列布局；
- 流程/结构示意图。

## 真实表格中的解析质量

11 个真实表格 candidate 中：

- Good：5
- Partial：6
- Bad：0

注意：同一个真实表格可能被 `camelot_lattice` 和 `camelot_stream` 重复生成多个 candidate，因此 candidate 数不能直接当作真实表格数量。

## 文件

- `table_ground_truth.csv`：逐 candidate 标注；
- `contact_sheet_1.jpg`～`contact_sheet_3.jpg`：全部候选区域人工审核图；
- `crops/`：每个 candidate 的裁剪图。
