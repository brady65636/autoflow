import pymupdf
import math


pdf_path = r"C:\Users\LX\Desktop\codebase\autoflow\01_SSP_511_EA211_petrol_engine.pdf"

doc = pymupdf.open(pdf_path)

page_number = 1
page = doc[page_number - 1]

data = page.get_text("dict")

elements = []


# ==========================
# 1. 提取所有文字元素
# ==========================

for block in data["blocks"]:

    if block["type"] != 0:
        continue

    for line in block["lines"]:

        dx, dy = line["dir"]

        angle = math.degrees(
            math.atan2(dy, dx)
        )

        text = "".join(
            span["text"]
            for span in line["spans"]
        ).strip()

        if not text:
            continue

        x0, y0, x1, y1 = line["bbox"]

        elements.append({
            "text": text,
            "bbox": (x0, y0, x1, y1),
            "angle": angle
        })


# ==========================
# 2. 找明显旋转文字
# ==========================

rotated_elements = [
    e
    for e in elements
    if abs(e["angle"]) > 8
]


print("所有文字元素:", len(elements))
print("明显旋转文字:", len(rotated_elements))


# ==========================
# 3. 计算旋转文字整体区域
# ==========================

watermark_region = None

# 大量旋转文字通常意味着水印/特殊装饰文字
if len(rotated_elements) >= 20:

    xs0 = [e["bbox"][0] for e in rotated_elements]
    ys0 = [e["bbox"][1] for e in rotated_elements]
    xs1 = [e["bbox"][2] for e in rotated_elements]
    ys1 = [e["bbox"][3] for e in rotated_elements]

    watermark_region = (
        min(xs0),
        min(ys0),
        max(xs1),
        max(ys1)
    )

    print("检测到疑似水印区域:")
    print(watermark_region)


# ==========================
# 判断一个文字中心是否在区域内
# ==========================

def inside_region(element, region):

    if region is None:
        return False

    x0, y0, x1, y1 = element["bbox"]

    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2

    rx0, ry0, rx1, ry1 = region

    return (
        rx0 <= center_x <= rx1
        and
        ry0 <= center_y <= ry1
    )


# ==========================
# 4. 清理
# ==========================

clean_elements = []

for e in elements:

    angle = e["angle"]
    text = e["text"].strip()

    # ==========================
    # 规则1：明显旋转文字
    # ==========================
    if abs(angle) > 8:
        continue

    # ==========================
    # 规则2：位于水印区域中的短碎片
    # 例如 olk / sw / ag
    # ==========================
    if inside_region(e, watermark_region):

        if len(text) <= 5:
            continue

    # 其他保留
    clean_elements.append(e)


print("\n========== 清理后的正文 ==========\n")

for e in clean_elements:
    print(e["text"])

# ==========================
# 5. 打印剩余内容
# ==========================

print("\n========== 清理后的正文 ==========\n")

for e in clean_elements:
    print(e["text"])