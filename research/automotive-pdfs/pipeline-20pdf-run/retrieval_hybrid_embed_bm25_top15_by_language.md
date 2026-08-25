# Qwen Embedding + BM25 + RRF（15 Chunk）语言分组测评记录

> 来源报告：`research/automotive-pdfs/pipeline-20pdf-run/retrieval_hybrid_embed_bm25_top15.json`

- 评测集：60 条正例（中文 31，英文 29）
- Dense candidate：50
- BM25 candidate：50
- RRF 后最终 chunk：15
- Reranker：未启用
- RRF：`k=60`

## 指标说明

- `rank` 是正确 section 在对应 chunk 排序中的首次命中位置；未命中记为 `—`。
- Hit@K：rank ≤ K 的比例。
- MRR：所有样本的 `1/rank` 平均值，未命中计 0。

## 汇总

| 语言 | 阶段 | 样本 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|
| 中文 | dense | 31 | 0.7097 | 0.9032 | 0.9355 | 1.0000 | 0.8215 |
| 中文 | bm25 | 31 | 0.0645 | 0.1935 | 0.2258 | 0.2903 | 0.1401 |
| 中文 | rrf | 31 | 0.4516 | 0.7097 | 0.8065 | 0.8710 | 0.5850 |
| 英文 | dense | 29 | 0.7931 | 0.9310 | 0.9655 | 1.0000 | 0.8764 |
| 英文 | bm25 | 29 | 0.6207 | 0.7931 | 0.8966 | 0.9655 | 0.7355 |
| 英文 | rrf | 29 | 0.6897 | 0.9655 | 0.9655 | 0.9655 | 0.8245 |

## 逐题明细

### 中文（31 条）

| case_id | query | dense rank | BM25 rank | RRF rank | top sections（RRF 后） |
|---|---|---:|---:|---:|---|
| `p001_005_pos_pdf001_01` | MQB平台截至2022年已经生产了多少辆汽车？ | 1 | 5 | 2 | PDF-001:s0003 (PDF-001:s0003:c001); PDF-001:s0002 (PDF-001:s0002:c001); PDF-001:s0004 (PDF-001:s0004:c003) |
| `p001_005_pos_pdf001_03` | MQB允许在同一生产线上制造哪些差异化产品？ | 1 | 3 | 1 | PDF-001:s0004 (PDF-001:s0004:c001); PDF-001:s0002 (PDF-001:s0002:c002); PDF-001:s0005 (PDF-001:s0005:c001) |
| `p001_005_pos_pdf002_02` | EA211的1.2升和1.4升TSI高压燃油系统压力范围分别是多少？ | 6 | 16 | 7 | PDF-002:s0009 (PDF-002:s0009:c001); PDF-002:s0011 (PDF-002:s0011:c001); PDF-007:s0031 (PDF-007:s0031:c001) |
| `p001_005_pos_pdf003_01` | 第三代EA888为何在排气凸轮轴上采用凸轮轴调节器？ | 10 | — | — | PDF-003:s0004 (PDF-003:s0004:c001); PDF-003:s0003 (PDF-003:s0003:c001); PDF-003:s0005 (PDF-003:s0005:c001) |
| `p001_005_pos_pdf003_03` | 第三代EA888的单涡管涡轮增压器在高转速全负荷时带来什么响应特点？ | 1 | 11 | 3 | PDF-003:s0004 (PDF-003:s0004:c001); PDF-003:s0005 (PDF-003:s0005:c001); PDF-003:s0093 (PDF-003:s0093:c001) |
| `p001_005_pos_pdf004_02` | ESP可以通过哪些系统干预来稳定车辆？ | 1 | 3 | 2 | PDF-004:s0043 (PDF-004:s0043:c002); PDF-004:s0044 (PDF-004:s0044:c001); PDF-004:s0007 (PDF-004:s0007:c002) |
| `p001_005_pos_pdf005_01` | 热泵回路由哪些主要部件组成？ | 1 | — | 1 | PDF-005:s0013 (PDF-005:s0013:c001); PDF-005:s0042 (PDF-005:s0042:c001); PDF-005:s0028 (PDF-005:s0028:c001) |
| `p001_005_pos_pdf005_03` | 热泵系统的维护资格和高压安全边界是什么？ | 1 | — | 1 | PDF-005:s0049 (PDF-005:s0049:c001); PDF-008:s0001 (PDF-008:s0001:c044); PDF-005:s0040 (PDF-005:s0040:c001) |
| `p006_010_003` | 熄火后如何让雨刷进入维修和冬季位置？ | 1 | — | 1 | PDF-006:s0071 (PDF-006:s0071:c001); PDF-008:s0001 (PDF-008:s0001:c044); PDF-006:s0076 (PDF-006:s0076:c001) |
| `p006_010_006` | 前雷达传感器在哪里进行静态标定？ | 1 | — | 1 | PDF-007:s0063 (PDF-007:s0063:c001); PDF-007:s0062 (PDF-007:s0062:c001); PDF-007:s0056 (PDF-007:s0056:c001) |
| `p006_010_009` | 1.9 升直喷柴油发动机的燃烧室为何采用两阶段喷射？ | 1 | 40 | 5 | PDF-003:s0124 (PDF-003:s0124:c001); PDF-017:s0044 (PDF-017:s0044:c001); PDF-017:s0043 (PDF-017:s0043:c001) |
| `p006_010_012` | OBD-II 的 readiness code 表示什么？ | 2 | 3 | 4 | PDF-010:s0101 (PDF-010:s0101:c001); PDF-010:s0102 (PDF-010:s0102:c001); PDF-010:s0016 (PDF-010:s0016:c003) |
| `p006_010_015` | 短途测试开始前应如何处理故障存储器？ | 1 | — | 1 | PDF-010:s0105 (PDF-010:s0105:c001); PDF-008:s0001 (PDF-008:s0001:c232); PDF-010:s0103 (PDF-010:s0103:c001) |
| `p011_015_001` | 低空气阻力系数对 Audi A3 有什么作用？ | 1 | 18 | 10 | PDF-011:s0007 (PDF-011:s0007:c001); PDF-011:s0002 (PDF-011:s0002:c002); PDF-011:s0004 (PDF-011:s0004:c001) |
| `p011_015_003` | Audi A3 使用水性涂料有什么环境效益？ | 1 | 1 | 3 | PDF-011:s0002 (PDF-011:s0002:c002); PDF-011:s0008 (PDF-011:s0008:c001); PDF-011:s0003 (PDF-011:s0003:c001) |
| `p011_015_004` | Audi A3 前部纵梁由哪些厚度的板材组成？ | 2 | 77 | — | PDF-011:s0005 (PDF-011:s0005:c001); PDF-012:s0004 (PDF-012:s0004:c001); PDF-011:s0002 (PDF-011:s0002:c002) |
| `p011_015_005` | Audi A3 的座椅横向刚性框架由哪些部件构成？ | 1 | 61 | — | PDF-012:s0004 (PDF-012:s0004:c001); PDF-011:s0002 (PDF-011:s0002:c002); PDF-011:s0005 (PDF-011:s0005:c001) |
| `p011_015_008` | LT 的 J234 控制单元安装在哪里？ | 1 | 1 | 1 | PDF-014:s0008 (PDF-014:s0008:c001); PDF-018:s0022 (PDF-018:s0022:c001); PDF-014:s0027 (PDF-014:s0027:c001) |
| `p011_015_009` | LT 的 EDL 在什么车速以下工作？ | 2 | 2 | 1 | PDF-014:s0038 (PDF-014:s0038:c001); PDF-014:s0039 (PDF-014:s0039:c001); PDF-004:s0031 (PDF-004:s0031:c001) |
| `p011_015_013` | 侧面碰撞时加强件将力传递到哪里？ | 1 | — | 1 | PDF-012:s0007 (PDF-012:s0007:c001); PDF-008:s0001 (PDF-008:s0001:c074); PDF-012:s0012 (PDF-012:s0012:c002) |
| `p011_015_014` | CAN 便利系统短路时会怎样？ | 2 | 18 | 3 | PDF-019:s0021 (PDF-019:s0021:c001); PDF-013:s0036 (PDF-013:s0036:c001); PDF-013:s0032 (PDF-013:s0032:c001) |
| `p016_020_001` | 废气涡轮增压器的用途是什么？ | 1 | — | 1 | PDF-016:s0005 (PDF-016:s0005:c001); PDF-008:s0001 (PDF-008:s0001:c154); PDF-016:s0010 (PDF-016:s0010:c001) |
| `p016_020_002` | 如果增压压力传感器信号失效，可变叶片会怎样，发动机输出有何变化？ | 1 | — | 1 | PDF-016:s0028 (PDF-016:s0028:c001); PDF-008:s0001 (PDF-008:s0001:c232); PDF-010:s0094 (PDF-010:s0094:c001) |
| `p016_020_004` | 可变气门正时系统中，进气凸轮轴由什么驱动？ | 2 | — | 3 | PDF-018:s0052 (PDF-018:s0052:c001); PDF-008:s0001 (PDF-008:s0001:c043); PDF-017:s0135 (PDF-017:s0135:c001) |
| `p016_020_005` | 三滚子万向节的滚子如何运动？ | 2 | — | 3 | PDF-018:s0088 (PDF-018:s0088:c001); PDF-008:s0001 (PDF-008:s0001:c232); PDF-017:s0166 (PDF-017:s0166:c001) |
| `p016_020_007` | 新型车轮轴承单元用于哪类车辆的后轴？ | 1 | — | 1 | PDF-018:s0095 (PDF-018:s0095:c001); PDF-008:s0001 (PDF-008:s0001:c004); PDF-017:s0171 (PDF-017:s0171:c001) |
| `p016_020_008` | ABS速度传感器信号失效时，导航系统会怎样？ | 1 | 14 | 2 | PDF-017:s0189 (PDF-017:s0189:c001); PDF-018:s0114 (PDF-018:s0114:c001); PDF-018:s0119 (PDF-018:s0119:c001) |
| `p016_020_010` | 如果整个CAN总线失效，便利系统会发生什么？ | 1 | 7 | 1 | PDF-019:s0021 (PDF-019:s0021:c001); PDF-013:s0036 (PDF-013:s0036:c001); PDF-013:s0032 (PDF-013:s0032:c001) |
| `p016_020_011` | 车内锁止功能如何锁定和解锁所有车门？ | 1 | — | 1 | PDF-019:s0034 (PDF-019:s0034:c001); PDF-008:s0001 (PDF-008:s0001:c156); PDF-019:s0009 (PDF-019:s0009:c001) |
| `p016_020_013` | V5发动机的V形夹角和排量是多少？ | 5 | 9 | 5 | PDF-020:s0005 (PDF-020:s0005:c001); PDF-020:s0012 (PDF-020:s0012:c001); PDF-020:s0006 (PDF-020:s0006:c001) |
| `p016_020_014` | 更换V5发动机机油滤清器时，需要更换哪个部件？ | 1 | — | 13 | PDF-020:s0079 (PDF-020:s0079:c001); PDF-020:s0001 (PDF-020:s0001:c001); PDF-020:s0012 (PDF-020:s0012:c001) |

### 英文（29 条）

| case_id | query | dense rank | BM25 rank | RRF rank | top sections（RRF 后） |
|---|---|---:|---:|---:|---|
| `p001_005_pos_pdf001_02` | How does the MQB support economies of scale and wider access to driver-assistance technology? | 1 | 1 | 1 | PDF-001:s0004 (PDF-001:s0004:c001); PDF-001:s0002 (PDF-001:s0002:c001); PDF-001:s0005 (PDF-001:s0005:c001) |
| `p001_005_pos_pdf002_01` | How is the EA211 camshaft toothed belt tensioned and guided? | 1 | 2 | 1 | PDF-002:s0017 (PDF-002:s0017:c001); PDF-002:s0064 (PDF-002:s0064:c001); PDF-002:s0061 (PDF-002:s0061:c001) |
| `p001_005_pos_pdf002_03` | What is the stated purpose of the T10494 camshaft clamp in the EA211 special-tools table? | 2 | 1 | 2 | PDF-003:s0130 (PDF-003:s0130:c001); PDF-002:s0061 (PDF-002:s0061:c001); PDF-007:s0033 (PDF-007:s0033:c001) |
| `p001_005_pos_pdf003_02` | What does actuator N493 regulate, and what temperature range can it enable? | 1 | 5 | 2 | PDF-003:s0072 (PDF-003:s0072:c001); PDF-003:s0070 (PDF-003:s0070:c001); PDF-003:s0071 (PDF-003:s0071:c001) |
| `p001_005_pos_pdf004_01` | What do passive and active speed sensors inform the traction-control systems about? | 1 | 1 | 1 | PDF-004:s0010 (PDF-004:s0010:c001); PDF-004:s0011 (PDF-004:s0011:c001); PDF-004:s0002 (PDF-004:s0002:c001) |
| `p001_005_pos_pdf004_03` | What does hill descent assist (HDC) do for the driver on hilly roads? | 1 | 1 | 1 | PDF-004:s0061 (PDF-004:s0061:c001); PDF-004:s0063 (PDF-004:s0063:c001); PDF-004:s0068 (PDF-004:s0068:c001) |
| `p001_005_pos_pdf005_02` | What happens to the refrigerant and the air in the heat condenser? | 1 | 5 | 2 | PDF-017:s0207 (PDF-017:s0207:c001); PDF-005:s0026 (PDF-005:s0026:c001); PDF-005:s0044 (PDF-005:s0044:c003) |
| `p006_010_001` | What is the transfer speed range of the LIN data bus? | 1 | 1 | 1 | PDF-006:s0020 (PDF-006:s0020:c001); PDF-006:s0018 (PDF-006:s0018:c001); PDF-007:s0092 (PDF-007:s0092:c001) |
| `p006_010_002` | Which control unit manages terminal 15 run-on and the sleep and wake-up modes? | 1 | 1 | 1 | PDF-006:s0040 (PDF-006:s0040:c001); PDF-006:s0041 (PDF-006:s0041:c001); PDF-006:s0043 (PDF-006:s0043:c001) |
| `p006_010_004` | What engine generation was used as the basis for the Jetta 1.4 l 110 kW TSI engine? | 1 | 4 | 2 | PDF-002:s0014 (PDF-002:s0014:c001); PDF-007:s0031 (PDF-007:s0031:c001); PDF-007:s0009 (PDF-007:s0009:c001) |
| `p006_010_005` | How does ACC respond when it recognizes a vehicle moving ahead? | 1 | 1 | 1 | PDF-007:s0054 (PDF-007:s0054:c001); PDF-004:s0078 (PDF-004:s0078:c001); PDF-004:s0083 (PDF-004:s0083:c001) |
| `p006_010_007` | Which cover title words are directly searchable in the PDF-008 OCR? | 1 | 1 | 1 | PDF-008:s0001 (PDF-008:s0001:c096); PDF-012:s0006 (PDF-012:s0006:c001); PDF-002:s0022 (PDF-002:s0022:c001) |
| `p006_010_008` | Which four index words are directly searchable in the PDF-008 OCR? | 1 | 7 | 1 | PDF-008:s0001 (PDF-008:s0001:c005); PDF-002:s0007 (PDF-002:s0007:c001); PDF-013:s0052 (PDF-013:s0052:c001) |
| `p006_010_010` | What does the needle lift sender G80 signal to the control unit? | 1 | 1 | 1 | PDF-009:s0001 (PDF-009:s0001:c004); PDF-010:s0086 (PDF-010:s0086:c001); PDF-010:s0083 (PDF-010:s0083:c001) |
| `p006_010_011` | What does the air-mass flow meter measure? | 1 | 1 | 1 | PDF-009:s0001 (PDF-009:s0001:c005); PDF-020:s0033 (PDF-020:s0033:c001); PDF-010:s0064 (PDF-010:s0064:c001) |
| `p006_010_013` | How does the engine management system diagnose catalytic-converter conversion? | 1 | 1 | 1 | PDF-010:s0024 (PDF-010:s0024:c001); PDF-010:s0029 (PDF-010:s0029:c001); PDF-010:s0027 (PDF-010:s0027:c001) |
| `p006_010_014` | Which searchable index words appear in this PDF-008 OCR fragment? | 1 | 1 | 1 | PDF-008:s0001 (PDF-008:s0001:c005); PDF-013:s0052 (PDF-013:s0052:c001); PDF-010:s0014 (PDF-010:s0014:c001) |
| `p011_015_002` | What percentage of the Audi A3 is made of recyclable materials? | 1 | 2 | 1 | PDF-011:s0010 (PDF-011:s0010:c001); PDF-011:s0009 (PDF-011:s0009:c001); PDF-011:s0008 (PDF-011:s0008:c001) |
| `p011_015_006` | In CAN arbitration, which protocol is sent first when multiple control units transmit simultaneously? | 1 | 1 | 1 | PDF-013:s0023 (PDF-013:s0023:c001); PDF-013:s0025 (PDF-013:s0025:c001); PDF-013:s0029 (PDF-013:s0029:c001) |
| `p011_015_007` | What does the CAN status field define? | 1 | 3 | 1 | PDF-013:s0015 (PDF-013:s0015:c001); PDF-013:s0012 (PDF-013:s0012:c001); PDF-013:s0035 (PDF-013:s0035:c001) |
| `p011_015_010` | What are the Audi A3 2.3 l petrol engine's maximum torque and engine speed? | 6 | 2 | 2 | PDF-015:s0007 (PDF-015:s0007:c001); PDF-015:s0008 (PDF-015:s0008:c001); PDF-015:s0006 (PDF-015:s0006:c001) |
| `p011_015_011` | What happens when the 2.3 l petrol engine exceeds its speed limit? | 1 | 1 | 1 | PDF-015:s0089 (PDF-015:s0089:c001); PDF-015:s0007 (PDF-015:s0007:c001); PDF-015:s0060 (PDF-015:s0060:c001) |
| `p011_015_012` | Where does the oil pump send oil for piston cooling in the LT 2.3 l petrol engine? | 1 | 1 | 1 | PDF-015:s0017 (PDF-015:s0017:c001); PDF-015:s0016 (PDF-015:s0016:c001); PDF-003:s0059 (PDF-003:s0059:c001) |
| `p011_015_015` | Which optional LT equipment improves occupant safety? | 2 | 6 | 3 | PDF-014:s0007 (PDF-014:s0007:c002); PDF-007:s0023 (PDF-007:s0023:c001); PDF-014:s0006 (PDF-014:s0006:c001) |
| `p016_020_003` | What happens if no signal is received from the engine speed sender G28? | 4 | 1 | 2 | PDF-015:s0060 (PDF-015:s0060:c001); PDF-016:s0042 (PDF-016:s0042:c001); PDF-010:s0055 (PDF-010:s0055:c001) |
| `p016_020_006` | At what speed are both ventilation flaps fully open in fresh-air mode? | 2 | 2 | 2 | PDF-018:s0149 (PDF-018:s0149:c001); PDF-017:s0215 (PDF-017:s0215:c001); PDF-018:s0152 (PDF-018:s0152:c001) |
| `p016_020_009` | What safety precaution does the document require before repairing gas-discharge headlights? | 1 | 27 | 13 | PDF-017:s0063 (PDF-017:s0063:c001); PDF-018:s0007 (PDF-018:s0007:c001); PDF-018:s0124 (PDF-018:s0124:c001) |
| `p016_020_012` | Which address word initiates self-diagnosis for the convenience system central module? | 1 | 1 | 1 | PDF-019:s0081 (PDF-019:s0081:c001); PDF-013:s0036 (PDF-013:s0036:c001); PDF-018:s0130 (PDF-018:s0130:c001) |
| `p016_020_015` | What happens if the Hall sender G40 fails? | 2 | 1 | 1 | PDF-020:s0071 (PDF-020:s0071:c001); PDF-012:s0025 (PDF-012:s0025:c001); PDF-010:s0055 (PDF-010:s0055:c001) |

