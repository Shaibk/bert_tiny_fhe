# accuracy_first (方案一：2Quad/p=2 + 二次GeLU，乘法层≤8)

目标：
- 先不考虑FHE推理，提升明文/浮点精度与训练稳定性；
- 同时保持算子形态：2Quad(或PowerSoftmax p=2) + 二次GeLU，便于后续回到FHE。

核心约束（严格口径）：
从输入构建QKV开始，包含Wo与FFN：
- QKV 线性(合并) 1
- QK^T 1
- 2Quad平方 1
- PV 1
- Wo 1
- FFN1 1
- GeLU二次平方 1
- FFN2 1
总计 8 次“乘法层”（按你要求的严格口径）。
