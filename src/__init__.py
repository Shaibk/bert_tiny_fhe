# 从子模块中导出核心类，方便外部直接调用
# 这样在 main.py 中就可以使用 "from src import BlockMatrix" 这种简洁写法

from .block_matrix import BlockMatrix
from .bert_layers import FHEBertTinyEncoder
from .ops import approx_gelu, approx_softmax_attention

# 定义当使用 "from src import *" 时导出的内容
__all__ = [
    "BlockMatrix",
    "FHEBertTinyEncoder",
    "approx_gelu",
    "approx_softmax_attention",
]