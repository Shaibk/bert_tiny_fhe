from .block_matrix import BlockMatrix

def approx_gelu(block_mat: BlockMatrix, hadamard_key) -> BlockMatrix:
    """GELU Approx: x^2"""
    return block_mat.square(hadamard_key)

def approx_softmax_attention(score_mat: BlockMatrix) -> BlockMatrix: # 这里的参数签名其实在 bert_layers 里直接调用的 .square
    # 为了保持接口统一，可以不改这里，直接在 bert_layers 里调 .square
    # 或者如果你在 bert_layers 依然调用这个函数：
    pass 
    # 注意：bert_layers.py 中我改成了直接调用 score.square(key)，所以 ops.py 的这个函数其实被跳过了
    # 但为了完整性，如果要用 ops.py:
    # return score_mat.square(hadamard_key)