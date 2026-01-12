import numpy as np
import math
import torch


class BlockMatrix:
    def __init__(self, engine, rows_logical, cols_logical, block_size=64):
        self.engine = engine
        self.rows = rows_logical
        self.cols = cols_logical
        self.block_size = block_size
        self.r_grid = math.ceil(rows_logical / block_size)
        self.c_grid = math.ceil(cols_logical / block_size)
        self.blocks = [[None for _ in range(self.c_grid)] for _ in range(self.r_grid)]

    @classmethod
    def encode_weights(cls, engine, plain_mat, block_size=64, level=None):
        """[权重模式 - 明文] 支持 level 控制空间/深度"""
        R, C = plain_mat.shape
        instance = cls(engine, R, C, block_size)
        batch_dim = engine.shape[0]

        for r in range(instance.r_grid):
            for c in range(instance.c_grid):
                r0, r1 = r * block_size, (r + 1) * block_size
                c0, c1 = c * block_size, (c + 1) * block_size

                sub_data = np.zeros((block_size, block_size), dtype=np.float32)
                real_r, real_c = min(R, r1) - r0, min(C, c1) - c0
                if real_r > 0 and real_c > 0:
                    sub_data[:real_r, :real_c] = plain_mat[r0:r0 + real_r, c0:c0 + real_c]

                batched_data = np.tile(sub_data, (batch_dim, 1, 1))

                if level is None:
                    instance.blocks[r][c] = engine.encode(batched_data)
                else:
                    instance.blocks[r][c] = engine.encode(batched_data, level=level)

        return instance

    @classmethod
    def encrypt_inputs(cls, engine, input_batch_list, sk, block_size=64, level=None):
        """[输入模式 - 密文] 显式 encode(level) 再 encrypt（若支持）"""
        num_samples = len(input_batch_list)
        R, C = input_batch_list[0].shape
        instance = cls(engine, R, C, block_size)

        for r in range(instance.r_grid):
            for c in range(instance.c_grid):
                combined_data = np.zeros((engine.shape[0], block_size, block_size), dtype=np.float32)

                for i, sample_mat in enumerate(input_batch_list):
                    r0, r1 = r * block_size, (r + 1) * block_size
                    c0, c1 = c * block_size, (c + 1) * block_size
                    sub_data = sample_mat[r0:r1, c0:c1]
                    h, w = sub_data.shape
                    combined_data[i, :h, :w] = sub_data

                if level is None:
                    instance.blocks[r][c] = engine.encrypt(combined_data, sk)
                else:
                    pt = engine.encode(combined_data, level=level)
                    try:
                        instance.blocks[r][c] = engine.encrypt(pt, sk)
                    except TypeError:
                        # 如果该版本 encrypt 不接受 plaintext 对象，只能退回默认 encrypt
                        instance.blocks[r][c] = engine.encrypt(combined_data, sk)

        return instance

    def matmul(self, other, mult_key):
        assert self.cols == other.rows, f"Dim mismatch: {self.cols} vs {other.rows}"
        res = BlockMatrix(self.engine, self.rows, other.cols, self.block_size)

        for i in range(res.r_grid):
            for j in range(res.c_grid):
                sum_ct = None
                for k in range(self.c_grid):
                    ct_a = self.blocks[i][k]
                    ct_b = other.blocks[k][j]

                    prod = self.engine.matrix_multiply(ct_a, ct_b, mult_key)

                    if sum_ct is None:
                        sum_ct = prod
                    else:
                        sum_ct = self.engine.add(sum_ct, prod)

                    del prod

                res.blocks[i][j] = sum_ct

        return res

    def add(self, other):
        res = BlockMatrix(self.engine, self.rows, self.cols, self.block_size)
        for r in range(self.r_grid):
            for c in range(self.c_grid):
                res.blocks[r][c] = self.engine.add(self.blocks[r][c], other.blocks[r][c])
        return res

    def square(self, hadamard_key):
        res = BlockMatrix(self.engine, self.rows, self.cols, self.block_size)
        for r in range(self.r_grid):
            for c in range(self.c_grid):
                if self.blocks[r][c] is not None:
                    res.blocks[r][c] = self.engine.hadamard_multiply(
                        self.blocks[r][c],
                        self.blocks[r][c],
                        hadamard_key
                    )
        return res

    def get_level(self):
        if self.blocks[0][0] and hasattr(self.blocks[0][0], "level"):
            return self.blocks[0][0].level
        return "Plaintext"
