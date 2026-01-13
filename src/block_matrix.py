import numpy as np
import math

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
        """
        [权重模式]
        支持 plain_mat 为 [Rows, Cols] (广播)
        或 [Heads, Rows, Cols] (Head Packing 分发)
        """
        is_stacked = (plain_mat.ndim == 3)
        if is_stacked:
            Heads, R, C = plain_mat.shape
            batch_per_head = engine.shape[0] // Heads
        else:
            R, C = plain_mat.shape

        instance = cls(engine, R, C, block_size)

        for r in range(instance.r_grid):
            for c in range(instance.c_grid):
                encoded_blk = cls._encode_weight_block(
                    engine=engine,
                    plain_mat=plain_mat,
                    r=r, c=c,
                    block_size=block_size,
                    level=level
                )
                instance.blocks[r][c] = encoded_blk

        return instance

    @staticmethod
    def _encode_weight_block(engine, plain_mat, r, c, block_size=64, level=None):
        """
        ✅ 新增：只编码一个权重 block（支持 2D/3D）。
        返回的是 engine.encode 得到的 GLPlaintext（在 GPU 上）。
        """
        is_stacked = (plain_mat.ndim == 3)

        if is_stacked:
            Heads, R, C = plain_mat.shape
            batch_per_head = engine.shape[0] // Heads
        else:
            R, C = plain_mat.shape

        r0, r1 = r * block_size, (r + 1) * block_size
        c0, c1 = c * block_size, (c + 1) * block_size

        combined_data = np.zeros((engine.shape[0], block_size, block_size), dtype=np.float32)

        if is_stacked:
            for h in range(Heads):
                sub_data = np.zeros((block_size, block_size), dtype=np.float32)
                real_r = min(R, r1) - r0
                real_c = min(C, c1) - c0
                if real_r > 0 and real_c > 0:
                    sub_data[:real_r, :real_c] = plain_mat[h, r0:r0+real_r, c0:c0+real_c]

                start_b = h * batch_per_head
                end_b = (h + 1) * batch_per_head
                combined_data[start_b:end_b] = sub_data
        else:
            sub_data = np.zeros((block_size, block_size), dtype=np.float32)
            real_r = min(R, r1) - r0
            real_c = min(C, c1) - c0
            if real_r > 0 and real_c > 0:
                sub_data[:real_r, :real_c] = plain_mat[r0:r0+real_r, c0:c0+real_c]
            combined_data[:] = sub_data

        if level is not None:
            return engine.encode(combined_data, level=level)
        return engine.encode(combined_data)

    @classmethod
    def encrypt_inputs(cls, engine, input_batch_list, sk, block_size=64):
        """[输入模式] 加密输入数据 (支持 Batch/Head Packing)"""
        if isinstance(input_batch_list, list):
            input_np = np.array(input_batch_list)
        else:
            input_np = input_batch_list

        B, R, C = input_np.shape
        instance = cls(engine, R, C, block_size)

        for r in range(instance.r_grid):
            for c in range(instance.c_grid):
                r0, r1 = r * block_size, (r + 1) * block_size
                c0, c1 = c * block_size, (c + 1) * block_size

                sub_batch = np.zeros((B, block_size, block_size), dtype=np.float32)
                real_r, real_c = min(R, r1) - r0, min(C, c1) - c0
                if real_r > 0 and real_c > 0:
                    sub_batch[:, :real_r, :real_c] = input_np[:, r0:r0+real_r, c0:c0+real_c]

                instance.blocks[r][c] = engine.encrypt(sub_batch, sk)
        return instance

    def matmul(self, other, mult_key):
        """矩阵乘法 (Cipher @ Cipher 或 Cipher @ Plain(BlockMatrix))"""
        assert self.cols == other.rows, f"Dim mismatch: {self.cols} vs {other.rows}"
        res = BlockMatrix(self.engine, self.rows, other.cols, self.block_size)

        for i in range(res.r_grid):
            for j in range(res.c_grid):
                sum_ct = None
                for k in range(self.c_grid):
                    ct_a = self.blocks[i][k]
                    val_b = other.blocks[k][j]
                    if ct_a is None or val_b is None:
                        continue

                    prod = self.engine.matrix_multiply(ct_a, val_b, mult_key)

                    if sum_ct is None:
                        sum_ct = prod
                    else:
                        sum_ct = self.engine.add(sum_ct, prod)
                        del prod
                res.blocks[i][j] = sum_ct
        return res

    def matmul_np_stream(self, plain_mat_np, mult_key, level=None):
        """
        ✅ 新增：Cipher @ Plain(Numpy) 的“流式权重编码”版本（OOM-safe）
        - plain_mat_np 支持 [R,C] 或 [Heads,R,C]
        - 不会把整个权重矩阵 encode 成 BlockMatrix 常驻 GPU
        - 每次只 encode 1 个 block -> matrix_multiply -> 立刻释放
        """
        is_stacked = (plain_mat_np.ndim == 3)
        if is_stacked:
            Heads, R, C = plain_mat_np.shape
        else:
            R, C = plain_mat_np.shape

        assert self.cols == R, f"Dim mismatch: {self.cols} vs {R}"
        res = BlockMatrix(self.engine, self.rows, C, self.block_size)

        if level is None:
            level = self.get_level()

        for i in range(res.r_grid):
            for j in range(res.c_grid):
                sum_ct = None
                for k in range(self.c_grid):
                    ct_a = self.blocks[i][k]
                    if ct_a is None:
                        continue

                    # encode 仅此一个 block： (k,j) 对应 other 的 rows block=k, cols block=j
                    encoded_b = BlockMatrix._encode_weight_block(
                        engine=self.engine,
                        plain_mat=plain_mat_np,
                        r=k, c=j,
                        block_size=self.block_size,
                        level=level
                    )

                    prod = self.engine.matrix_multiply(ct_a, encoded_b, mult_key)
                    del encoded_b  # ✅ 关键：立即释放 GPU plaintext

                    if sum_ct is None:
                        sum_ct = prod
                    else:
                        sum_ct = self.engine.add(sum_ct, prod)
                        del prod

                res.blocks[i][j] = sum_ct
        return res

    def add(self, other):
        """逐元素加法 (支持自动 Level 对齐)"""
        res = BlockMatrix(self.engine, self.rows, self.cols, self.block_size)
        for r in range(self.r_grid):
            for c in range(self.c_grid):
                blk_a = self.blocks[r][c]
                blk_b = other.blocks[r][c]

                if blk_a is None:
                    res.blocks[r][c] = blk_b
                elif blk_b is None:
                    res.blocks[r][c] = blk_a
                else:
                    res.blocks[r][c] = self.engine.add(blk_a, blk_b)
        return res

    def add_scalar(self, scalar_val):
        """
        密文加标量: Matrix + c
        用于 Attention 的 (+4.0) 操作
        """
        res = BlockMatrix(self.engine, self.rows, self.cols, self.block_size)
        scalar_mat = np.full((self.block_size, self.block_size), scalar_val, dtype=np.float32)
        current_level = self.get_level()

        for r in range(self.r_grid):
            for c in range(self.c_grid):
                if self.blocks[r][c] is not None:
                    encoded_scalar = self.engine.encode(scalar_mat, level=current_level)
                    res.blocks[r][c] = self.engine.add(self.blocks[r][c], encoded_scalar)
        return res

    def mul_plain(self, plain_mat_np):
        """
        密文与明文矩阵逐元素相乘 (Hadamard Product)
        Cipher * Plain 不消耗 Key
        """
        if plain_mat_np.shape != (self.rows, self.cols):
            raise ValueError(f"Mask shape {plain_mat_np.shape} mismatch matrix {(self.rows, self.cols)}")

        res = BlockMatrix(self.engine, self.rows, self.cols, self.block_size)
        current_level = self.get_level()

        for r in range(self.r_grid):
            for c in range(self.c_grid):
                if self.blocks[r][c] is not None:
                    r0, r1 = r * self.block_size, (r + 1) * self.block_size
                    c0, c1 = c * self.block_size, (c + 1) * self.block_size

                    sub_mask = np.zeros((self.block_size, self.block_size), dtype=np.float32)
                    real_r = min(self.rows, r1) - r0
                    real_c = min(self.cols, c1) - c0

                    if real_r > 0 and real_c > 0:
                        sub_mask[:real_r, :real_c] = plain_mat_np[r0:r0+real_r, c0:c0+real_c]

                    encoded_mask = self.engine.encode(sub_mask, level=current_level)
                    res.blocks[r][c] = self.engine.hadamard_multiply(self.blocks[r][c], encoded_mask)
        return res

    def square(self, hadamard_key):
        """平方 (Hadamard Mul)"""
        res = BlockMatrix(self.engine, self.rows, self.cols, self.block_size)
        for r in range(self.r_grid):
            for c in range(self.c_grid):
                if self.blocks[r][c] is not None:
                    res.blocks[r][c] = self.engine.hadamard_multiply(
                        self.blocks[r][c], self.blocks[r][c], hadamard_key
                    )
        return res

    def transpose(self, transposition_key):
        """矩阵转置: M.T"""
        res = BlockMatrix(self.engine, self.cols, self.rows, self.block_size)
        for r in range(self.r_grid):
            for c in range(self.c_grid):
                blk = self.blocks[r][c]
                if blk is not None:
                    transposed_blk = self.engine.transpose(blk, transposition_key)
                    res.blocks[c][r] = transposed_blk
        return res

    def get_level(self):
        """获取第一个有效块的 Level"""
        for r in range(self.r_grid):
            for c in range(self.c_grid):
                if self.blocks[r][c] is not None:
                    if hasattr(self.blocks[r][c], "level"):
                        return self.blocks[r][c].level
        return None
