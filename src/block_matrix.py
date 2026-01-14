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
        只编码一个权重 block（支持 2D/3D），返回 engine.encode 的 GLPlaintext
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
        """加密输入数据 (支持 Batch/Head Packing)"""
        input_np = np.array(input_batch_list) if isinstance(input_batch_list, list) else input_batch_list
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
        ✅ Cipher @ Plain(Numpy) 的流式权重编码
        - plain_mat_np 支持 [R,C] 或 [Heads,R,C]
        - 每次只 encode 1 个权重 block，乘完立刻释放，避免 OOM
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

                    encoded_b = BlockMatrix._encode_weight_block(
                        engine=self.engine,
                        plain_mat=plain_mat_np,
                        r=k, c=j,
                        block_size=self.block_size,
                        level=level
                    )

                    prod = self.engine.matrix_multiply(ct_a, encoded_b, mult_key)
                    del encoded_b  # ✅ 关键：释放 GPU plaintext

                    if sum_ct is None:
                        sum_ct = prod
                    else:
                        sum_ct = self.engine.add(sum_ct, prod)
                        del prod

                res.blocks[i][j] = sum_ct
        return res

    def add(self, other):
        """逐元素加法"""
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
        """Matrix + scalar（用于 +4）"""
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
        Cipher ⊙ Plain（逐元素乘）
        支持:
        - plain_mat_np: [rows, cols]        (shared mask, old behavior)
        - plain_mat_np: [B, rows, cols]     (per-lane mask, new behavior)
        """
        # Determine mask mode
        if plain_mat_np.ndim == 2:
            shared = True
            if plain_mat_np.shape != (self.rows, self.cols):
                raise ValueError(f"Mask shape {plain_mat_np.shape} mismatch matrix {(self.rows, self.cols)}")
            B = int(self.engine.shape[0])
        elif plain_mat_np.ndim == 3:
            shared = False
            B = int(plain_mat_np.shape[0])
            if B != int(self.engine.shape[0]):
                raise ValueError(f"Mask batch {B} mismatch engine batch {self.engine.shape[0]}")
            if plain_mat_np.shape[1:] != (self.rows, self.cols):
                raise ValueError(f"Mask shape {plain_mat_np.shape[1:]} mismatch matrix {(self.rows, self.cols)}")
        else:
            raise ValueError(f"mask ndim must be 2 or 3, got {plain_mat_np.ndim}")

        res = BlockMatrix(self.engine, self.rows, self.cols, self.block_size)
        current_level = self.get_level()

        for r in range(self.r_grid):
            for c in range(self.c_grid):
                if self.blocks[r][c] is None:
                    continue

                r0, r1 = r * self.block_size, (r + 1) * self.block_size
                c0, c1 = c * self.block_size, (c + 1) * self.block_size

                real_r = min(self.rows, r1) - r0
                real_c = min(self.cols, c1) - c0

                if shared:
                    # [64,64] -> broadcast to [B,64,64]
                    sub = np.zeros((self.block_size, self.block_size), dtype=np.float32)
                    if real_r > 0 and real_c > 0:
                        sub[:real_r, :real_c] = plain_mat_np[r0:r0+real_r, c0:c0+real_c]
                    combined = np.zeros((B, self.block_size, self.block_size), dtype=np.float32)
                    combined[:] = sub
                else:
                    # [B,rows,cols] -> [B,64,64]
                    combined = np.zeros((B, self.block_size, self.block_size), dtype=np.float32)
                    if real_r > 0 and real_c > 0:
                        combined[:, :real_r, :real_c] = plain_mat_np[:, r0:r0+real_r, c0:c0+real_c]

                encoded_mask = self.engine.encode(combined, level=current_level)
                res.blocks[r][c] = self.engine.hadamard_multiply(self.blocks[r][c], encoded_mask)

        return res



    def square(self, hadamard_key):
        """平方（Hadamard 自乘）"""
        res = BlockMatrix(self.engine, self.rows, self.cols, self.block_size)
        for r in range(self.r_grid):
            for c in range(self.c_grid):
                if self.blocks[r][c] is not None:
                    res.blocks[r][c] = self.engine.hadamard_multiply(
                        self.blocks[r][c], self.blocks[r][c], hadamard_key
                    )
        return res

    def transpose(self, transposition_key):
        """矩阵转置"""
        res = BlockMatrix(self.engine, self.cols, self.rows, self.block_size)
        for r in range(self.r_grid):
            for c in range(self.c_grid):
                blk = self.blocks[r][c]
                if blk is not None:
                    res.blocks[c][r] = self.engine.transpose(blk, transposition_key)
        return res

    # ==========================
    # ✅ 新增：方案A核心 - 视图切片
    # ==========================

    def slice_cols(self, col_start: int, col_end: int) -> "BlockMatrix":
        """
        视图切列（不做任何 HE 运算，只重组 blocks 引用）
        目前要求 col_start/col_end 与 block_size 对齐（你这里 0/64/128 完全满足）
        """
        if col_start % self.block_size != 0 or col_end % self.block_size != 0:
            raise ValueError("slice_cols requires block-aligned boundaries for now.")
        if not (0 <= col_start < col_end <= self.cols):
            raise ValueError(f"slice_cols range invalid: [{col_start}, {col_end}) for cols={self.cols}")

        new_cols = col_end - col_start
        new = BlockMatrix(self.engine, self.rows, new_cols, self.block_size)

        c0 = col_start // self.block_size
        c1 = col_end // self.block_size  # exclusive
        # copy the selected column-blocks
        for r in range(self.r_grid):
            for j_new, j_old in enumerate(range(c0, c1)):
                new.blocks[r][j_new] = self.blocks[r][j_old]
        return new

    def slice_rows(self, row_start: int, row_end: int) -> "BlockMatrix":
        """
        视图切行（不做任何 HE 运算，只重组 blocks 引用）
        要求 row_start/row_end 与 block_size 对齐（你这里 0/64/128 完全满足）
        """
        if row_start % self.block_size != 0 or row_end % self.block_size != 0:
            raise ValueError("slice_rows requires block-aligned boundaries for now.")
        if not (0 <= row_start < row_end <= self.rows):
            raise ValueError(f"slice_rows range invalid: [{row_start}, {row_end}) for rows={self.rows}")

        new_rows = row_end - row_start
        new = BlockMatrix(self.engine, new_rows, self.cols, self.block_size)

        r0 = row_start // self.block_size
        r1 = row_end // self.block_size  # exclusive
        for i_new, i_old in enumerate(range(r0, r1)):
            for c in range(self.c_grid):
                new.blocks[i_new][c] = self.blocks[i_old][c]
        return new

    def get_level(self):
        """获取任意一个有效块的 level"""
        for r in range(self.r_grid):
            for c in range(self.c_grid):
                if self.blocks[r][c] is not None and hasattr(self.blocks[r][c], "level"):
                    return self.blocks[r][c].level
        return None
