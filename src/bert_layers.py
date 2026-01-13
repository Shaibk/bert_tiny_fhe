import gc
import torch
import numpy as np
from src.block_matrix import BlockMatrix

class FHEBertTinyEncoder:
    """
    FHE Tiny-BERT Encoder (8-Level Optimized)
    - Dynamic weight encoding (streamed) to avoid OOM
    """

    def __init__(self, engine, mult_key, hadamard_key, transposition_key, weights_path=None, layer_idx=0):
        self.engine = engine
        self.mult_key = mult_key
        self.hadamard_key = hadamard_key
        self.transposition_key = transposition_key
        self.layer_idx = layer_idx
        self.hidden_size = 128

        if weights_path:
            self._load_weights(weights_path)
        else:
            raise ValueError("Weights path required.")

    def _load_weights(self, path):
        print(f"   [Init] Loading fused weights from: {path}")
        data = np.load(path)
        p = f"encoder.layer.{self.layer_idx}"

        def ld_vec(k):
            return data[f"{p}.{k}"].astype(np.float32)

        def ld_w(k):
            w = data[f"{p}.{k}"].astype(np.float32)
            if w.ndim == 2:
                return w.T                    # [out,in] -> [in,out]
            elif w.ndim == 3:
                return np.transpose(w, (0, 2, 1))  # [h,out,in] -> [h,in,out]
            else:
                raise ValueError(f"Unexpected weight ndim for {p}.{k}: {w.ndim}")

        self.np_wq = ld_w("attention.self.query.weight")
        self.np_wk = ld_w("attention.self.key.weight")
        self.np_wv_fused = ld_w("attention.self.value_fused.weight")

        self.np_norm1_bias = ld_vec("attention.output.norm.bias")

        self.np_ff1 = ld_w("intermediate.dense.weight")
        self.np_ff2_quad = ld_w("output.dense.weight_quad")
        self.np_ff2_lin  = ld_w("output.dense.weight_lin")
        self.np_ff2_bias = ld_vec("output.dense.bias_fused")
        self.np_norm2_bias = ld_vec("output.norm.bias")

        print(f"   [DEBUG] np_wq shape: {self.np_wq.shape}, ndim={self.np_wq.ndim}")
        print(f"   [DEBUG] np_wk shape: {self.np_wk.shape}, ndim={self.np_wk.ndim}")
        print(f"   [DEBUG] np_wv_fused shape: {self.np_wv_fused.shape}, ndim={self.np_wv_fused.ndim}")

    def _add_bias(self, x: BlockMatrix, bias_vec):
        bias_full = np.tile(bias_vec, (x.rows, 1)).astype(np.float32)
        # bias 体积小，encode_weights 可以接受；如果你也想更省，可改成分块 encode（但通常没必要）
        w_bias = BlockMatrix.encode_weights(self.engine, bias_full, level=x.get_level())
        res = x.add(w_bias)
        del w_bias
        return res

    def forward_one_layer(self, x_enc: BlockMatrix, attention_mask=None):
        current_level = x_enc.get_level()
        print(f"   [FHE] Input Level: {current_level}")

        # ==========================
        # Part 1: Attention
        # ==========================

        # ✅ Q = X @ Wq（流式编码，不生成整张 w_q BlockMatrix）
        q = x_enc.matmul_np_stream(self.np_wq, self.mult_key, level=current_level)
        torch.cuda.empty_cache()

        # ✅ K = X @ Wk（流式编码）
        k = x_enc.matmul_np_stream(self.np_wk, self.mult_key, level=current_level)
        torch.cuda.empty_cache()

        # K^T
        k_t = k.transpose(self.transposition_key)
        del k
        torch.cuda.empty_cache()

        # Score = Q @ K.T
        score = q.matmul(k_t, self.mult_key)
        del q, k_t
        torch.cuda.empty_cache()

        # Shift (+4)
        print("     > [Attention] Shift Score (+4.0)...")
        score_shifted = score.add_scalar(4.0)
        del score

        # Square
        print("     > [Attention] Square...")
        probs = score_shifted.square(self.hadamard_key)
        del score_shifted
        torch.cuda.empty_cache()

        # Mask (key-mask only)
        if attention_mask is not None:
            print("     > [Attention] Applying Mask...")
            probs = probs.mul_plain(attention_mask)

        # ✅ V_projected = X @ Wv_fused（流式编码）
        v_projected = x_enc.matmul_np_stream(self.np_wv_fused, self.mult_key, level=current_level)
        torch.cuda.empty_cache()

        # Context
        output = probs.matmul(v_projected, self.mult_key)
        del probs, v_projected
        torch.cuda.empty_cache()

        # Residual
        res1 = x_enc.add(output)

        # Norm1 bias-only
        print(f"     > [Norm1] Bias Addition (Level {res1.get_level()})...")
        norm1 = self._add_bias(res1, self.np_norm1_bias)
        del res1
        torch.cuda.empty_cache()

        # ==========================
        # Part 2: FFN
        # ==========================

        print("     > [FFN] Linear 1...")
        ff1 = norm1.matmul_np_stream(self.np_ff1, self.mult_key, level=norm1.get_level())
        torch.cuda.empty_cache()

        print("     > [FFN] Split-Path PolyGELU...")

        ff1_sq = ff1.square(self.hadamard_key)

        term_quad = ff1_sq.matmul_np_stream(self.np_ff2_quad, self.mult_key, level=ff1_sq.get_level())
        del ff1_sq
        torch.cuda.empty_cache()

        term_lin = ff1.matmul_np_stream(self.np_ff2_lin, self.mult_key, level=ff1.get_level())
        del ff1
        torch.cuda.empty_cache()

        ff2 = term_quad.add(term_lin)
        del term_quad, term_lin

        ff2 = self._add_bias(ff2, self.np_ff2_bias)

        res2 = norm1.add(ff2)
        del norm1, ff2

        print("     > [Norm2] Bias Addition...")
        final_out = self._add_bias(res2, self.np_norm2_bias)

        gc.collect()
        torch.cuda.empty_cache()
        return final_out
