import numpy as np
import torch
import gc

from .block_matrix import BlockMatrix
from .ops import approx_gelu


class FHEBertTinyEncoder:
    def __init__(self, engine, mult_key, hadamard_key, level=None, ffn_w1_delta=5, ffn_w2_delta=7):
        self.engine = engine
        self.mult_key = mult_key
        self.hadamard_key = hadamard_key

        self.hidden_size = 128
        self.ffn_size = 512

        # 初始 encode level（None 表示用库默认 max）
        self.level = level

        # 你测出来的“从 init level 到 FFN 两次 matmul 前激活的 level 下降量”
        # 默认按你现在的实验：W1 用 level-5，W2 用 level-7
        self.ffn_w1_delta = ffn_w1_delta
        self.ffn_w2_delta = ffn_w2_delta

        self._init_random_weights_cpu()

    def _init_random_weights_cpu(self):
        print("   [Init] Generating Weights on CPU...")
        self.np_wq = np.random.randn(self.hidden_size, self.hidden_size).astype(np.float32)
        self.np_wk = np.random.randn(self.hidden_size, self.hidden_size).astype(np.float32)
        self.np_wv = np.random.randn(self.hidden_size, self.hidden_size).astype(np.float32)
        self.np_wo = np.random.randn(self.hidden_size, self.hidden_size).astype(np.float32)
        self.np_ff1 = np.random.randn(self.hidden_size, self.ffn_size).astype(np.float32)
        self.np_ff2 = np.random.randn(self.ffn_size, self.hidden_size).astype(np.float32)

    def forward_one_layer(self, x_enc: BlockMatrix):
        # --- Q ---
        print("     > [1/6] Encoding W_Q & Computing Q...")
        w_q_pt = BlockMatrix.encode_weights(self.engine, self.np_wq, level=self.level)
        q = x_enc.matmul(w_q_pt, self.mult_key)
        del w_q_pt
        torch.cuda.empty_cache()

        # --- K ---
        print("     > [2/6] Encoding W_K & Computing K...")
        w_k_pt = BlockMatrix.encode_weights(self.engine, self.np_wk, level=self.level)
        k = x_enc.matmul(w_k_pt, self.mult_key)
        del w_k_pt
        torch.cuda.empty_cache()

        # --- Score ---
        print("     > [3/6] Computing Attention Score (Q * K)...")
        score = q.matmul(k, self.mult_key)
        del q, k
        torch.cuda.empty_cache()
        gc.collect()

        # --- Softmax approx ---
        print("     > [Softmax] Approx with Square...")
        attn_probs = score.square(self.hadamard_key)
        del score
        torch.cuda.empty_cache()

        # --- V ---
        print("     > [4/6] Encoding W_V & Computing V...")
        w_v_pt = BlockMatrix.encode_weights(self.engine, self.np_wv, level=self.level)
        v = x_enc.matmul(w_v_pt, self.mult_key)
        del w_v_pt
        torch.cuda.empty_cache()

        # --- Context ---
        print("     > [5/6] Computing Context (Probs * V)...")
        context = attn_probs.matmul(v, self.mult_key)
        del attn_probs, v
        torch.cuda.empty_cache()

        # --- Output proj ---
        print("     > [6/6] Output Projection...")
        w_o_pt = BlockMatrix.encode_weights(self.engine, self.np_wo, level=self.level)
        output = context.matmul(w_o_pt, self.mult_key)
        del w_o_pt
        torch.cuda.empty_cache()

        attention_out = x_enc.add(output)
        del output
        torch.cuda.empty_cache()

        # ===== FFN with level-aware weight encoding =====
        if self.level is None:
            # 没有指定初始 level 时，无法按 “level-Δ” 推导，退回默认
            level_w1 = None
            level_w2 = None
        else:
            level_w1 = max(1, self.level - self.ffn_w1_delta)
            level_w2 = max(1, self.level - self.ffn_w2_delta)

        print(f"     > [FFN] Linear 1 (encode level={level_w1})...")
        w_ff1_pt = BlockMatrix.encode_weights(self.engine, self.np_ff1, level=level_w1)
        ff1 = attention_out.matmul(w_ff1_pt, self.mult_key)
        print("[Check] W1 pt.level =", w_ff1_pt.blocks[0][0].level)
        del w_ff1_pt
        torch.cuda.empty_cache()

        print("     > [FFN] GELU...")
        ff_gelu = approx_gelu(ff1, self.hadamard_key)
        del ff1
        torch.cuda.empty_cache()

        print(f"     > [FFN] Linear 2 (encode level={level_w2})...")
        w_ff2_pt = BlockMatrix.encode_weights(self.engine, self.np_ff2, level=level_w2)
        ff2 = ff_gelu.matmul(w_ff2_pt, self.mult_key)
        print("[Check] W2 pt.level =", w_ff2_pt.blocks[0][0].level)
        del w_ff2_pt, ff_gelu
        torch.cuda.empty_cache()

        final_out = attention_out.add(ff2)
        return final_out
