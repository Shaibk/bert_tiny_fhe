import os
import gc
import torch
import numpy as np

from src.block_matrix import BlockMatrix


class FHEBertTinyEncoder:
    """
    FHE Tiny-BERT Encoder (8-Level Optimized)
    - Heads inside lane
    - Slice-view head split (no extra mult depth)
    - Uses fused V@O per head
    - Supports per-head masks: attention_mask can be:
        - mask: np.ndarray [L,L] (shared for both heads)
        - (mask0, mask1): tuple of np.ndarray [L,L], applied to head0/head1 respectively

    Debug:
    - FHE_DEBUG=1 enables prints
    - set encoder.debug_sk=sk to decrypt probs/attn-output
    """

    def __init__(self, engine, mult_key, hadamard_key, transposition_key, weights_path=None, layer_idx=0):
        self.engine = engine
        self.mult_key = mult_key
        self.hadamard_key = hadamard_key
        self.transposition_key = transposition_key
        self.layer_idx = layer_idx
        self.hidden_size = 128
        self.head_dim = 64

        self.debug_enabled = os.getenv("FHE_DEBUG", "0") == "1"
        self.debug_sk = None
        self.debug_real_len = None

        if not weights_path:
            raise ValueError("Weights path required.")
        self._load_weights(weights_path)

        self._dbg_probs_done = False
        self._dbg_attn_out_done = False

    def _load_weights(self, path: str):
        print(f"   [Init] Loading fused weights from: {path}")
        data = np.load(path)
        p = f"encoder.layer.{self.layer_idx}"

        def ld_vec(k: str) -> np.ndarray:
            return data[f"{p}.{k}"].astype(np.float32)

        def ld_w2d(k: str) -> np.ndarray:
            w = data[f"{p}.{k}"].astype(np.float32)
            if w.ndim != 2:
                raise ValueError(f"Expected 2D weight for {p}.{k}, got ndim={w.ndim}")
            return w.T  # [out,in] -> [in,out]

        self.np_wq = ld_w2d("attention.self.query.weight")
        self.np_wk = ld_w2d("attention.self.key.weight")
        self.np_wv_fused_h0 = ld_w2d("attention.self.value_fused_head0.weight")
        self.np_wv_fused_h1 = ld_w2d("attention.self.value_fused_head1.weight")

        self.tau = ld_vec("attention.self.tau")  # [2]

        self.np_norm1_bias = ld_vec("attention.output.norm.bias")

        self.np_ff1 = ld_w2d("intermediate.dense.weight")
        self.np_ff2_quad = ld_w2d("output.dense.weight_quad")
        self.np_ff2_lin  = ld_w2d("output.dense.weight_lin")
        self.np_ff2_bias = ld_vec("output.dense.bias_fused")
        self.np_norm2_bias = ld_vec("output.norm.bias")

        print(f"   [DEBUG] np_wq shape: {self.np_wq.shape}, ndim={self.np_wq.ndim}")
        print(f"   [DEBUG] np_wk shape: {self.np_wk.shape}, ndim={self.np_wk.ndim}")
        print(f"   [DEBUG] np_wv_fused_h0 shape: {self.np_wv_fused_h0.shape}, ndim={self.np_wv_fused_h0.ndim}")
        print(f"   [DEBUG] np_wv_fused_h1 shape: {self.np_wv_fused_h1.shape}, ndim={self.np_wv_fused_h1.ndim}")

        if self.debug_enabled:
            print(f"   [DEBUG] tau(L{self.layer_idx}) = {self.tau}")

    def _add_bias(self, x: BlockMatrix, bias_vec: np.ndarray) -> BlockMatrix:
        bias_full = np.tile(bias_vec, (x.rows, 1)).astype(np.float32)
        w_bias = BlockMatrix.encode_weights(self.engine, bias_full, level=x.get_level())
        res = x.add(w_bias)
        del w_bias
        return res

    def _parse_masks(self, attention_mask):
        """
        Returns (mask0, mask1).

        Supported:
        - None
        - mask: np.ndarray [L,L] or [B,L,L]  (shared for both heads)
        - (mask0, mask1): tuple/list where each is [L,L] or [B,L,L]
        """
        if attention_mask is None:
            return None, None
        if isinstance(attention_mask, (tuple, list)):
            assert len(attention_mask) == 2, "per-head attention_mask must be (mask0, mask1)"
            return attention_mask[0], attention_mask[1]
        return attention_mask, attention_mask


    def _dbg_probs(self, probs0: BlockMatrix, probs1: BlockMatrix):
        if (not self.debug_enabled) or self._dbg_probs_done:
            return
        self._dbg_probs_done = True
        if self.debug_sk is None:
            print("   [FHE DEBUG] debug_sk not set; set bert_l*.debug_sk = sk to enable decrypt debug")
            return
        try:
            b0 = probs0.blocks[0][0]
            b1 = probs1.blocks[0][0]
            t0 = self.engine.decrypt(b0, self.debug_sk)
            t1 = self.engine.decrypt(b1, self.debug_sk)
            print(f"   [FHE DEBUG][L{self.layer_idx}] head0 probs row0 sum/max (0:32): {float(t0[0,0,:32].sum()):.6g} / {float(t0[0,0,:32].max()):.6g}")
            print(f"   [FHE DEBUG][L{self.layer_idx}] head1 probs row0 sum/max (0:32): {float(t1[0,0,:32].sum()):.6g} / {float(t1[0,0,:32].max()):.6g}")
        except Exception as e:
            print(f"   [FHE DEBUG] decrypt probs failed: {type(e).__name__}: {e}")

    def _dbg_attn_output(self, output: BlockMatrix):
        if (not self.debug_enabled) or self._dbg_attn_out_done:
            return
        self._dbg_attn_out_done = True
        if self.debug_sk is None:
            print("   [FHE DEBUG] debug_sk not set; cannot decrypt attn-output")
            return
        try:
            t0 = self.engine.decrypt(output.blocks[0][0], self.debug_sk)
            t1 = self.engine.decrypt(output.blocks[0][1], self.debug_sk)
            vec = np.concatenate([t0[0,0,:], t1[0,0,:]]).astype(np.float32)
            print(f"   [FHE DEBUG][L{self.layer_idx}] attn-output CLS first5: {vec[:5]}")
            print(f"   [FHE DEBUG][L{self.layer_idx}] attn-output CLS L2/absmax: {float(np.linalg.norm(vec)):.6g} / {float(np.max(np.abs(vec))):.6g}")
        except Exception as e:
            print(f"   [FHE DEBUG] decrypt attn-output failed: {type(e).__name__}: {e}")

    def forward_one_layer(self, x_enc: BlockMatrix, attention_mask=None) -> BlockMatrix:
        current_level = x_enc.get_level()
        print(f"   [FHE] Input Level: {current_level}")

        mask0, mask1 = self._parse_masks(attention_mask)

        # Q, K full
        q = x_enc.matmul_np_stream(self.np_wq, self.mult_key, level=current_level)  # [Seq,128]
        k = x_enc.matmul_np_stream(self.np_wk, self.mult_key, level=current_level)  # [Seq,128]
        torch.cuda.empty_cache()

        k_t = k.transpose(self.transposition_key)  # [128,Seq]
        del k
        torch.cuda.empty_cache()

        # Slice views for head dims
        q0 = q.slice_cols(0, 64)
        q1 = q.slice_cols(64, 128)
        k0_t = k_t.slice_rows(0, 64)
        k1_t = k_t.slice_rows(64, 128)
        del q, k_t
        torch.cuda.empty_cache()

        if self.debug_enabled:
            print("     > [Attn h0] score -> +4 -> square -> mask0")
            print("     > [Attn h1] score -> +4 -> square -> mask1")

        # Head0 probs
        score0 = q0.matmul(k0_t, self.mult_key)
        del q0, k0_t
        probs0 = score0.add_scalar(4.0).square(self.hadamard_key)
        del score0
        if mask0 is not None:
            probs0 = probs0.mul_plain(mask0)

        # Head1 probs
        score1 = q1.matmul(k1_t, self.mult_key)
        del q1, k1_t
        probs1 = score1.add_scalar(4.0).square(self.hadamard_key)
        del score1
        if mask1 is not None:
            probs1 = probs1.mul_plain(mask1)

        torch.cuda.empty_cache()

        self._dbg_probs(probs0, probs1)

        # V_fused per head (NOTE: no STATIC_SCALE/tau fused anymore)
        v0 = x_enc.matmul_np_stream(self.np_wv_fused_h0, self.mult_key, level=current_level)
        v1 = x_enc.matmul_np_stream(self.np_wv_fused_h1, self.mult_key, level=current_level)
        torch.cuda.empty_cache()

        out0 = probs0.matmul(v0, self.mult_key)
        out1 = probs1.matmul(v1, self.mult_key)
        del probs0, probs1, v0, v1
        torch.cuda.empty_cache()

        output = out0.add(out1)
        del out0, out1
        torch.cuda.empty_cache()

        self._dbg_attn_output(output)

        # Residual + Norm1
        res1 = x_enc.add(output)
        del output
        print(f"     > [Norm1] Bias Addition (Level {res1.get_level()})...")
        norm1 = self._add_bias(res1, self.np_norm1_bias)
        del res1
        torch.cuda.empty_cache()

        # FFN
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
