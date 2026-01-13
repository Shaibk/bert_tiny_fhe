import os
import sys
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from experiments.accuracy_first.plaintext.model_plain_tinybert import PlainTinyBert

# ================= 配置区 =================
STATIC_SCALE = 0.01
HIDDEN_SIZE = 128
HEADS = 2
LAYERS = 2

# ✅ 区分 Attention / FFN 的残差缩放
ATTN_RESIDUAL_SCALE = 0.2   # Attention 子层：x + 0.2 * AttnOut
FFN_RESIDUAL_SCALE  = 1.0   # FFN 子层：x + 1.0 * FfnOut

# 额外阻尼（目前不使用）
GLOBAL_DAMPING = 1.0
# =========================================


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "experiments/accuracy_first/plaintext/student_8level.pt")
    output_path = os.path.join(project_root, "fhe_weights_8level_optimized.npz")

    print(f"Loading optimized model from {model_path}...")

    model = PlainTinyBert(
        vocab_size=30522, max_len=32, hidden=HIDDEN_SIZE, layers=LAYERS, heads=HEADS,
        intermediate=512, dropout=0.0,
        attn_type="2quad", attn_kwargs={"c": 4.0},
        act="gelu_poly_learnable", act_kwargs={"init_a": 0.02, "init_b": 0.5, "init_d": 0.5},
        norm_type="bias_only", learnable_tau=True, num_classes=150
    )

    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型文件 {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    fhe_weights = {}

    # 1) Embeddings
    fhe_weights["embeddings.word_embeddings"] = model.embedding.weight.detach().cpu().numpy()
    fhe_weights["embeddings.position_embeddings"] = model.pos_embedding.detach().cpu().squeeze(0).numpy()
    print("  - Exported Embeddings.")

    for i, layer in enumerate(model.layers):
        print(f"Processing Layer {i}...")
        prefix = f"encoder.layer.{i}"

        # --- A) Attention ---
        tau = layer.attention.tau.detach().cpu().numpy()            # [heads]
        W_qkv = layer.attention.qkv.weight.detach().cpu().numpy()   # [3H, H] (out,in)
        W_o_mat = layer.attention.wo.weight.detach().cpu().numpy().T  # [H, H]  (transpose for your fuse math)

        W_q = W_qkv[:128, :]        # [128,128] out,in
        W_k = W_qkv[128:256, :]     # [128,128] out,in
        W_v = W_qkv[256:, :]        # [128,128] out,in

        # ✅ 关键：Wq/Wk 也做 head-stacked (按输出通道切 head)
        # 生成 [Heads, Out(128), In(128)]
        W_q_stacked = np.zeros((HEADS, HIDDEN_SIZE, HIDDEN_SIZE), dtype=np.float32)
        W_k_stacked = np.zeros((HEADS, HIDDEN_SIZE, HIDDEN_SIZE), dtype=np.float32)

        # head0: out 0:64
        W_q_stacked[0, :64, :] = W_q[:64, :]
        W_k_stacked[0, :64, :] = W_k[:64, :]

        # head1: out 64:128
        W_q_stacked[1, 64:, :] = W_q[64:, :]
        W_k_stacked[1, 64:, :] = W_k[64:, :]

        fhe_weights[f"{prefix}.attention.self.query.weight"] = W_q_stacked
        fhe_weights[f"{prefix}.attention.self.key.weight"] = W_k_stacked

        # ✅ V@O 融合（保持你现有逻辑）：
        # 先按 head 的 V 子矩阵与对应 O 子矩阵融合，再乘 residual scale，再乘 (static_scale/tau)
        W_v_T = W_v.T  # [in,out] = [128,128]

        # 每个 head 只负责自己的 64 维子空间（按输出通道 0:64 / 64:128 切）
        W_fused_1 = np.matmul(W_v_T[:, :64], W_o_mat[:64, :]) * ATTN_RESIDUAL_SCALE
        W_fused_2 = np.matmul(W_v_T[:, 64:], W_o_mat[64:, :]) * ATTN_RESIDUAL_SCALE

        scale_1 = (STATIC_SCALE / tau[0]) * GLOBAL_DAMPING
        scale_2 = (STATIC_SCALE / tau[1]) * GLOBAL_DAMPING
        W_fused_1 *= scale_1
        W_fused_2 *= scale_2

        # 保存为 [Heads, Out, In]（out,in）
        W_final_stacked = np.stack([W_fused_1.T, W_fused_2.T]).astype(np.float32)  # [2,128,128]
        fhe_weights[f"{prefix}.attention.self.value_fused.weight"] = W_final_stacked

        # ✅ Norm1 bias：不要乘 residual scale（bias 是 residual 后加）
        if hasattr(layer, "norm1"):
            fhe_weights[f"{prefix}.attention.output.norm.bias"] = \
                layer.norm1.bias.detach().cpu().numpy().astype(np.float32) * GLOBAL_DAMPING

        # --- B) FFN Linear1 ---
        fhe_weights[f"{prefix}.intermediate.dense.weight"] = layer.linear1.weight.detach().cpu().numpy().astype(np.float32)

        # --- C) FFN Linear2 (PolyGELU) ---
        a = layer.activation.a.detach().cpu().numpy().astype(np.float32)
        b = layer.activation.b.detach().cpu().numpy().astype(np.float32)
        d = layer.activation.d.detach().cpu().numpy().astype(np.float32)

        W2_raw = layer.linear2.weight.detach().cpu().numpy().astype(np.float32)  # [out=128, in=512] (out,in)

        # ✅ FFN residual scale（按你明文：不乘 0.2）
        W2_effective = W2_raw * FFN_RESIDUAL_SCALE * GLOBAL_DAMPING

        W2_quad = W2_effective * a
        W2_lin  = W2_effective * b
        d_effect = d * np.sum(W2_effective, axis=1)

        fhe_weights[f"{prefix}.output.dense.weight_quad"] = W2_quad
        fhe_weights[f"{prefix}.output.dense.weight_lin"] = W2_lin
        fhe_weights[f"{prefix}.output.dense.bias_fused"] = d_effect.astype(np.float32)

        # ✅ Norm2 bias：不要乘 residual scale
        if hasattr(layer, "norm2"):
            fhe_weights[f"{prefix}.output.norm.bias"] = \
                layer.norm2.bias.detach().cpu().numpy().astype(np.float32) * GLOBAL_DAMPING

    # 4) Classifier
    fhe_weights["classifier.weight"] = model.classifier.weight.detach().cpu().numpy().astype(np.float32)
    fhe_weights["classifier.bias"] = model.classifier.bias.detach().cpu().numpy().astype(np.float32)

    np.savez(output_path, **fhe_weights)
    print(f"\n✅ Export Success! ATTN_RESIDUAL_SCALE={ATTN_RESIDUAL_SCALE}, FFN_RESIDUAL_SCALE={FFN_RESIDUAL_SCALE}, DAMPING={GLOBAL_DAMPING}")
    print(f"   File saved to: {output_path}")


if __name__ == "__main__":
    main()
