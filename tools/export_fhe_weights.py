import os
import sys
import argparse
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from experiments.accuracy_first.plaintext.model_plain_tinybert import PlainTinyBert
from experiments.accuracy_first.plaintext.dataset_registry import add_dataset_args, normalize_dataset_name
from experiments.accuracy_first.plaintext.artifact_utils import build_ckpt_path, build_weights_path

# ================= 配置区 =================
STATIC_SCALE = 0.01
HIDDEN_SIZE = 128
HEADS = 2
HEAD_DIM = HIDDEN_SIZE // HEADS
LAYERS = 2

ATTN_RESIDUAL_SCALE = 0.2
FFN_RESIDUAL_SCALE = 1.0
GLOBAL_DAMPING = 1.0
# =========================================


def parse_args():
    parser = argparse.ArgumentParser(description="Export FHE weights with dataset suffix.")
    add_dataset_args(parser, default_dataset="clinc150")
    parser.add_argument("--num-classes", type=int, default=None, help="Override num_classes if needed.")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_name = normalize_dataset_name(args.dataset)
    model_path = build_ckpt_path(
        os.path.join(project_root, "experiments/accuracy_first/plaintext"),
        dataset_name,
        args.dataset_version,
        "student_kd_plain",
    )
    output_path = build_weights_path(
        project_root,
        dataset_name,
        args.dataset_version,
        "fhe_weights_8level_optimized",
    )

    print(f"Loading model from {model_path}...")

    if args.num_classes is None:
        if dataset_name == "clinc150":
            num_classes = 150
        else:
            raise ValueError("num_classes is required for non-CLINC datasets.")
    else:
        num_classes = args.num_classes

    model = PlainTinyBert(
        vocab_size=30522, max_len=32, hidden=HIDDEN_SIZE, layers=LAYERS, heads=HEADS,
        intermediate=512, dropout=0.0,
        attn_type="2quad", attn_kwargs={"c": 4.0},
        act="gelu_poly_learnable", act_kwargs={"init_a": 0.02, "init_b": 0.5, "init_d": 0.5},
        norm_type="bias_only", learnable_tau=True, num_classes=num_classes
    )

    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件: {model_path}")
        return

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    fhe_weights = {}

    # 1) Embeddings
    fhe_weights["embeddings.word_embeddings"] = model.embedding.weight.detach().cpu().numpy().astype(np.float32)
    fhe_weights["embeddings.position_embeddings"] = model.pos_embedding.detach().cpu().squeeze(0).numpy().astype(np.float32)
    print("  - Exported embeddings.")

    for i, layer in enumerate(model.layers):
        print(f"Processing Layer {i}...")
        prefix = f"encoder.layer.{i}"

        # ---- Attention weights ----
        tau = layer.attention.tau.detach().cpu().numpy().astype(np.float32)  # [heads]
        print("export layer", i, "tau:", tau)

        # 保存 tau（方案B：推理时用于构造 per-head mask）
        fhe_weights[f"{prefix}.attention.self.tau"] = tau

        W_qkv = layer.attention.qkv.weight.detach().cpu().numpy().astype(np.float32)  # [3H,H] out,in
        W_q = W_qkv[:HIDDEN_SIZE, :]
        W_k = W_qkv[HIDDEN_SIZE:2*HIDDEN_SIZE, :]
        W_v = W_qkv[2*HIDDEN_SIZE:, :]

        W_o = layer.attention.wo.weight.detach().cpu().numpy().astype(np.float32)  # [H,H] out,in

        # 2D query/key
        fhe_weights[f"{prefix}.attention.self.query.weight"] = W_q
        fhe_weights[f"{prefix}.attention.self.key.weight"] = W_k

        # ✅ 方案B：W_fused 只融合 (W_o slice @ W_v slice) 和 residual scale 0.2
        # ❌ 不再把 (STATIC_SCALE/tau) 融进 W_fused（改由 per-head mask 处理）
        sl0 = slice(0, HEAD_DIM)
        sl1 = slice(HEAD_DIM, HIDDEN_SIZE)

        W_fused_0 = (W_o[:, sl0] @ W_v[sl0, :]) * ATTN_RESIDUAL_SCALE * GLOBAL_DAMPING
        W_fused_1 = (W_o[:, sl1] @ W_v[sl1, :]) * ATTN_RESIDUAL_SCALE * GLOBAL_DAMPING

        print("W_fused_0 abs mean/max:", np.mean(np.abs(W_fused_0)), np.max(np.abs(W_fused_0)))
        print("W_fused_1 abs mean/max:", np.mean(np.abs(W_fused_1)), np.max(np.abs(W_fused_1)))

        fhe_weights[f"{prefix}.attention.self.value_fused_head0.weight"] = W_fused_0.astype(np.float32)
        fhe_weights[f"{prefix}.attention.self.value_fused_head1.weight"] = W_fused_1.astype(np.float32)

        # Norm1 bias（bias-only，在 residual 之后加；不要乘 residual scale）
        if hasattr(layer, "norm1"):
            fhe_weights[f"{prefix}.attention.output.norm.bias"] = (
                layer.norm1.bias.detach().cpu().numpy().astype(np.float32) * GLOBAL_DAMPING
            )

        # ---- FFN ----
        fhe_weights[f"{prefix}.intermediate.dense.weight"] = layer.linear1.weight.detach().cpu().numpy().astype(np.float32)

        a = layer.activation.a.detach().cpu().numpy().astype(np.float32)
        b = layer.activation.b.detach().cpu().numpy().astype(np.float32)
        d = layer.activation.d.detach().cpu().numpy().astype(np.float32)

        W2_raw = layer.linear2.weight.detach().cpu().numpy().astype(np.float32)  # [out=128, in=512]
        W2_eff = W2_raw * FFN_RESIDUAL_SCALE * GLOBAL_DAMPING

        W2_quad = W2_eff * a
        W2_lin  = W2_eff * b
        d_effect = d * np.sum(W2_eff, axis=1)

        fhe_weights[f"{prefix}.output.dense.weight_quad"] = W2_quad
        fhe_weights[f"{prefix}.output.dense.weight_lin"] = W2_lin
        fhe_weights[f"{prefix}.output.dense.bias_fused"] = d_effect.astype(np.float32)

        if hasattr(layer, "norm2"):
            fhe_weights[f"{prefix}.output.norm.bias"] = (
                layer.norm2.bias.detach().cpu().numpy().astype(np.float32) * GLOBAL_DAMPING
            )

    # Classifier
    fhe_weights["classifier.weight"] = model.classifier.weight.detach().cpu().numpy().astype(np.float32)
    fhe_weights["classifier.bias"] = model.classifier.bias.detach().cpu().numpy().astype(np.float32)

    np.savez(output_path, **fhe_weights)
    print(f"\n✅ Export OK -> {output_path}")
    print(f"   ATTN_RESIDUAL_SCALE={ATTN_RESIDUAL_SCALE}, FFN_RESIDUAL_SCALE={FFN_RESIDUAL_SCALE}, STATIC_SCALE={STATIC_SCALE}, DAMPING={GLOBAL_DAMPING}")
    print("   NOTE: STATIC_SCALE/tau is NOT fused into W_fused; it must be applied via per-head mask in inference.")


if __name__ == "__main__":
    main()
