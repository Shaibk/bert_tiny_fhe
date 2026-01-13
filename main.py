# inference.py
import gc
import torch
import numpy as np
from transformers import AutoTokenizer

import desilofhe as fhe
from src.bert_layers import FHEBertTinyEncoder
from src.block_matrix import BlockMatrix

ID2LABEL = {9: "accept_reservations", 20: "freeze_account"}


def main():
    print("=== TinyBERT FHE Inference (Streamed Encoding, True Multi-Head) ===")

    text = "freeze my account"
    weights_path = "fhe_weights_8level_optimized.npz"

    tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=32)
    input_ids = inputs["input_ids"]

    attn_1d = inputs["attention_mask"].numpy().astype(np.float32)[0]
    real_len = int(attn_1d.sum())
    total_len = int(attn_1d.shape[0])
    print(f"Text: '{text}' | Real Length: {real_len} / {total_len}")

    # 仅用于构造 embedding 输入
    from experiments.accuracy_first.plaintext.model_plain_tinybert import PlainTinyBert
    pt_model = PlainTinyBert(
        vocab_size=30522, max_len=32, hidden=128, layers=2, heads=2, intermediate=512,
        dropout=0.0,
        attn_type="2quad", attn_kwargs={"c": 4.0},
        act="gelu_poly_learnable", act_kwargs={"init_a": 0.02},
        norm_type="bias_only", learnable_tau=True, num_classes=150
    )
    pt_model.load_state_dict(torch.load("experiments/accuracy_first/plaintext/student_8level.pt", map_location="cpu"))
    pt_model.eval()

    with torch.no_grad():
        x_emb = pt_model.embedding(input_ids) + pt_model.pos_embedding[:, :total_len, :]
        x_emb = pt_model.emb_norm(x_emb)
        x_plain_np = x_emb.numpy().astype(np.float32)  # [1,32,128]

    # key-mask only（对齐明文 Attn2Quad）
    mask = np.tile(attn_1d.reshape(1, -1), (total_len, 1)).astype(np.float32)
    print("[Mask] KEY-mask only (tile).")

    # FHE init
    PHYSICAL = 256
    print("Initializing Engine...")
    engine = fhe.GLEngine(shape=(PHYSICAL, 64, 64), mode="gpu")
    sk = engine.create_secret_key()
    keys = {
        "mult": engine.create_matrix_multiplication_key(sk),
        "hadamard": engine.create_hadamard_multiplication_key(sk),
        "trans": engine.create_transposition_key(sk),
    }

    # 输入 tile 到所有 lanes（不要切输入）
    x_packed = np.tile(x_plain_np, (PHYSICAL, 1, 1))
    input_enc = BlockMatrix.encrypt_inputs(engine, x_packed, sk, block_size=64)

    # 两层 encoder（内部会流式 encode 权重）
    bert_l0 = FHEBertTinyEncoder(engine, keys["mult"], keys["hadamard"], keys["trans"], weights_path, layer_idx=0)
    bert_l1 = FHEBertTinyEncoder(engine, keys["mult"], keys["hadamard"], keys["trans"], weights_path, layer_idx=1)

    print("Running Layer 0 (with Mask)...")
    out_l0 = bert_l0.forward_one_layer(input_enc, attention_mask=mask)

    print("Running Layer 1 (with Mask)...")
    out_l1 = bert_l1.forward_one_layer(out_l0, attention_mask=mask)

    # 解密 + stitch：lane0 + lane128（代表 head0/head1）
    print("Decrypting & Stitching (lane sum for heads)...")
    res_00 = engine.decrypt(out_l1.blocks[0][0], sk)
    res_01 = engine.decrypt(out_l1.blocks[0][1], sk)

    lane_h0 = 0
    lane_h1 = 128

    cls_h0 = np.concatenate([res_00[lane_h0, 0, :], res_01[lane_h0, 0, :]]).astype(np.float32)
    cls_h1 = np.concatenate([res_00[lane_h1, 0, :], res_01[lane_h1, 0, :]]).astype(np.float32)
    cls_vec = (cls_h0 + cls_h1).astype(np.float32)

    print(f"   CLS(head0 lane) first5: {cls_h0[:5]}")
    print(f"   CLS(head1 lane) first5: {cls_h1[:5]}")
    print(f"   CLS(sum)        first5: {cls_vec[:5]}")
    print(f"   CLS L2 norm: {float(np.linalg.norm(cls_vec)):.4f} | abs max: {float(np.max(np.abs(cls_vec))):.4f}")

    # classifier
    w_data = np.load(weights_path)
    W = w_data["classifier.weight"]
    b = w_data["classifier.bias"]

    dot = np.dot(cls_vec, W.T)
    logits = dot + b
    pred_id = int(np.argmax(logits))

    print(f"[DEBUG] dot abs max: {float(np.max(np.abs(dot))):.4f}")
    print(f"[DEBUG] bias max: {float(np.max(b)):.4f}, bias argmax: {int(np.argmax(b))}")
    print(f"[DEBUG] final argmax: {pred_id}, logits max: {float(np.max(logits)):.4f}")

    print("\n" + "=" * 30)
    print(f"Prediction: {pred_id}")
    print(f"Label:      {ID2LABEL.get(pred_id, 'Unknown')}")
    print(f"Logits Max: {np.max(logits):.4f}")
    print("=" * 30)

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
