import os
import gc
import torch
import numpy as np
from transformers import AutoTokenizer

import desilofhe as fhe
from src.bert_layers import FHEBertTinyEncoder
from src.block_matrix import BlockMatrix

ID2LABEL = {9: "accept_reservations", 20: "freeze_account"}

STATIC_SCALE = 0.01


def build_keymask_tile(attn_1d: np.ndarray) -> np.ndarray:
    L = attn_1d.shape[0]
    return np.tile(attn_1d.reshape(1, -1), (L, 1)).astype(np.float32)  # [L,L]


def main():
    print("=== TinyBERT FHE Inference (Scheme B: scale in per-head mask; debug-ready) ===")

    text = "freeze my account"
    weights_path = "fhe_weights_8level_optimized.npz"

    tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=32)
    input_ids = inputs["input_ids"]

    attn_1d = inputs["attention_mask"].numpy().astype(np.float32)[0]
    real_len = int(attn_1d.sum())
    total_len = int(attn_1d.shape[0])
    print(f"Text: '{text}' | Real Length: {real_len} / {total_len}")

    # only for embeddings
    from experiments.accuracy_first.plaintext.model_plain_tinybert import PlainTinyBert
    pt_model = PlainTinyBert(
        vocab_size=30522, max_len=32, hidden=128, layers=2, heads=2, intermediate=512,
        dropout=0.0,
        attn_type="2quad", attn_kwargs={"c": 4.0},
        act="gelu_poly_learnable", act_kwargs={"init_a": 0.02},
        norm_type="bias_only", learnable_tau=True, num_classes=150
    )
    # 注意：这里你按自己当前用的 ckpt 路径
    pt_model.load_state_dict(torch.load("experiments/accuracy_first/plaintext/student_kd_plain.pt", map_location="cpu"))
    pt_model.eval()

    with torch.no_grad():
        x_emb = pt_model.embedding(input_ids) + pt_model.pos_embedding[:, :total_len, :]
        x_emb = pt_model.emb_norm(x_emb)
        x_plain_np = x_emb.numpy().astype(np.float32)  # [1,32,128]

    # Load tau from npz and build per-head masks for each layer
    w_data = np.load(weights_path)
    tau0 = w_data["encoder.layer.0.attention.self.tau"].astype(np.float32)  # [2]
    tau1 = w_data["encoder.layer.1.attention.self.tau"].astype(np.float32)  # [2]

    keymask = build_keymask_tile(attn_1d)  # [L,L], values 0/1

    # per-head scaled masks (Scheme B): mask_h = keymask * (0.01/tau_h)
    mask0_l0 = keymask * (STATIC_SCALE / float(tau0[0]))
    mask1_l0 = keymask * (STATIC_SCALE / float(tau0[1]))
    mask0_l1 = keymask * (STATIC_SCALE / float(tau1[0]))
    mask1_l1 = keymask * (STATIC_SCALE / float(tau1[1]))

    print("[Mask] KEY-mask tile with per-head scale (0.01/tau).")
    if os.getenv("FHE_DEBUG", "0") == "1":
        print(f"   tau layer0: {tau0}, scale: {[STATIC_SCALE/float(tau0[0]), STATIC_SCALE/float(tau0[1])]}")
        print(f"   tau layer1: {tau1}, scale: {[STATIC_SCALE/float(tau1[0]), STATIC_SCALE/float(tau1[1])]}")
        print(f"   mask0_l0 nonzero value: {mask0_l0[0,0]:.6g}")

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

    # batch lanes: replicate same sample
    x_packed = np.tile(x_plain_np, (PHYSICAL, 1, 1))
    input_enc = BlockMatrix.encrypt_inputs(engine, x_packed, sk, block_size=64)

    bert_l0 = FHEBertTinyEncoder(engine, keys["mult"], keys["hadamard"], keys["trans"], weights_path, layer_idx=0)
    bert_l1 = FHEBertTinyEncoder(engine, keys["mult"], keys["hadamard"], keys["trans"], weights_path, layer_idx=1)

    # Enable decrypt-based debug only if FHE_DEBUG=1
    if os.getenv("FHE_DEBUG", "0") == "1":
        bert_l0.debug_sk = sk
        bert_l1.debug_sk = sk
        bert_l0.debug_real_len = real_len
        bert_l1.debug_real_len = real_len

    print("Running Layer 0...")
    out_l0 = bert_l0.forward_one_layer(input_enc, attention_mask=(mask0_l0, mask1_l0))

    print("Running Layer 1...")
    out_l1 = bert_l1.forward_one_layer(out_l0, attention_mask=(mask0_l1, mask1_l1))

    # decrypt CLS from lane0 only
    print("Decrypting CLS (lane0)...")
    res_00 = engine.decrypt(out_l1.blocks[0][0], sk)
    res_01 = engine.decrypt(out_l1.blocks[0][1], sk)

    lane = 0
    cls_vec = np.concatenate([res_00[lane, 0, :], res_01[lane, 0, :]]).astype(np.float32)

    print(f"   CLS first5: {cls_vec[:5]}")
    print(f"   CLS L2 norm: {float(np.linalg.norm(cls_vec)):.4f} | abs max: {float(np.max(np.abs(cls_vec))):.4f}")

    # classifier
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
