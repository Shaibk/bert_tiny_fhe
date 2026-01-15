import os
import gc
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer

import desilofhe as fhe
from src.bert_layers import FHEBertTinyEncoder
from src.block_matrix import BlockMatrix

ID2LABEL = {9: "accept_reservations", 20: "freeze_account"}

STATIC_SCALE = 0.01


def build_keymask_tile(attn_1d: np.ndarray) -> np.ndarray:
    """
    attn_1d: [L] values 0/1
    return:  [L,L] tiled key-mask (multiplicative key-mask)
    """
    L = attn_1d.shape[0]
    return np.tile(attn_1d.reshape(1, -1), (L, 1)).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="TinyBERT FHE inference.")
    parser.add_argument(
        "--texts",
        nargs="+",
        default=[
            "freeze my account",
            "can you accept reservations",
            "freeze account now",
            "i want to reserve a table",
        ],
    )
    from experiments.accuracy_first.plaintext.dataset_registry import add_dataset_args
    add_dataset_args(parser, default_dataset="clinc150")
    return parser.parse_args()


def main():
    print("=== TinyBERT FHE Inference (GL+Desilo | per-lane inputs & per-lane masks | Scheme B) ===")
    args = parse_args()
    from experiments.accuracy_first.plaintext.dataset_registry import normalize_dataset_name
    from experiments.accuracy_first.plaintext.artifact_utils import build_ckpt_path, build_weights_path

    # --------
    # Inputs: multiple texts (<=256)
    # --------
    texts = args.texts

    dataset_name = normalize_dataset_name(args.dataset)
    if dataset_name != "clinc150":
        ID2LABEL.clear()

    tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
    project_root = os.path.dirname(os.path.abspath(__file__))
    weights_path = build_weights_path(
        project_root,
        dataset_name,
        args.dataset_version,
        "fhe_weights_8level_optimized",
    )

    # --------
    # Plain model only for embeddings
    # --------
    from experiments.accuracy_first.plaintext.model_plain_tinybert import PlainTinyBert
    pt_model = PlainTinyBert(
        vocab_size=30522, max_len=32, hidden=128, layers=2, heads=2, intermediate=512,
        dropout=0.0,
        attn_type="2quad", attn_kwargs={"c": 4.0},
        act="gelu_poly_learnable", act_kwargs={"init_a": 0.02},
        norm_type="bias_only", learnable_tau=True, num_classes=150
    )
    ckpt_path = build_ckpt_path(
        os.path.join(project_root, "experiments/accuracy_first/plaintext"),
        dataset_name,
        args.dataset_version,
        "student_kd_plain",
    )
    pt_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    pt_model.eval()

    # --------
    # Load tau from npz
    # --------
    w_data = np.load(weights_path)
    tau_l0 = w_data["encoder.layer.0.attention.self.tau"].astype(np.float32)  # [2]
    tau_l1 = w_data["encoder.layer.1.attention.self.tau"].astype(np.float32)  # [2]

    # --------
    # FHE init
    # --------
    PHYSICAL = 256
    B = PHYSICAL

    print("Initializing Engine...")
    engine = fhe.GLEngine(shape=(PHYSICAL, 64, 64), mode="gpu")
    sk = engine.create_secret_key()
    keys = {
        "mult": engine.create_matrix_multiplication_key(sk),
        "hadamard": engine.create_hadamard_multiplication_key(sk),
        "trans": engine.create_transposition_key(sk),
    }

    # --------
    # Build per-lane embeddings + per-lane masks
    # --------
    max_len = 32
    N = len(texts)
    if N > PHYSICAL:
        raise ValueError(f"Too many inputs: {N} > PHYSICAL={PHYSICAL}")

    # x_packed: [B, 32, 128]
    x_packed = np.zeros((B, max_len, 128), dtype=np.float32)

    # masks: [B, 32, 32]
    mask0_l0 = np.zeros((B, max_len, max_len), dtype=np.float32)
    mask1_l0 = np.zeros((B, max_len, max_len), dtype=np.float32)
    mask0_l1 = np.zeros((B, max_len, max_len), dtype=np.float32)
    mask1_l1 = np.zeros((B, max_len, max_len), dtype=np.float32)

    print(f"Packing {N} inputs into {PHYSICAL} lanes (remaining lanes padded as zeros).")

    for lane in range(N):
        text = texts[lane]
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=max_len, truncation=True)
        input_ids = inputs["input_ids"]
        attn_1d = inputs["attention_mask"].numpy().astype(np.float32)[0]
        real_len = int(attn_1d.sum())

        with torch.no_grad():
            x_emb = pt_model.embedding(input_ids) + pt_model.pos_embedding[:, :max_len, :]
            x_emb = pt_model.emb_norm(x_emb)
            x_np = x_emb.numpy().astype(np.float32)[0]  # [32,128]

        x_packed[lane] = x_np

        keymask = build_keymask_tile(attn_1d)  # [L,L] values 0/1

        # Scheme B: per-head scaled masks: keymask * (0.01 / tau_h)
        s00 = STATIC_SCALE / float(tau_l0[0])
        s01 = STATIC_SCALE / float(tau_l0[1])
        s10 = STATIC_SCALE / float(tau_l1[0])
        s11 = STATIC_SCALE / float(tau_l1[1])

        mask0_l0[lane] = keymask * s00
        mask1_l0[lane] = keymask * s01
        mask0_l1[lane] = keymask * s10
        mask1_l1[lane] = keymask * s11

        print(f"  lane={lane:03d} | text='{text}' | real_len={real_len}")

    if os.getenv("FHE_DEBUG", "0") == "1":
        print("[Mask] per-lane per-head mask enabled (3D [B,L,L]).")
        print(f"   tau layer0: {tau_l0}, scales: {[STATIC_SCALE/float(tau_l0[0]), STATIC_SCALE/float(tau_l0[1])]}")
        print(f"   tau layer1: {tau_l1}, scales: {[STATIC_SCALE/float(tau_l1[0]), STATIC_SCALE/float(tau_l1[1])]}")
        # show lane0 sanity
        if N > 0:
            nz0 = float(mask0_l0[0, 0, 0])
            print(f"   lane0 mask0_l0 nonzero value sample: {nz0:.6g}")

    # --------
    # Encrypt inputs (now truly per-lane different)
    # --------
    input_enc = BlockMatrix.encrypt_inputs(engine, x_packed, sk, block_size=64)

    bert_l0 = FHEBertTinyEncoder(engine, keys["mult"], keys["hadamard"], keys["trans"], weights_path, layer_idx=0)
    bert_l1 = FHEBertTinyEncoder(engine, keys["mult"], keys["hadamard"], keys["trans"], weights_path, layer_idx=1)

    # Enable decrypt-based debug only if FHE_DEBUG=1
    if os.getenv("FHE_DEBUG", "0") == "1":
        bert_l0.debug_sk = sk
        bert_l1.debug_sk = sk

    # --------
    # Run FHE
    # --------
    print("Running Layer 0...")
    out_l0 = bert_l0.forward_one_layer(input_enc, attention_mask=(mask0_l0, mask1_l0))

    print("Running Layer 1...")
    out_l1 = bert_l1.forward_one_layer(out_l0, attention_mask=(mask0_l1, mask1_l1))

    # --------
    # Decrypt CLS for ALL lanes (we will only use first N)
    # --------
    print(f"Decrypting CLS for lanes [0..{N-1}] ...")
    res_00 = engine.decrypt(out_l1.blocks[0][0], sk)  # [B,64,64]
    res_01 = engine.decrypt(out_l1.blocks[0][1], sk)  # [B,64,64]

    # --------
    # Classifier on CPU (per sample)
    # --------
    W = w_data["classifier.weight"]  # [150,128]
    b = w_data["classifier.bias"]    # [150]

    preds = []
    for lane in range(N):
        cls_vec = np.concatenate([res_00[lane, 0, :], res_01[lane, 0, :]]).astype(np.float32)  # [128]
        logits = np.dot(cls_vec, W.T) + b
        pred_id = int(np.argmax(logits))
        preds.append(pred_id)

        print("\n" + "-" * 48)
        print(f"lane={lane:03d} | text='{texts[lane]}'")
        print(f"  pred_id: {pred_id}")
        print(f"  label:   {ID2LABEL.get(pred_id, 'Unknown')}")
        print(f"  logits_max: {float(np.max(logits)):.4f}")
        print("-" * 48)

    print("\n" + "=" * 60)
    print("Pred IDs:", preds)
    print("=" * 60)

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
