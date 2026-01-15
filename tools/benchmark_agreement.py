import os
import sys
import numpy as np
import torch
from transformers import AutoTokenizer
import desilofhe as fhe
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from experiments.accuracy_first.plaintext.model_plain_tinybert import PlainTinyBert
from experiments.accuracy_first.plaintext.dataset_registry import add_dataset_args, normalize_dataset_name
from experiments.accuracy_first.plaintext.artifact_utils import build_ckpt_path, build_weights_path
from src.bert_layers import FHEBertTinyEncoder
from src.block_matrix import BlockMatrix

MODEL_ID = "google/bert_uncased_L-2_H-128_A-2"
CKPT_PREFIX = "student_kd_plain"
WEIGHTS_PREFIX = "fhe_weights_8level_optimized"

MAX_LEN = 32
HIDDEN = 128
PHYSICAL = 256
BLOCK = 64
STATIC_SCALE = 0.01

TEST_SENTENCES = [
    "freeze my account",
    "can you accept reservations",
    "what is your name",
    "transfer money to mom",
    "book a table for two",
]


def build_keymask_tile(attn_1d: np.ndarray) -> np.ndarray:
    L = attn_1d.shape[0]
    return np.tile(attn_1d.reshape(1, -1), (L, 1)).astype(np.float32)


def load_plain_model(vocab_size: int, num_classes: int, device: str, ckpt_path: str):
    model = PlainTinyBert(
        vocab_size=vocab_size, max_len=MAX_LEN, hidden=HIDDEN, layers=2, heads=2,
        intermediate=512, dropout=0.0,
        attn_type="2quad", attn_kwargs={"c": 4.0},
        act="gelu_poly_learnable", act_kwargs={"init_a": 0.02, "init_b": 0.5, "init_d": 0.5},
        norm_type="bias_only", learnable_tau=True, num_classes=num_classes,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Agreement check between plaintext and FHE.")
    add_dataset_args(parser, default_dataset="clinc150")
    args = parser.parse_args()

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    dataset_name = normalize_dataset_name(args.dataset)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_path = build_ckpt_path(
        os.path.join(project_root, "experiments/accuracy_first/plaintext"),
        dataset_name,
        args.dataset_version,
        CKPT_PREFIX,
    )
    weights_path = build_weights_path(project_root, dataset_name, args.dataset_version, WEIGHTS_PREFIX)

    device = "cpu"
    print("=== Agreement Check: Plain vs FHE (5 samples) ===")
    print("CKPT:", ckpt_path)
    print("NPZ :", weights_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, local_files_only=True)

    inputs = tokenizer(
        TEST_SENTENCES,
        return_tensors="pt",
        padding="max_length",
        max_length=MAX_LEN,
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    w = np.load(weights_path)
    tau_l0 = w["encoder.layer.0.attention.self.tau"].astype(np.float32)
    tau_l1 = w["encoder.layer.1.attention.self.tau"].astype(np.float32)
    W_cls = w["classifier.weight"].astype(np.float32)
    b_cls = w["classifier.bias"].astype(np.float32)

    pt_model = load_plain_model(len(tokenizer), W_cls.shape[0], device, ckpt_path)

    with torch.no_grad():
        pt_logits = pt_model(input_ids, attention_mask=attention_mask)["logits"].cpu().numpy()
        pt_pred = np.argmax(pt_logits, axis=1)

        x_emb = pt_model.embedding(input_ids) + pt_model.pos_embedding[:, :MAX_LEN, :]
        x_emb = pt_model.emb_norm(x_emb)
        x_plain = x_emb.cpu().numpy().astype(np.float32)

    # pack to PHYSICAL lanes
    B = x_plain.shape[0]
    x_pack = np.zeros((PHYSICAL, MAX_LEN, HIDDEN), dtype=np.float32)
    x_pack[:B] = x_plain

    s00 = STATIC_SCALE / float(tau_l0[0])
    s01 = STATIC_SCALE / float(tau_l0[1])
    s10 = STATIC_SCALE / float(tau_l1[0])
    s11 = STATIC_SCALE / float(tau_l1[1])

    mask0_l0 = np.zeros((PHYSICAL, MAX_LEN, MAX_LEN), dtype=np.float32)
    mask1_l0 = np.zeros((PHYSICAL, MAX_LEN, MAX_LEN), dtype=np.float32)
    mask0_l1 = np.zeros((PHYSICAL, MAX_LEN, MAX_LEN), dtype=np.float32)
    mask1_l1 = np.zeros((PHYSICAL, MAX_LEN, MAX_LEN), dtype=np.float32)

    for lane in range(B):
        attn_1d = attention_mask[lane].cpu().numpy().astype(np.float32)
        keymask = build_keymask_tile(attn_1d)
        mask0_l0[lane] = keymask * s00
        mask1_l0[lane] = keymask * s01
        mask0_l1[lane] = keymask * s10
        mask1_l1[lane] = keymask * s11

    print("Initializing FHE engine...")
    engine = fhe.GLEngine(shape=(PHYSICAL, BLOCK, BLOCK), mode="gpu")
    sk = engine.create_secret_key()
    keys = {
        "mult": engine.create_matrix_multiplication_key(sk),
        "had": engine.create_hadamard_multiplication_key(sk),
        "trans": engine.create_transposition_key(sk),
    }

    enc0 = FHEBertTinyEncoder(engine, keys["mult"], keys["had"], keys["trans"], weights_path, layer_idx=0)
    enc1 = FHEBertTinyEncoder(engine, keys["mult"], keys["had"], keys["trans"], weights_path, layer_idx=1)

    x_enc = BlockMatrix.encrypt_inputs(engine, x_pack, sk, block_size=BLOCK)
    out0 = enc0.forward_one_layer(x_enc, attention_mask=(mask0_l0, mask1_l0))
    out1 = enc1.forward_one_layer(out0, attention_mask=(mask0_l1, mask1_l1))

    r0 = engine.decrypt(out1.blocks[0][0], sk)
    r1 = engine.decrypt(out1.blocks[0][1], sk)

    fhe_pred = []
    for lane in range(B):
        cls_0 = r0[lane, 0, :]
        cls_1 = r1[lane, 0, :]
        cls = np.concatenate([cls_0, cls_1], axis=0).astype(np.float32)
        logits = cls @ W_cls.T + b_cls
        fhe_pred.append(int(np.argmax(logits)))

    print("\nidx | sentence                     | plain | fhe | match")
    print("-" * 65)
    matches = 0
    for i, text in enumerate(TEST_SENTENCES):
        p = int(pt_pred[i])
        f = int(fhe_pred[i])
        ok = p == f
        matches += int(ok)
        short = (text[:24] + "..") if len(text) > 24 else text
        print(f"{i:>3} | {short:<27} | {p:>5} | {f:>3} | {str(ok)}")

    print("-" * 65)
    print(f"Agreement: {matches}/{B} = {matches/B:.3f}")


if __name__ == "__main__":
    main()
