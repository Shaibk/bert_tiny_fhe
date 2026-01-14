import os
import gc
import numpy as np
import torch
from transformers import AutoTokenizer

import desilofhe as fhe
from src.block_matrix import BlockMatrix
from src.bert_layers import FHEBertTinyEncoder

from experiments.accuracy_first.plaintext.model_plain_tinybert import PlainTinyBert
from experiments.accuracy_first.plaintext.data_clinc150 import build_clinc150_dataloaders

# ====== 配置 ======
CKPT_PATH = "experiments/accuracy_first/plaintext/student_kd_plain.pt"
WEIGHTS_NPZ = "fhe_weights_8level_optimized.npz"
MODEL_ID = "google/bert_uncased_L-2_H-128_A-2"

MAX_LEN = 32
HIDDEN = 128
PHYSICAL = 256          # engine batch lanes
BLOCK = 64

STATIC_SCALE = 0.01     # 方案B：scale 放在 mask 里
USE_TRAIN_AS_TEST = True

# 可选：只跑前 N 条做 sanity check；None 表示全跑
LIMIT_SAMPLES = 200
# ==================


def build_keymask_tile(real_len: int, L: int = 32) -> np.ndarray:
    """key-mask only: [L,L], 每行相同，前 real_len 列为 1，其余为 0"""
    m1d = np.zeros((L,), dtype=np.float32)
    m1d[:real_len] = 1.0
    return np.tile(m1d.reshape(1, -1), (L, 1)).astype(np.float32)


@torch.no_grad()
def embed_batch(pt_model: PlainTinyBert, input_ids: torch.Tensor) -> np.ndarray:
    """
    用明文 student 生成 embedding（这是 FHE 的输入）
    input_ids: [B,32]
    return: np [B,32,128]
    """
    x = pt_model.embedding(input_ids) + pt_model.pos_embedding[:, :MAX_LEN, :]
    x = pt_model.emb_norm(x)
    return x.cpu().numpy().astype(np.float32)


def main():
    device = "cpu"  # 这里只用明文模型做 embedding，用 CPU 就行；你也可以改成 cuda
    print("=== FHE Accuracy Eval (Train as Test, bucket by real_len) ===")
    print("CKPT:", CKPT_PATH)
    print("NPZ :", WEIGHTS_NPZ)

    # 1) tokenizer + dataloaders（只为了拿 dataset）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    train_loader, val_loader, num_classes = build_clinc150_dataloaders(
        tokenizer=tokenizer, max_len=MAX_LEN, batch_size=128
    )
    dataset = train_loader.dataset if USE_TRAIN_AS_TEST else val_loader.dataset
    print("Dataset size:", len(dataset))

    # 2) 明文 student（用于 embedding）
    pt_model = PlainTinyBert(
        vocab_size=len(tokenizer), max_len=MAX_LEN, hidden=HIDDEN, layers=2, heads=2,
        intermediate=512, dropout=0.0,
        attn_type="2quad", attn_kwargs={"c": 4.0},
        act="gelu_poly_learnable", act_kwargs={"init_a": 0.02, "init_b": 0.5, "init_d": 0.5},
        norm_type="bias_only", learnable_tau=True, num_classes=num_classes
    ).to(device)
    pt_model.load_state_dict(torch.load(CKPT_PATH, map_location=device), strict=True)
    pt_model.eval()

    # 3) load npz weights（tau + classifier）
    w = np.load(WEIGHTS_NPZ)
    tau_l0 = w["encoder.layer.0.attention.self.tau"].astype(np.float32)  # [2]
    tau_l1 = w["encoder.layer.1.attention.self.tau"].astype(np.float32)  # [2]
    W_cls = w["classifier.weight"].astype(np.float32)  # [C,128]
    b_cls = w["classifier.bias"].astype(np.float32)    # [C]

    # 4) 初始化 FHE engine + keys（只建一次）
    print("Initializing FHE engine...")
    engine = fhe.GLEngine(shape=(PHYSICAL, BLOCK, BLOCK), mode="gpu")
    sk = engine.create_secret_key()
    keys = {
        "mult": engine.create_matrix_multiplication_key(sk),
        "had": engine.create_hadamard_multiplication_key(sk),
        "trans": engine.create_transposition_key(sk),
    }

    enc0 = FHEBertTinyEncoder(engine, keys["mult"], keys["had"], keys["trans"], WEIGHTS_NPZ, layer_idx=0)
    enc1 = FHEBertTinyEncoder(engine, keys["mult"], keys["had"], keys["trans"], WEIGHTS_NPZ, layer_idx=1)

    # 5) 按 real_len 分桶（因为 mask 必须一致）
    buckets = {L: [] for L in range(1, MAX_LEN + 1)}  # real_len 至少 1
    total_to_take = len(dataset) if LIMIT_SAMPLES is None else min(LIMIT_SAMPLES, len(dataset))

    for idx in range(total_to_take):
        item = dataset[idx]  # 期望返回 dict: input_ids, attention_mask, labels
        attn = item["attention_mask"]
        # 兼容 torch/np/list
        if hasattr(attn, "sum"):
            real_len = int(attn.sum().item() if hasattr(attn.sum(), "item") else attn.sum())
        else:
            real_len = int(sum(attn))
        real_len = max(1, min(MAX_LEN, real_len))
        buckets[real_len].append(idx)

    # 6) 评测循环：每个 real_len 一个 mask，一次塞 <=256
    correct = 0
    total = 0

    def run_one_group(indices, real_len: int):
        nonlocal correct, total

        # build per-layer per-head masks (Scheme B): keymask * (0.01/tau_head)
        keymask = build_keymask_tile(real_len, MAX_LEN)  # [32,32]
        mask0_l0 = keymask * (STATIC_SCALE / float(tau_l0[0]))
        mask1_l0 = keymask * (STATIC_SCALE / float(tau_l0[1]))
        mask0_l1 = keymask * (STATIC_SCALE / float(tau_l1[0]))
        mask1_l1 = keymask * (STATIC_SCALE / float(tau_l1[1]))

        # process in chunks of PHYSICAL lanes
        for start in range(0, len(indices), PHYSICAL):
            chunk = indices[start:start + PHYSICAL]
            B = len(chunk)

            # build plaintext batch [B,32] and labels
            input_ids = []
            labels = []
            for j in chunk:
                it = dataset[j]
                input_ids.append(it["input_ids"])
                labels.append(int(it["labels"]))

            input_ids = torch.stack(input_ids, dim=0).to(device).long()  # [B,32]
            labels = np.array(labels, dtype=np.int64)                    # [B]

            # plaintext embedding [B,32,128]
            x_b = embed_batch(pt_model, input_ids)

            # pack into PHYSICAL lanes: [256,32,128]
            x_pack = np.zeros((PHYSICAL, MAX_LEN, HIDDEN), dtype=np.float32)
            x_pack[:B] = x_b

            # encrypt
            x_enc = BlockMatrix.encrypt_inputs(engine, x_pack, sk, block_size=BLOCK)

            # fhe forward
            out0 = enc0.forward_one_layer(x_enc, attention_mask=(mask0_l0, mask1_l0))
            out1 = enc1.forward_one_layer(out0, attention_mask=(mask0_l1, mask1_l1))

            # decrypt CLS for lanes 0..B-1
            r0 = engine.decrypt(out1.blocks[0][0], sk)  # [256,64,64]
            r1 = engine.decrypt(out1.blocks[0][1], sk)

            cls_0 = r0[:B, 0, :]  # [B,64]
            cls_1 = r1[:B, 0, :]  # [B,64]
            cls = np.concatenate([cls_0, cls_1], axis=1).astype(np.float32)  # [B,128]

            # logits + pred
            logits = cls @ W_cls.T + b_cls  # [B,C]
            pred = np.argmax(logits, axis=1)

            correct += int((pred == labels).sum())
            total += B

            # 清理
            del x_enc, out0, out1
            gc.collect()
            torch.cuda.empty_cache()

    print("Start evaluating...")
    for L in range(1, MAX_LEN + 1):
        idxs = buckets[L]
        if not idxs:
            continue
        run_one_group(idxs, L)
        print(f"  done real_len={L:2d} | seen={total} | acc={correct/total:.4f}")

    print("\n===== RESULT =====")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct/total:.6f}")


if __name__ == "__main__":
    main()
