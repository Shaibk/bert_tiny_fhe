import os
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import gc
import time
import sys
import numpy as np
import torch
from transformers import AutoTokenizer
import argparse

import desilofhe as fhe
from src.block_matrix import BlockMatrix
from src.bert_layers import FHEBertTinyEncoder

from experiments.accuracy_first.plaintext.model_plain_tinybert import PlainTinyBert
from experiments.accuracy_first.plaintext.dataset_registry import (
    add_dataset_args,
    build_dataloaders_with_test,
    normalize_dataset_name,
)
from experiments.accuracy_first.plaintext.artifact_utils import build_ckpt_path, build_weights_path

# ====== 配置 ======
CKPT_PREFIX = "student_kd_plain"
WEIGHTS_PREFIX = "fhe_weights_8level_optimized"
MODEL_ID = "google/bert_uncased_L-2_H-128_A-2"

MAX_LEN = 32
HIDDEN = 128
PHYSICAL = 256          # engine batch lanes
BLOCK = 64

STATIC_SCALE = 0.01     # 方案B：scale 放在 mask 里
DATA_SPLIT = "test"     # train/validation/test

# 可选：只跑前 N 条做 sanity check；None 表示全跑
LIMIT_SAMPLES = None
DEBUG_FIRST_BATCH = True
# ==================


def build_keymask_tile(attn_1d: np.ndarray) -> np.ndarray:
    """key-mask only: [L,L], 每行相同，0/1 来自 attention_mask"""
    L = attn_1d.shape[0]
    return np.tile(attn_1d.reshape(1, -1), (L, 1)).astype(np.float32)


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


@torch.no_grad()
def eval_plain_accuracy(pt_model: PlainTinyBert, loader, device: str):
    correct = 0
    total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = pt_model(input_ids, attention_mask=attention_mask)["logits"]
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.size(0))
    return correct, total


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Plain/FHE accuracy with dataset sharding.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use (sets CUDA_VISIBLE_DEVICES).")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards.")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index [0..num_shards-1].")
    add_dataset_args(parser, default_dataset="clinc150")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("shard_id must be in [0, num_shards-1]")

    dataset_name = normalize_dataset_name(args.dataset)
    project_root = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = build_ckpt_path(
        os.path.join(project_root, "experiments/accuracy_first/plaintext"),
        dataset_name,
        args.dataset_version,
        CKPT_PREFIX,
    )
    weights_path = build_weights_path(project_root, dataset_name, args.dataset_version, WEIGHTS_PREFIX)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cpu"  # 这里只用明文模型做 embedding，用 CPU 就行；你也可以改成 cuda
    print(f"=== FHE Accuracy Eval ({dataset_name}, shared label mapping) ===")
    print("CKPT:", ckpt_path)
    print("NPZ :", weights_path)
    print(f"Shard: {args.shard_id}/{args.num_shards} | GPU: {args.gpu}")

    # 1) tokenizer + dataset
    print("Loading tokenizer (offline)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, local_files_only=True)
    print("Loading dataset via registry (offline)...")
    train_loader, val_loader, test_loader, num_classes = build_dataloaders_with_test(
        dataset_name,
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        batch_size=128,
        dataset_config=args.dataset_config,
        dataset_source=args.dataset_source,
    )
    if DATA_SPLIT == "train":
        dataset = train_loader.dataset
    elif DATA_SPLIT == "validation":
        dataset = val_loader.dataset
    else:
        if test_loader is None:
            raise ValueError("Requested test split, but dataset has no test split.")
        dataset = test_loader.dataset
    print("Dataset split:", DATA_SPLIT)
    print("Dataset size:", len(dataset))
    sample_n = min(2000, len(dataset))
    sample_labels = [int(dataset[i]["labels"]) for i in range(sample_n)]
    print(
        f"Label sample (n={sample_n}): min={min(sample_labels)}, "
        f"max={max(sample_labels)}, uniq={len(set(sample_labels))}"
    )

    # 2) 明文 student（用于 embedding）
    pt_model = PlainTinyBert(
        vocab_size=len(tokenizer), max_len=MAX_LEN, hidden=HIDDEN, layers=2, heads=2,
        intermediate=512, dropout=0.0,
        attn_type="2quad", attn_kwargs={"c": 4.0},
        act="gelu_poly_learnable", act_kwargs={"init_a": 0.02, "init_b": 0.5, "init_d": 0.5},
        norm_type="bias_only", learnable_tau=True, num_classes=num_classes
    ).to(device)
    pt_model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
    pt_model.eval()

    # 3) 明文 accuracy
    plain_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    pt_correct, pt_total = eval_plain_accuracy(pt_model, plain_loader, device)
    print(f"Plaintext accuracy: {pt_correct}/{pt_total} = {pt_correct/pt_total:.6f}")

    # 4) load npz weights（tau + classifier）
    w = np.load(weights_path)
    tau_l0 = w["encoder.layer.0.attention.self.tau"].astype(np.float32)  # [2]
    tau_l1 = w["encoder.layer.1.attention.self.tau"].astype(np.float32)  # [2]
    W_cls = w["classifier.weight"].astype(np.float32)  # [C,128]
    b_cls = w["classifier.bias"].astype(np.float32)    # [C]

    # 5) 初始化 FHE engine + keys（只建一次）
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

    total_to_take = len(dataset) if LIMIT_SAMPLES is None else min(LIMIT_SAMPLES, len(dataset))
    all_indices = list(range(total_to_take))
    shard_indices = all_indices[args.shard_id::args.num_shards]
    total_chunks = (len(shard_indices) + PHYSICAL - 1) // PHYSICAL

    # 6) FHE 评测循环：每次塞 <=256，per-lane mask
    correct = 0
    total = 0

    s00 = STATIC_SCALE / float(tau_l0[0])
    s01 = STATIC_SCALE / float(tau_l0[1])
    s10 = STATIC_SCALE / float(tau_l1[0])
    s11 = STATIC_SCALE / float(tau_l1[1])

    def run_one_group(indices):
        nonlocal correct, total
        chunk_idx = 0

        def progress(stage: str, elapsed: float):
            pct = 100.0 * chunk_idx / max(1, total_chunks)
            bar_width = 30
            filled = int(round(bar_width * chunk_idx / max(1, total_chunks)))
            bar = "#" * filled + "-" * (bar_width - filled)
            msg = (
                f"\r[FHE] [{bar}] {chunk_idx}/{total_chunks} ({pct:5.1f}%) | "
                f"{stage:<12} | {elapsed:6.1f}s"
            )
            sys.stdout.write(msg)
            sys.stdout.flush()

        # process in chunks of PHYSICAL lanes
        for start in range(0, len(indices), PHYSICAL):
            chunk = indices[start:start + PHYSICAL]
            B = len(chunk)
            chunk_idx += 1
            t0 = time.time()
            progress("prepare", 0.0)

            # build plaintext batch [B,32] and labels
            input_ids = []
            attention_mask = []
            labels = []
            for j in chunk:
                it = dataset[j]
                input_ids.append(it["input_ids"])
                attention_mask.append(it["attention_mask"])
                labels.append(int(it["labels"]))

            input_ids = torch.stack(input_ids, dim=0).to(device).long()  # [B,32]
            labels = np.array(labels, dtype=np.int64)                    # [B]
            attention_mask_t = torch.stack(attention_mask, dim=0).to(device).long()

            if DEBUG_FIRST_BATCH and chunk_idx == 1:
                with torch.no_grad():
                    pt_logits = pt_model(input_ids, attention_mask=attention_mask_t)["logits"]
                    pt_pred = pt_logits.argmax(dim=1).cpu().numpy()
                pt_correct = int((pt_pred == labels).sum())
                sys.stdout.write(
                    f"\n[Plain] first batch acc={pt_correct}/{B} = {pt_correct/B:.6f}\n"
                )
                sys.stdout.write(
                    f"[Plain] labels[:8]={labels[:8].tolist()} pred[:8]={pt_pred[:8].tolist()}\n"
                )
                sys.stdout.flush()

            # plaintext embedding [B,32,128]
            x_b = embed_batch(pt_model, input_ids)

            # pack into PHYSICAL lanes: [256,32,128]
            x_pack = np.zeros((PHYSICAL, MAX_LEN, HIDDEN), dtype=np.float32)
            x_pack[:B] = x_b

            # build per-lane masks: [B,32,32] -> [PHYSICAL,32,32]
            mask0_l0 = np.zeros((PHYSICAL, MAX_LEN, MAX_LEN), dtype=np.float32)
            mask1_l0 = np.zeros((PHYSICAL, MAX_LEN, MAX_LEN), dtype=np.float32)
            mask0_l1 = np.zeros((PHYSICAL, MAX_LEN, MAX_LEN), dtype=np.float32)
            mask1_l1 = np.zeros((PHYSICAL, MAX_LEN, MAX_LEN), dtype=np.float32)

            for lane in range(B):
                attn_1d = attention_mask_t[lane]
                if torch.is_tensor(attn_1d):
                    attn_1d = attn_1d.cpu().numpy()
                else:
                    attn_1d = np.array(attn_1d, dtype=np.float32)
                attn_1d = attn_1d.astype(np.float32)
                keymask = build_keymask_tile(attn_1d)  # [32,32]

                mask0_l0[lane] = keymask * s00
                mask1_l0[lane] = keymask * s01
                mask0_l1[lane] = keymask * s10
                mask1_l1[lane] = keymask * s11

            # encrypt
            progress("encrypt", time.time() - t0)
            x_enc = BlockMatrix.encrypt_inputs(engine, x_pack, sk, block_size=BLOCK)

            # fhe forward
            progress("layer0", time.time() - t0)
            out0 = enc0.forward_one_layer(x_enc, attention_mask=(mask0_l0, mask1_l0))
            progress("layer1", time.time() - t0)
            out1 = enc1.forward_one_layer(out0, attention_mask=(mask0_l1, mask1_l1))

            # decrypt CLS for lanes 0..B-1
            progress("decrypt", time.time() - t0)
            r0 = engine.decrypt(out1.blocks[0][0], sk)  # [256,64,64]
            r1 = engine.decrypt(out1.blocks[0][1], sk)

            cls_0 = r0[:B, 0, :]  # [B,64]
            cls_1 = r1[:B, 0, :]  # [B,64]
            cls = np.concatenate([cls_0, cls_1], axis=1).astype(np.float32)  # [B,128]

            # logits + pred
            progress("classify", time.time() - t0)
            logits = cls @ W_cls.T + b_cls  # [B,C]
            pred = np.argmax(logits, axis=1)

            batch_correct = int((pred == labels).sum())
            batch_acc = batch_correct / B
            correct += batch_correct
            total += B

            elapsed = time.time() - t0
            progress("done", elapsed)
            sys.stdout.write(
                f" | batch={B} | batch_acc={batch_acc:.6f} | running_acc={correct/total:.6f}\n"
            )
            sys.stdout.flush()

            # 清理
            del x_enc, out0, out1
            gc.collect()
            torch.cuda.empty_cache()

    print("Start evaluating...")
    run_one_group(shard_indices)
    print(f"  done | seen={total} | acc={correct/total:.4f}")

    print("\n===== RESULT =====")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct/total:.6f}")


if __name__ == "__main__":
    main()
