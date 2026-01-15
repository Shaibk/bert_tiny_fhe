import os
import sys
import time
import argparse
import torch
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from experiments.accuracy_first.plaintext.model_plain_tinybert import PlainTinyBert
from experiments.accuracy_first.plaintext.dataset_registry import (
    add_dataset_args,
    build_dataloaders_with_test,
    normalize_dataset_name,
)
from experiments.accuracy_first.plaintext.artifact_utils import build_ckpt_path


MODEL_ID = "google/bert_uncased_L-2_H-128_A-2"
CKPT_PREFIX = "student_kd_plain"
MAX_LEN = 32
HIDDEN = 128


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark plaintext test-set runtime.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use (sets CUDA_VISIBLE_DEVICES).")
    parser.add_argument("--batch-size", type=int, default=128)
    add_dataset_args(parser, default_dataset="clinc150")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = normalize_dataset_name(args.dataset)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_path = build_ckpt_path(
        os.path.join(project_root, "experiments/accuracy_first/plaintext"),
        dataset_name,
        args.dataset_version,
        CKPT_PREFIX,
    )

    print(f"Device: {device} (gpu={args.gpu})")

    print("Loading tokenizer (offline)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, local_files_only=True)
    print("Loading dataset (offline)...")
    _, _, test_loader, num_classes = build_dataloaders_with_test(
        dataset_name,
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        batch_size=args.batch_size,
        dataset_config=args.dataset_config,
        dataset_source=args.dataset_source,
    )
    if test_loader is None:
        raise ValueError("No test split found in dataset.")

    model = PlainTinyBert(
        vocab_size=len(tokenizer), max_len=MAX_LEN, hidden=HIDDEN, layers=2, heads=2,
        intermediate=512, dropout=0.0,
        attn_type="2quad", attn_kwargs={"c": 4.0},
        act="gelu_poly_learnable", act_kwargs={"init_a": 0.02, "init_b": 0.5, "init_d": 0.5},
        norm_type="bias_only", learnable_tau=True, num_classes=num_classes,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
    model.eval()

    total = 0
    correct = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask=attention_mask)["logits"]
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.size(0))

    elapsed = time.time() - t0
    print(f"Plaintext accuracy: {correct}/{total} = {correct/total:.6f}")
    print(f"Elapsed: {elapsed:.1f}s | per_sample: {elapsed/total:.6f}s")


if __name__ == "__main__":
    main()
