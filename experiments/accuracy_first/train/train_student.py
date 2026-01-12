from __future__ import annotations
import argparse
import torch

from ..utils.config import load_yaml, ensure_dir
from ..models.bert_tiny_variant import BertTinyVariant
from ..distillation.trainer_kd import KDTrainer

def build_dummy_loader(batch_size: int = 8, steps: int = 100, vocab_size: int = 30522, seq_len: int = 128, num_classes: int = 2):
    for _ in range(steps):
        yield {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "labels": torch.randint(0, num_classes, (batch_size,)),
        }

def build_model_from_cfg(mcfg: dict) -> BertTinyVariant:
    attn_cfg = mcfg["attention"]
    act_cfg = mcfg["activation"]

    attn_type = attn_cfg["type"]
    attn_kwargs = dict(attn_cfg)
    attn_kwargs.pop("type", None)

    act_type = act_cfg["type"]
    act_kwargs = dict(act_cfg)
    act_kwargs.pop("type", None)

    return BertTinyVariant(
        vocab_size=30522,
        hidden_size=mcfg["hidden_size"],
        num_layers=mcfg["num_layers"],
        num_heads=mcfg["num_heads"],
        intermediate_size=mcfg["intermediate_size"],
        dropout=mcfg["dropout"],
        attn_type=attn_type,
        attn_kwargs=attn_kwargs,
        act_type=act_type,
        act_kwargs=act_kwargs,
        num_classes=2,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    teacher = build_model_from_cfg({
        **cfg["model"],
        "attention": {"type": "softmax"},
        "activation": {"type": "gelu"},
    }).to(device)

    teacher_ckpt = cfg["output"]["teacher_ckpt"]
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))

    student = build_model_from_cfg(cfg["model"]).to(device)

    loader = build_dummy_loader(batch_size=cfg["train"]["batch_size"])
    trainer = KDTrainer(student, teacher, cfg)

    for epoch in range(cfg["train"]["epochs"]):
        trainer.train_epoch(loader, device=device, epoch=epoch)

    out_dir = cfg["output"]["dir"]
    ensure_dir(out_dir)
    trainer.save(f"{out_dir}/student.pt")
    print(f"Saved student to {out_dir}/student.pt")

if __name__ == "__main__":
    main()
