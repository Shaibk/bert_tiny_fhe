from __future__ import annotations
import argparse
import torch

from ..utils.config import load_yaml, ensure_dir
from ..models.bert_tiny_variant import BertTinyVariant

def build_dummy_loader(batch_size: int = 8, steps: int = 100, vocab_size: int = 30522, seq_len: int = 128, num_classes: int = 2):
    # TODO: 用你们真实DataLoader替换
    for _ in range(steps):
        yield {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "labels": torch.randint(0, num_classes, (batch_size,)),
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    mcfg = cfg["model"]
    model = BertTinyVariant(
        vocab_size=30522,
        hidden_size=mcfg["hidden_size"],
        num_layers=mcfg["num_layers"],
        num_heads=mcfg["num_heads"],
        intermediate_size=mcfg["intermediate_size"],
        dropout=mcfg["dropout"],
        attn_type=mcfg["attention"]["type"],
        attn_kwargs={},
        act_type=mcfg["activation"]["type"],
        act_kwargs={},
        num_classes=2,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    ce = torch.nn.CrossEntropyLoss()

    loader = build_dummy_loader(batch_size=cfg["train"]["batch_size"])
    model.train()
    step = 0
    for epoch in range(cfg["train"]["epochs"]):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids)
            loss = ce(out["logits"], labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["max_grad_norm"])
            opt.step()
            if step % 20 == 0:
                print(f"[teacher] step={step} loss={loss.item():.4f}")
            step += 1

    out_dir = cfg["output"]["dir"]
    ensure_dir(out_dir)
    torch.save(model.state_dict(), f"{out_dir}/teacher.pt")
    print(f"Saved teacher to {out_dir}/teacher.pt")

if __name__ == "__main__":
    main()
