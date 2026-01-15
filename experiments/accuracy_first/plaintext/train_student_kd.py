import os
import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from .model_plain_tinybert import PlainTinyBert
from .distill_losses import kd_logits_kl
from .dataset_registry import add_dataset_args, build_dataloaders, normalize_dataset_name
from .artifact_utils import build_ckpt_path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def evaluate_student(model, val_loader, device):
    model.eval()
    correct, total = 0, 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device).long()
        attn_mask = batch["attention_mask"].to(device).long()
        labels = batch["labels"].to(device).long()

        out = model(input_ids, attention_mask=attn_mask)
        logits = out["logits"] if isinstance(out, dict) else out
        preds = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.numel()

    return correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="Train student with KD.")
    add_dataset_args(parser, default_dataset="clinc150")
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ===== 1. Data & Teacher =====
    model_id = "google/bert_uncased_L-2_H-128_A-2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    max_len = args.max_len
    batch_size = args.batch_size
    epochs = args.epochs

    dataset_name = normalize_dataset_name(args.dataset)
    train_loader, val_loader, num_classes = build_dataloaders(
        dataset_name,
        tokenizer=tokenizer,
        max_len=max_len,
        batch_size=batch_size,
        num_workers=0,
        dataset_config=args.dataset_config,
        dataset_source=args.dataset_source,
    )

    print(f"Loading pre-trained weights from {model_id}...")
    teacher = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_classes
    ).to(device)
    teacher.eval()
    teacher.requires_grad_(False)

    # ===== 2. Student =====
    print("Initializing Student (8-Level, Bias-Only)...")
    student = PlainTinyBert(
        vocab_size=len(tokenizer),
        max_len=max_len,
        hidden=128,
        layers=2,
        heads=2,
        intermediate=512,
        dropout=0.1,
        attn_type="2quad",
        attn_kwargs={"c": 4.0},
        act="gelu_poly_learnable",
        act_kwargs={"init_a": 0.02, "init_b": 0.5, "init_d": 0.5},
        norm_type="bias_only",
        learnable_tau=True,
        num_classes=num_classes,
    ).to(device)

    # ===== 3. Optimizer & Scheduler =====
    opt = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    ce = nn.CrossEntropyLoss()

    # ===== 4. Train =====
    best_acc = 0.0
    save_path = build_ckpt_path(THIS_DIR, dataset_name, args.dataset_version, "student_kd_plain")

    print("Start Training Student (KD only)...")

    for epoch in range(epochs):
        student.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device).long()
            attn_mask = batch["attention_mask"].to(device).long()
            labels = batch["labels"].to(device).long()

            with torch.no_grad():
                t_logits = teacher(input_ids, attention_mask=attn_mask).logits

            s_out = student(input_ids, attention_mask=attn_mask)
            s_logits = s_out["logits"] if isinstance(s_out, dict) else s_out

            loss_ce = ce(s_logits, labels)
            loss_kd = kd_logits_kl(s_logits, t_logits, T=2.0)
            loss = loss_ce + 4.0 * loss_kd

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            scheduler.step()

            total_loss += loss.item()

        val_acc = evaluate_student(student, val_loader, device)
        avg_loss = total_loss / len(train_loader)

        print(f"[student] epoch={epoch} loss={avg_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), save_path)
            print(f"  --> New best student saved! Acc: {best_acc:.4f}")

    print(f"Student Training finished. Best Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
