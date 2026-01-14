import os
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, # 直接使用 HF 的标准 BERT 类
    get_linear_schedule_with_warmup
)
from .data_clinc150 import build_clinc150_dataloaders

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 使用对应的预训练模型 ID
    # prajjwal1/bert-tiny 结构正好是: Layers=2, Hidden=128, Heads=2
    model_id = "google/bert_uncased_L-2_H-128_A-2"
    print("A: start")
    print("B: before tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    max_len = 32
    batch_size = 64
    epochs = 20  # 预训练模型收敛很快，20轮足够了

    # 2. 加载数据
    print("C: after tokenizer")
    print("D: before dataloader")
    train_loader, val_loader, num_classes = build_clinc150_dataloaders(
        tokenizer=tokenizer,
        max_len=max_len,
        batch_size=batch_size,
    )

    # 3. 加载预训练的 Teacher (God Mode)
    print("E: after dataloader")
    print("F: before model")
    print(f"Loading pre-trained weights from {model_id}...")
    teacher = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_classes
    ).to(device)
    print("G: after model")
    # 4. 优化器 & 调度器
    opt = torch.optim.AdamW(teacher.parameters(), lr=5e-4, weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    
    # HF 模型内部集成了 CrossEntropyLoss，但为了代码清晰我们显式定义一下
    # (注意: HF 输出的 logits 是未归一化的，直接用 CE Loss 没问题)
    
    best_acc = 0.0
    # 注意：这里我们保存为同样的路径，方便 Student 读取
    # 但因为这是 HF 格式，Student 加载时需要一点小技巧（或者我们还是把 Teacher 转换一下）
    # 为了最简单，我们这里直接保存这个 HF 模型，Student 训练时也用 HF 接口加载 Teacher 即可
    save_path = os.path.join(THIS_DIR, "teacher_pretrained_bert_tiny.pt")

    print(f"Start Fine-tuning Teacher...")

    for epoch in range(epochs):
        teacher.train()
        total_loss = 0
        
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # HF 模型 forward 返回的是对象，包含 .logits
            outputs = teacher(input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs.loss # HF 自动计算 Loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            opt.step()
            scheduler.step()

            total_loss += loss.item()

        # ----- Validation -----
        teacher.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = teacher(input_ids, attention_mask=attn_mask)
                pred = outputs.logits.argmax(dim=-1)

                correct += (pred == labels).sum().item()
                total += labels.numel()

        acc = correct / total
        avg_loss = total_loss / len(train_loader)
        
        print(f"[teacher] epoch={epoch} loss={avg_loss:.4f} val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            # 保存整个模型对象或者 state_dict
            torch.save(teacher.state_dict(), save_path)
            print(f"  --> New best teacher saved! Acc: {best_acc:.4f}")

    print(f"Teacher Fine-tuning finished. Best Acc: {best_acc:.4f}")

if __name__ == "__main__":
    main()