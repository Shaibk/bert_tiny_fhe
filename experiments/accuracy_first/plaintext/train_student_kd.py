import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

# 引入模型
from .model_plain_tinybert import PlainTinyBert
from .distill_losses import kd_logits_kl
from .data_clinc150 import build_clinc150_dataloaders

# ================= 配置区 =================
# 正则化强度：越大，激活值越小。建议 0.01 ~ 0.1
# 如果设太大，可能会影响准确率；设太小，压不住数值。
LAMBDA_ACT = 0.05 
# =========================================

class ActivationMonitor:
    """一个小工具，用于捕获中间层的输出"""
    def __init__(self, model):
        self.activations = []
        self.hooks = []
        
        # 遍历所有 Linear 层，注册 Hook
        for name, module in model.named_modules():
            # 我们主要监控 Linear 层的输出 (Attention Output, FFN Output)
            if isinstance(module, nn.Linear):
                self.hooks.append(module.register_forward_hook(self.hook_fn))
    
    def hook_fn(self, module, input, output):
        # 记录输出的均方值 (Mean Squared Value)
        self.activations.append(output.pow(2).mean())

    def get_loss(self):
        if not self.activations:
            return 0.0
        # 将所有层的均方值加起来
        loss = sum(self.activations)
        self.activations = [] # 清空，准备下一轮
        return loss
    
    def close(self):
        for h in self.hooks: h.remove()

def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device).long()
            attn_mask = batch["attention_mask"].to(device).long()
            labels = batch["labels"].to(device).long()
            
            outputs = model(input_ids, attention_mask=attn_mask)
            preds = torch.argmax(outputs["logits"], dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. 准备数据 & Teacher
    model_id = "google/bert_uncased_L-2_H-128_A-2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    
    train_loader, val_loader, num_classes = build_clinc150_dataloaders(
        tokenizer=tokenizer, max_len=32, batch_size=128
    )

    teacher = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_classes).to(device)
    teacher.eval() # 确保 Teacher 固定

    # 2. 初始化 Student (保持你的 Bias-Only 配置)
    print("Initializing Student (8-Level, Bias-Only)...")
    student = PlainTinyBert(
        vocab_size=len(tokenizer), 
        max_len=32, hidden=128, layers=2, heads=2, intermediate=512, dropout=0.1, 
        attn_type="2quad", attn_kwargs={"c": 4.0}, 
        act="gelu_poly_learnable", act_kwargs={"init_a": 0.02, "init_b": 0.5, "init_d": 0.5},
        norm_type="bias_only", # 依然无 BN
        learnable_tau=True, num_classes=num_classes
    ).to(device)

    # 3. 注册激活值监控器
    act_monitor = ActivationMonitor(student)

    # 4. 优化器
    opt = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=0.01)
    epochs = 60
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    ce = nn.CrossEntropyLoss()

    # 5. 训练循环
    best_val_acc = 0.0
    print(f"Start Training with Activation Regularization (Lambda={LAMBDA_ACT})...")

    for epoch in range(epochs):
        student.train()
        
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device).long()
            attn_mask = batch["attention_mask"].to(device).long()
            labels = batch["labels"].to(device).long()

            # Teacher Forward
            with torch.no_grad():
                t_logits = teacher(input_ids, attention_mask=attn_mask).logits

            # Student Forward
            # Hook 会自动在后台运行，收集激活值
            s_out = student(input_ids, attention_mask=attn_mask)
            s_logits = s_out["logits"]

            # 计算 Loss
            # Task Loss
            loss_ce = ce(s_logits, labels)
            # KD Loss
            loss_kd = kd_logits_kl(s_logits, t_logits, T=2.0)
            # [关键] Activation Penalty Loss
            loss_reg = act_monitor.get_loss() * LAMBDA_ACT

            # Total Loss
            loss = 1.0 * loss_ce + 4.0 * loss_kd + loss_reg

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            scheduler.step()

            if step % 50 == 0:
                print(f"Ep {epoch} | Loss {loss.item():.4f} (Reg: {loss_reg.item():.4f})")

        # Validation
        val_acc = evaluate(student, val_loader, device)
        print(f"--- Epoch {epoch} Val Acc: {val_acc:.4f} ---")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), "student_8level_regularized.pt")
            print("--> Saved Best Model")

    act_monitor.close()

if __name__ == "__main__":
    main()