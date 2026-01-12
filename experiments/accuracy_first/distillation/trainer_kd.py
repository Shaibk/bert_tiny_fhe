from __future__ import annotations
import os
import torch
import torch.nn as nn
from torch.optim import AdamW

from .losses import kd_kl_logits, attn_distill, hidden_distill
from ..utils.logging import log_kv
from ..utils.config import ensure_dir

class KDTrainer:
    def __init__(self, student: nn.Module, teacher: nn.Module, cfg: dict):
        self.student = student
        self.teacher = teacher
        self.cfg = cfg

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        lr = cfg["train"]["lr"]
        wd = cfg["train"]["weight_decay"]
        self.opt = AdamW(self.student.parameters(), lr=lr, weight_decay=wd)

        self.ce = nn.CrossEntropyLoss()

    def train_epoch(self, loader, device: str, epoch: int):
        self.student.train()
        T = float(self.cfg["distill"]["temperature"])
        lam_ce = float(self.cfg["distill"]["lambda_ce"])
        lam_l = float(self.cfg["distill"]["lambda_logits"])
        lam_a = float(self.cfg["distill"]["lambda_attn"])
        lam_h = float(self.cfg["distill"]["lambda_hidden"])

        step = 0
        for batch in loader:
            # TODO: 对齐你们数据batch字段
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attn_mask = batch.get("attn_mask", None)
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)

            with torch.no_grad():
                t_out = self.teacher(input_ids, attn_mask=attn_mask)

            s_out = self.student(input_ids, attn_mask=attn_mask)

            loss_ce = self.ce(s_out["logits"], labels)
            loss_kd = kd_kl_logits(s_out["logits"], t_out["logits"], T)
            loss_a = attn_distill(s_out["attn_probs"], t_out["attn_probs"])
            loss_h = hidden_distill(s_out["hiddens"], t_out["hiddens"])

            loss = lam_ce * loss_ce + lam_l * loss_kd + lam_a * loss_a + lam_h * loss_h

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.cfg["train"]["max_grad_norm"])
            self.opt.step()

            if step % 20 == 0:
                log_kv(step, {
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "ce": float(loss_ce.item()),
                    "kd": float(loss_kd.item()),
                    "attn": float(loss_a.item()),
                    "hid": float(loss_h.item()),
                })
            step += 1

    def save(self, path: str):
        ensure_dir(os.path.dirname(path))
        torch.save(self.student.state_dict(), path)
