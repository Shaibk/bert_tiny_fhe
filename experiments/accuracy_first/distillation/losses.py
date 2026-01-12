from __future__ import annotations
import torch
import torch.nn.functional as F

def kd_kl_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    """
    KL( softmax(t/T) || softmax(s/T) ) * T^2
    """
    p_t = F.softmax(teacher_logits / T, dim=-1)
    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    loss = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
    return loss

def mse_list(student_list, teacher_list) -> torch.Tensor:
    assert len(student_list) == len(teacher_list)
    loss = 0.0
    for s, t in zip(student_list, teacher_list):
        loss = loss + F.mse_loss(s, t)
    return loss

def attn_distill(student_attn_probs, teacher_attn_probs) -> torch.Tensor:
    # 简单MSE；如需更强可改成KL
    return mse_list(student_attn_probs, teacher_attn_probs)

def hidden_distill(student_hiddens, teacher_hiddens) -> torch.Tensor:
    return mse_list(student_hiddens, teacher_hiddens)
