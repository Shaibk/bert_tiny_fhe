import torch
import torch.nn.functional as F

def kd_logits_kl(student_logits, teacher_logits, T: float):
    p_t = F.softmax(teacher_logits / T, dim=-1)
    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)

def mse_list(student_list, teacher_list):
    assert len(student_list) == len(teacher_list)
    loss = 0.0
    for s, t in zip(student_list, teacher_list):
        loss = loss + F.mse_loss(s, t)
    return loss

def kd_attn(student_attn, teacher_attn):
    return mse_list(student_attn, teacher_attn)

def kd_hidden(student_h, teacher_h):
    return mse_list(student_h, teacher_h)
