import torch
import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from experiments.accuracy_first.plaintext.model_plain_tinybert import PlainTinyBert

def check_values():
    # 1. 加载模型
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "experiments/accuracy_first/plaintext/student_8level.pt")
    
    model = PlainTinyBert(
        vocab_size=30522, max_len=32, hidden=128, layers=2, heads=2, 
        intermediate=512, dropout=0.0, 
        attn_type="2quad", attn_kwargs={"c": 4.0}, 
        act="gelu_poly_learnable", act_kwargs={"init_a": 0.02, "init_b": 0.5, "init_d": 0.5},
        norm_type="bias_only", learnable_tau=True, num_classes=150
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 2. 构造输入 "freeze my account"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
    text = "freeze my account"
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=32)

    # 3. Hook 钩子：抓取中间层数值
    print(f"\n🔍 Inspecting Internal Values for: '{text}'\n")
    
    def print_stat(name, tensor):
        print(f"[{name}] Mean: {tensor.mean().item():.4f} | Max: {tensor.max().item():.4f} | Min: {tensor.min().item():.4f} | Std: {tensor.std().item():.4f}")

    # Hook Attention Score
    def hook_attn(module, input, output):
        # output is tuple (out, probs) or just out
        # PlainSelfAttention returns out. 
        # But we want internal score. Hard to hook without changing code.
        # Let's verify Layer Output
        print_stat(f"Layer Output", output)

    model.layers[0].register_forward_hook(hook_attn)
    model.layers[1].register_forward_hook(hook_attn)

    # 4. Forward
    with torch.no_grad():
        out = model(inputs["input_ids"])
        logits = out["logits"]
    
    print("\n------------------------------------------------")
    print_stat("Final Logits", logits)
    print("------------------------------------------------")

if __name__ == "__main__":
    check_values()