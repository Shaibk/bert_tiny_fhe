import torch
import os
import sys
import numpy as np
from transformers import AutoTokenizer

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from experiments.accuracy_first.plaintext.model_plain_tinybert import PlainTinyBert

# CLINC150 部分标签映射 (用于验证)
ID2LABEL = {
    9: "accept_reservations",
    20: "freeze_account", 
}

def main():
    # 1. 路径配置
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "experiments/accuracy_first/plaintext/student_8level.pt")
    
    print(f"Loading PyTorch Model: {model_path}")
    
    # 2. 初始化模型 (配置必须与训练一致)
    model = PlainTinyBert(
        vocab_size=30522, max_len=32, hidden=128, layers=2, heads=2, 
        intermediate=512, dropout=0.0, 
        attn_type="2quad", attn_kwargs={"c": 4.0}, 
        act="gelu_poly_learnable", act_kwargs={"init_a": 0.02, "init_b": 0.5, "init_d": 0.5},
        norm_type="bias_only", learnable_tau=True, num_classes=150
    )
    
    if not os.path.exists(model_path):
        print("❌ Error: Model file not found!")
        return

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 3. 准备输入
    text = "freeze my account"
    print(f"Input Text: '{text}'")
    
    tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=32)

    # 4. 推理
    with torch.no_grad():
        outputs = model(inputs["input_ids"])
        logits = outputs["logits"][0] # [150]

    # 5. 解析结果
    pred_id = torch.argmax(logits).item()
    pred_score = logits[pred_id].item()
    
    print("\n" + "="*30)
    print("📢 PyTorch Prediction Result")
    print("="*30)
    print(f"   Predicted ID:   {pred_id}")
    print(f"   Label Name:     {ID2LABEL.get(pred_id, 'Other/Unknown')}")
    print(f"   Logit Value:    {pred_score:.4f}")
    
    # 对比 ID 20 和 ID 9
    score_20 = logits[20].item()
    score_9 = logits[9].item()
    
    print("-" * 30)
    print(f"   Score for ID 20 (freeze_account):    {score_20:.4f}")
    print(f"   Score for ID 9  (accept_reservations): {score_9:.4f}")
    
    diff = score_20 - score_9
    print(f"   Difference (20 - 9): {diff:.4f}")
    
    if pred_id == 20:
        print("\n✅ Conclusion: The PyTorch model is CORRECT.")
        print("   The issue is definitely FHE precision/noise.")
    else:
        print("\n❌ Conclusion: The PyTorch model is WRONG.")
        print("   The model itself failed to learn this query.")

if __name__ == "__main__":
    main()