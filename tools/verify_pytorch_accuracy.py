import torch
import numpy as np
import os
import sys
from transformers import AutoTokenizer
import argparse

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from experiments.accuracy_first.plaintext.model_plain_tinybert import PlainTinyBert

# [新增] 引入数据加载器，这是“真理”的来源
from experiments.accuracy_first.plaintext.dataset_registry import (
    add_dataset_args,
    build_dataloaders,
    normalize_dataset_name,
)
from experiments.accuracy_first.plaintext.artifact_utils import build_ckpt_path

# The sentences we used in the benchmark
TEST_SENTENCES = [
    "freeze my account",              
    "tell me the weather",            
    "what is your name",              
    "book a table for two",           
    "transfer money to mom",          
    "how is the traffic",             
    "play some music",                
    "set an alarm for 8am",           
    "what is the date today",         
    "where is the nearest gas station",
    "cancel my reservation",          
    "who are you",                    
    "exchange rate for euro",         
    "my card is lost",                
    "do you like pizza",              
    "what is my balance"              
]

def get_real_label_map(tokenizer, dataset_name, dataset_config=None, dataset_source=None):
    """
    通过初始化 DataLoader 来获取真实的 ID->Label 映射表
    """
    print("⏳ Loading dataset to extract REAL label mapping...")
    # 我们只需要 train_loader 来获取类别列表，batch_size 设为 1 即可
    train_loader, _, _ = build_dataloaders(
        dataset_name,
        tokenizer,
        max_len=32,
        batch_size=1,
        dataset_config=dataset_config,
        dataset_source=dataset_source,
    )
    
    # 获取数据集对象
    dataset = train_loader.dataset
    
    # 尝试寻找存储类别名称的属性
    # 大多数 Dataset 实现会将类别列表存储在 .classes 或 .labels 中
    id2label = {}
    
    if hasattr(dataset, 'classes'):
        # 情况 A: dataset.classes 是一个列表 ['label_a', 'label_b', ...]
        print("   -> Found 'classes' attribute.")
        id2label = {i: label for i, label in enumerate(dataset.classes)}
        
    elif hasattr(dataset, 'label_to_id'):
        # 情况 B: dataset.label_to_id 是一个字典 {'label_a': 0, ...}
        print("   -> Found 'label_to_id' attribute.")
        id2label = {v: k for k, v in dataset.label_to_id.items()}
        
    elif hasattr(dataset, 'features') and 'label' in dataset.features:
        # 情况 C: HuggingFace Datasets 格式
        print("   -> Found HuggingFace features.")
        names = dataset.features['label'].names
        id2label = {i: name for i, name in enumerate(names)}
        
    else:
        print("⚠️ Warning: Could not automatically find label map in dataset.")
        print(f"   Available attributes: {dir(dataset)}")
        # 兜底：如果找不到，我们可以尝试遍历一下数据（比较慢，但管用）
        # 这里先跳过，通常 .classes 都是有的
        
    return id2label

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch accuracy verification with label mapping.")
    add_dataset_args(parser, default_dataset="clinc150")
    return parser.parse_args()


def main():
    args = parse_args()
    print("==========================================================")
    print("   PyTorch Model Accuracy Verification (Real Labels)")
    print("==========================================================")

    # 1. Load Tokenizer & Real Labels
    model_id = "google/bert_uncased_L-2_H-128_A-2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # [关键步骤] 获取真实的 ID 映射
    dataset_name = normalize_dataset_name(args.dataset)
    ID2LABEL = get_real_label_map(
        tokenizer,
        dataset_name,
        dataset_config=args.dataset_config,
        dataset_source=args.dataset_source,
    )
    
    if not ID2LABEL:
        print("❌ Failed to load label map. Using empty map.")
    else:
        print(f"✅ Successfully loaded {len(ID2LABEL)} labels.")
        # 打印几个看看是否合理
        print(f"   ID 0: {ID2LABEL.get(0)}")
        print(f"   ID 20: {ID2LABEL.get(20)}")
        print(f"   ID 45: {ID2LABEL.get(45)}") # 看看你之前的 45 是什么

    # 2. Load Model
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 确保加载的是你最新的、训练好的正则化模型
    model_path = build_ckpt_path(
        os.path.join(project_root, "experiments/accuracy_first/plaintext"),
        dataset_name,
        args.dataset_version,
        "student_kd_plain",
    )
    
    print(f"\nLoading Model: {model_path}")
    
    model = PlainTinyBert(
        vocab_size=30522, max_len=32, hidden=128, layers=2, heads=2, 
        intermediate=512, dropout=0.0, 
        attn_type="2quad", attn_kwargs={"c": 4.0}, 
        act="gelu_poly_learnable", act_kwargs={"init_a": 0.02, "init_b": 0.5, "init_d": 0.5},
        norm_type="bias_only", learnable_tau=True, num_classes=150
    )
    
    if not os.path.exists(model_path):
        print("❌ Error: Model file not found!")
        # 尝试回退到旧路径
        model_path_old = os.path.join(project_root, "experiments/accuracy_first/plaintext/student_kd_plain.pt")
        if os.path.exists(model_path_old):
            print(f"⚠️ Falling back to: {model_path_old}")
            model_path = model_path_old
        else:
            return

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 3. Tokenize (WITH Padding & Mask)
    inputs = tokenizer(TEST_SENTENCES, return_tensors="pt", padding="max_length", max_length=32, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 4. Inference
    print("Running Inference...")
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=1).numpy()
        scores = torch.max(torch.softmax(logits, dim=1), dim=1).values.numpy()

    # 5. Display Results
    print("\nResults:")
    print(f"{'idx':<4} | {'Sentence':<30} | {'Pred ID':<8} | {'Label Name':<25} | {'Conf':<6}")
    print("-" * 90)
    
    for i, sent in enumerate(TEST_SENTENCES):
        pid = preds[i]
        conf = scores[i]
        
        # 获取真实标签名
        label_name = ID2LABEL.get(pid, "Unknown")
        
        # 截断长句子显示
        short_sent = (sent[:27] + '..') if len(sent) > 27 else sent
        
        print(f"{i:<4} | {short_sent:<30} | {pid:<8} | {label_name:<25} | {conf:.2f}")

    print("-" * 90)

if __name__ == "__main__":
    main()
