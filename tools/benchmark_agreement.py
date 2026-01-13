import torch
import numpy as np
import time
import os
import sys
from transformers import AutoTokenizer
import desilofhe as fhe

# 引入项目模块
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from experiments.accuracy_first.plaintext.model_plain_tinybert import PlainTinyBert
from src.bert_layers import FHEBertTinyEncoder
from src.block_matrix import BlockMatrix

# 模拟一些测试句子 (循环填充到 128 个)
TEST_SENTENCES = [
    "freeze my account", "tell me the weather", "what is your name", 
    "book a table for two", "transfer money to mom", "how is the traffic",
    "play some music", "set an alarm for 8am", "what is the date today",
    "where is the nearest gas station", "cancel my reservation", "who are you",
    "exchange rate for euro", "my card is lost", "do you like pizza",
    "what is my balance"
]

def get_batch_data(batch_size=128):
    """生成 128 个输入句子和对应的 Embedding"""
    sentences = [TEST_SENTENCES[i % len(TEST_SENTENCES)] for i in range(batch_size)]
    model_id = "google/bert_uncased_L-2_H-128_A-2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer(sentences, return_tensors="pt", padding="max_length", max_length=32, truncation=True)
    return sentences, inputs

def load_pytorch_model(device="cpu"):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "experiments/accuracy_first/plaintext/student_8level.pt")
    
    model = PlainTinyBert(
        vocab_size=30522, max_len=32, hidden=128, layers=2, heads=2, 
        intermediate=512, dropout=0.0, 
        attn_type="2quad", attn_kwargs={"c": 4.0}, 
        act="gelu_poly_learnable", act_kwargs={"init_a": 0.02, "init_b": 0.5, "init_d": 0.5},
        norm_type="bias_only", learnable_tau=True, num_classes=150
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def main():
    print("==========================================================")
    print("   PyTorch vs FHE: The Ultimate Agreement Test")
    print("==========================================================")
    
    # 1. 准备数据
    BATCH_SIZE = 128
    print(f"1. Generating {BATCH_SIZE} inputs...")
    sentences, inputs_pt = get_batch_data(BATCH_SIZE)
    input_ids_pt = inputs_pt["input_ids"]
    attention_mask_pt = inputs_pt["attention_mask"] # [关键] 获取 mask
    
    # 2. PyTorch 推理 (Ground Truth)
    print("2. Running PyTorch Inference...")
    TOTAL_DAMPING = 1e-6 # 模拟 FHE 阻尼效果
    pt_model = load_pytorch_model()
    with torch.no_grad():
        # [关键修复] 传入 attention_mask，避免 PAD 干扰预测结果
        pt_out = pt_model(input_ids_pt, attention_mask=attention_mask_pt)
        pt_logits = pt_out["logits"].numpy() * TOTAL_DAMPING 
        pt_preds = np.argmax(pt_logits, axis=1)
    
    print(f"   PyTorch completed. Predictions preview: {pt_preds[:10]}")

    # 3. FHE 推理
    print("\n3. Running FHE Inference (Full Batch)...")
    
    # FHE 配置
    NUM_HEADS = 2
    PHYSICAL_BATCH = BATCH_SIZE * NUM_HEADS 
    weights_path = "fhe_weights_8level_damped.npz" # 用阻尼后的权重
    
    # 计算 Embedding (Client Side)
    # [模拟] FHE 暂时无法高效处理 Mask，我们这里只验证数值计算的一致性。
    # 为了保证对比公平，我们需要用 PyTorch 算出的无 Mask 干扰的中间 Embedding 作为输入？
    # 不，FHE 推理目前没有 Mask 逻辑，所以它实际上跑的是 "无 Mask" 的版本。
    # 为了让 PyTorch 和 FHE 对齐，我们有两种选择：
    # A. 给 FHE 加上 Mask (很难)
    # B. 让 PyTorch 也不用 Mask (简单，但预测结果可能全是 9)
    # 
    # 既然之前的测试显示不加 Mask 会导致全 9，而加上 Mask 后预测正常，
    # 说明 PyTorch 模型本身没问题。
    # 现在的目标是验证 "FHE 是否正确执行了计算"。
    # 所以我们应该让 PyTorch *也不加 Mask*，看看 FHE 的结果是否和这个 "全 9" 的结果一致。
    # 如果一致，说明 FHE 没算错，只是缺了 Mask 功能。
    # 
    # [决定] 这里我们暂时不给 PyTorch 加 Mask，以验证 FHE 的计算正确性 (Agreement)。
    # 如果你想看 FHE 的真实预测能力，未来需要在 bert_layers.py 里实现 Mask。
    
    # 回退到无 Mask 推理以对齐 FHE 现状
    with torch.no_grad():
        # 这里故意不传 mask，模拟 FHE 目前的状态
        pt_out_nomask = pt_model(input_ids_pt) 
        pt_logits_nomask = pt_out_nomask["logits"].numpy() * TOTAL_DAMPING
        pt_preds_nomask = np.argmax(pt_logits_nomask, axis=1)
        
    print(f"   PyTorch (No Mask) Preview: {pt_preds_nomask[:10]}")

    with torch.no_grad():
        x_emb = pt_model.embedding(input_ids_pt) + pt_model.pos_embedding[:, :32, :]
        x_emb = pt_model.emb_norm(x_emb) 
        x_plain_np = x_emb.numpy().astype(np.float32) 

    # 初始化 FHE
    print("   Initializing Engine...")
    engine = fhe.GLEngine(shape=(PHYSICAL_BATCH, 64, 64), mode='gpu')
    sk = engine.create_secret_key()
    mult_key = engine.create_matrix_multiplication_key(sk)
    hadamard_key = engine.create_hadamard_multiplication_key(sk)
    transposition_key = engine.create_transposition_key(sk)
    
    # 加密
    print("   Encrypting...")
    x_packed = np.tile(x_plain_np, (NUM_HEADS, 1, 1)) 
    input_enc = BlockMatrix.encrypt_inputs(engine, x_packed, sk, block_size=64)
    
    # 加载 FHE 层
    bert_l0 = FHEBertTinyEncoder(engine, mult_key, hadamard_key, transposition_key, weights_path, layer_idx=0)
    bert_l1 = FHEBertTinyEncoder(engine, mult_key, hadamard_key, transposition_key, weights_path, layer_idx=1)
    
    # 执行层
    print("   Executing Layer 0...")
    out_l0 = bert_l0.forward_one_layer(input_enc)
    
    print("   Executing Layer 1...")
    out_l1 = bert_l1.forward_one_layer(out_l0)
    
    # 解密与提取
    print("\n4. Decrypting & Comparing...")
    w_data = np.load(weights_path)
    w_cls = w_data["classifier.weight"]
    b_cls = w_data["classifier.bias"]
    
    correct_count = 0
    decrypted_full = np.zeros(x_packed.shape, dtype=np.float32)
    
    for r in range(out_l1.r_grid):
        for c in range(out_l1.c_grid):
            blk = out_l1.blocks[r][c]
            if blk is not None:
                blk_np = engine.decrypt(blk, sk)
                r0, r1 = r*64, (r+1)*64
                c0, c1 = c*64, (c+1)*64
                real_r = min(32, r1) - r0
                real_c = min(128, c1) - c0
                if real_r > 0 and real_c > 0:
                    decrypted_full[:, r0:r0+real_r, c0:c0+real_c] = blk_np[:, :real_r, :real_c]

    print(f"\n{'idx':<5} | {'Sentence':<25} | {'PT(NoMask)':<10} | {'FHE ID':<8} | {'Match?':<10}")
    print("-" * 75)
    
    for i in range(BATCH_SIZE):
        cls_vec = decrypted_full[i, 0, :] 
        fhe_logits = np.dot(cls_vec, w_cls.T) + b_cls
        fhe_pred = np.argmax(fhe_logits)
        
        # 对比对象是无 Mask 的 PyTorch 结果
        pt_target = pt_preds_nomask[i]
        
        is_match = (fhe_pred == pt_target)
        if is_match: correct_count += 1
        
        if i < 10 or i == BATCH_SIZE - 1:
            short_sent = (sentences[i][:22] + '..') if len(sentences[i]) > 22 else sentences[i]
            match_str = "✅" if is_match else "❌"
            print(f"{i:<5} | {short_sent:<25} | {pt_target:<10} | {fhe_pred:<8} | {match_str:<10}")

    agreement_rate = (correct_count / BATCH_SIZE) * 100
    print("-" * 75)
    print(f"🏆 Final Agreement Rate: {agreement_rate:.2f}% ({correct_count}/{BATCH_SIZE})")
    
    if agreement_rate > 99.0:
        print("\n🎉 SUCCESS: FHE matches Plaintext (No Mask) perfectly!")
    else:
        print("\n⚠️ Warning: Still mismatching. Check Damping or Calculation logic.")

if __name__ == "__main__":
    main()