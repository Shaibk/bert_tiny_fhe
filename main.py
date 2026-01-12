import time
import numpy as np
import torch
import desilofhe as fhe
from src.bert_layers import FHEBertTinyEncoder
from src.block_matrix import BlockMatrix

def run_bert_tiny_benchmark():
    print("=== BERT-Tiny FHE Benchmark (Privacy-Preserving Mode) ===")
    print("Config: Shape=(256, 64, 64) | Weights=Plaintext | Inputs=Ciphertext")
    
    # 1. 初始化引擎
    target_shape = (256, 64, 64)
    engine = fhe.GLEngine(shape=target_shape, mode='gpu')
    
    # 2. 模拟 Client 端: 生成密钥
    print("1. [Client] Generating Secret Key...")
    sk = engine.create_secret_key()
    
    # 3. 模拟 Server 端: 接收 Evaluation Keys
    # ... (前文初始化) ...
    
    print("2. [Server] Receiving Evaluation Keys...")
    t0 = time.time()
    mult_key = engine.create_matrix_multiplication_key(sk)
    
    # [新增] 生成 Hadamard Key
    print("   Generating Hadamard Key (for GELU/Softmax)...")
    hadamard_key = engine.create_hadamard_multiplication_key(sk)
    
    print(f"   Keys generated in {time.time()-t0:.2f}s")
    
    # 4. 初始化模型 (传入 hadamard_key)
    print("3. [Server] Loading Model...")
    bert = FHEBertTinyEncoder(engine, mult_key, hadamard_key)
    
    # ... (后续代码不变) ...
    
    # 5. 加密输入 (Client 端操作)
    print("4. [Client] Encrypting Input Batch...")
    dummy_inputs = [np.random.randn(128, 128) for _ in range(128)]
    input_enc = BlockMatrix.encrypt_inputs(engine, dummy_inputs, sk)
    
    # 6. 推理 (Server 端操作)
    print("\n5. [Server] Running Layer 1 Inference...")
    if torch.cuda.is_available(): torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    out_l1 = bert.forward_one_layer(input_enc)
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed = start_event.elapsed_time(end_event)
    
    print(f"\n✅ Layer 1 Complete!")
    print(f"   Latency: {elapsed:.2f} ms")
    print(f"   Output Level: {out_l1.get_level()}")

    # 尝试 Layer 2
    if isinstance(out_l1.get_level(), int) and out_l1.get_level() > 6:
        print("\n6. [Server] Running Layer 2 Inference...")
        start_event.record()
        out_l2 = bert.forward_one_layer(out_l1)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_l2 = start_event.elapsed_time(end_event)
        print(f"✅ Layer 2 Complete!")
        print(f"   Latency: {elapsed_l2:.2f} ms")
        print(f"   Total Latency: {elapsed + elapsed_l2:.2f} ms")

if __name__ == "__main__":
    run_bert_tiny_benchmark()