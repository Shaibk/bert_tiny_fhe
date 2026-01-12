import numpy as np
import torch

H = 128
FF = 512

def main():
    sd = torch.load("plain_tinybert_student.pt", map_location="cpu")

    # ====== 1) QKV ======
    # 明文模型：layers.0.attn.qkv.weight: [3H, H]
    # FHE 需要：Wq/Wk/Wv: [H, H]  (用于 X(HxH) @ W(HxH))
    qkv_w = sd["layers.0.attn.qkv.weight"]          # [3H, H]
    qkv_b = sd.get("layers.0.attn.qkv.bias", None)  # 你FHE当前没用bias，可忽略

    wq, wk, wv = qkv_w.chunk(3, dim=0)  # 每个 [H, H]（注意：这是 Linear 的 out,in）
    # FHE 使用 X @ W，因此要把 PyTorch 的 Linear 权重转成 W = W_linear.T
    np_wq = wq.T.numpy().astype(np.float32)  # [H,H]
    np_wk = wk.T.numpy().astype(np.float32)
    np_wv = wv.T.numpy().astype(np.float32)

    # ====== 2) Wo ======
    wo_w = sd["layers.0.attn.wo.weight"]  # [H, H] (out,in)
    np_wo = wo_w.T.numpy().astype(np.float32)

    # ====== 3) FFN ======
    # 明文模型：layers.0.ffn.fc1.weight: [FF, H]
    # FHE 需要：np_ff1: [H, FF]
    ff1_w = sd["layers.0.ffn.fc1.weight"]
    np_ff1 = ff1_w.T.numpy().astype(np.float32)  # [H,FF]

    # 明文模型：layers.0.ffn.fc2.weight: [H, FF]
    # FHE 需要：np_ff2: [FF, H]
    ff2_w = sd["layers.0.ffn.fc2.weight"]
    np_ff2 = ff2_w.T.numpy().astype(np.float32)  # [FF,H]

    # ====== 保存 ======
    np.savez(
        "fhe_layer0_weights.npz",
        np_wq=np_wq,
        np_wk=np_wk,
        np_wv=np_wv,
        np_wo=np_wo,
        np_ff1=np_ff1,
        np_ff2=np_ff2,
    )
    print("saved: fhe_layer0_weights.npz")
    for k, v in {
        "np_wq": np_wq, "np_wk": np_wk, "np_wv": np_wv,
        "np_wo": np_wo, "np_ff1": np_ff1, "np_ff2": np_ff2
    }.items():
        print(k, v.shape, v.dtype)

if __name__ == "__main__":
    main()
