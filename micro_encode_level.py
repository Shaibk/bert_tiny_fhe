import time
import numpy as np
import torch
import desilofhe as fhe
import pynvml

from src.block_matrix import BlockMatrix

def nvml_used(dev=0):
    h = pynvml.nvmlDeviceGetHandleByIndex(dev)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    return info.used / 1024**3

def test(level, dev=0):
    torch.cuda.set_device(dev)
    engine = fhe.GLEngine(shape=(256,64,64), mode="gpu")
    W1 = np.random.randn(128, 512).astype(np.float32)

    before = nvml_used(dev)
    t0 = time.time()
    w1_pt = BlockMatrix.encode_weights(engine, W1, level=level)
    torch.cuda.synchronize()
    dt = time.time() - t0
    after = nvml_used(dev)

    # 验证 encode 后 plaintext 的 level 是否正确
    blk_level = getattr(w1_pt.blocks[0][0], "level", None)
    print(f"level={level} | encode_time={dt:.3f}s | nvml {before:.3f}->{after:.3f} (Δ {after-before:+.3f}) | pt.level={blk_level}")

    del w1_pt
    torch.cuda.empty_cache()
    time.sleep(0.2)

if __name__ == "__main__":
    pynvml.nvmlInit()
    for L in [16, 12, 8, 4, 2, 1]:
        test(L, dev=0)
    pynvml.nvmlShutdown()
