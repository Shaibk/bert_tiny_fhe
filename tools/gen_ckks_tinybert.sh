#!/usr/bin/env bash
set -euo pipefail

BASE="experiments/ckks_tinybert"
mkdir -p "$BASE"/{src,configs,scripts,results}

cat > "$BASE/README.md" << 'EOF'
# CKKS TinyBERT (结构对齐FHE版本) — Benchmark工程

目标：
- 用 desilofhe.Engine (CKKS) 搭建 TinyBERT 相关算子/结构的明文权重 + 密文输入路径
- 先做 benchmark：int/float/plaintext乘法、rotate、square、(可选) 简化线性层
- 后续再决定是否实现完整 Attention(QK^T)

注意：
- CKKS 的 “矩阵乘”需要用 rotate + hadamard 的对角线方法或自定义packing；
  这和 GL 的 BlockMatrix 矩阵乘完全不同，先用 benchmark 验证是否值得做。
EOF

cat > "$BASE/src/ckks_utils.py" << 'EOF'
import numpy as np
from desilofhe import Engine

def lvl(x):
    return getattr(x, "level", None)

def make_engine(mode="gpu"):
    return Engine(mode=mode)

def keygen(engine):
    sk = engine.create_secret_key()
    pk = engine.create_public_key(sk)
    rlk = engine.create_relinearization_key(sk)
    # rotate key（如果你要测 rotate）
    rotk = engine.create_rotation_key(sk)
    return sk, pk, rlk, rotk

def encrypt_vec(engine, pk, vec: np.ndarray):
    vec = np.asarray(vec, dtype=np.float64)
    return engine.encrypt(vec, pk)

def encode_vec(engine, vec: np.ndarray, level=None):
    vec = np.asarray(vec, dtype=np.float64)
    if level is None:
        return engine.encode(vec)
    return engine.encode(vec, level=level)
EOF

cat > "$BASE/src/bench_ops.py" << 'EOF'
import time
import numpy as np
from .ckks_utils import make_engine, keygen, encrypt_vec, encode_vec, lvl

def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    t0 = time.time()
    for _ in range(iters):
        fn()
    t1 = time.time()
    return (t1 - t0) * 1000 / iters

def main():
    print("=== CKKS op-level benchmark ===")
    engine = make_engine(mode="gpu")
    sk, pk, rlk, rotk = keygen(engine)

    n = 4096  # 向量长度（先选一个中等规模；你也可以改成 16384/32768）
    x = np.random.randn(n).astype(np.float64)

    ct = encrypt_vec(engine, pk, x)
    pt = encode_vec(engine, np.full(n, 3.0, dtype=np.float64))

    print("base ct.level =", lvl(ct))

    # 1) ct * int
    def f_mul_int():
        return engine.multiply(ct, 3)

    # 2) ct * float
    def f_mul_float():
        return engine.multiply(ct, 3.0)

    # 3) ct * plaintext
    def f_mul_pt():
        return engine.multiply(ct, pt)

    # 4) ct * ct (relinearize key)
    def f_mul_ct():
        return engine.multiply(ct, ct, rlk)

    # 5) square (如果有 square API，就用；否则 ct*ct)
    def f_square():
        try:
            return engine.square(ct, rlk)
        except Exception:
            return engine.multiply(ct, ct, rlk)

    # 6) rotate (如果 rotate 消耗与否取决于max level/multiparty；我们只测耗时)
    def f_rotate():
        try:
            return engine.rotate(ct, 1, rotk)
        except Exception:
            # 有些版本 rotate 参数顺序不同
            return engine.rotate(ct, 1)

    results = {}
    results["mul_int_ms"] = bench(f_mul_int)
    results["mul_float_ms"] = bench(f_mul_float)
    results["mul_pt_ms"] = bench(f_mul_pt)
    results["mul_ct_ms"] = bench(f_mul_ct)
    results["square_ms"] = bench(f_square)
    results["rotate_ms"] = bench(f_rotate)

    # level 对比（只做一次）
    out_int = f_mul_int()
    out_float = f_mul_float()
    out_pt = f_mul_pt()
    out_ct = f_mul_ct()

    print("\n--- timing (ms/op) ---")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    print("\n--- level check ---")
    print("ct.level        =", lvl(ct))
    print("ct*int.level    =", lvl(out_int))
    print("ct*float.level  =", lvl(out_float))
    print("ct*pt.level     =", lvl(out_pt))
    print("ct*ct.level     =", lvl(out_ct))

if __name__ == "__main__":
    main()
EOF

cat > "$BASE/scripts/run_ckks_ops_bench.sh" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail
python -m experiments.ckks_tinybert.src.bench_ops
EOF

chmod +x "$BASE/scripts/"*.sh
echo "Generated $BASE"

