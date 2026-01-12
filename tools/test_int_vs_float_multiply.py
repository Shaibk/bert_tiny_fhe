import numpy as np
from desilofhe import Engine


def lvl(x):
    return getattr(x, "level", None)


def main():
    print("=== CKKS Engine: int vs float multiply level test ===")

    # 1) 创建 CKKS Engine（注意：不是 GLEngine）
    engine = Engine(mode="gpu")  # 也可以 'cpu'
    print("[1] Engine created")

    # 2) 生成密钥
    sk = engine.create_secret_key()
    pk = engine.create_public_key(sk)
    rlk = engine.create_relinearization_key(sk)
    print("[2] Keys created")

    # 3) 加密一个向量（CKKS Engine 的消息可以是 list/np array）
    msg = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    ct = engine.encrypt(msg, pk)
    print("[3] Ciphertext encrypted")
    print("    ct.level =", lvl(ct))

    # ---- A) ct * int ----
    print("\n[4A] Multiply by int: engine.multiply(ct, 3)")
    ct_int = engine.multiply(ct, 3)
    print("    out.level =", lvl(ct_int))

    # ---- B) ct * float ----
    print("\n[4B] Multiply by float: engine.multiply(ct, 3.0)")
    ct_flt = engine.multiply(ct, 3.0)
    print("    out.level =", lvl(ct_flt))

    # ---- C) ct * Plaintext(double) ----
    print("\n[4C] Multiply by Plaintext: engine.multiply(ct, pt)")
    pt = engine.encode(np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float64))
    print("    pt.level  =", lvl(pt))
    ct_pt = engine.multiply(ct, pt)
    print("    out.level =", lvl(ct_pt))

    # ---- D) ct * ct (needs relinearization key ideally) ----
    print("\n[4D] Multiply by Ciphertext: engine.multiply(ct, ct, rlk)")
    ct_ct = engine.multiply(ct, ct, rlk)
    print("    out.level =", lvl(ct_ct))

    # ---- Optional: Explicit rescale (should consume level) ----
    print("\n[5] Optional: rescale the ciphertext")
    ct_rs = engine.rescale(ct)
    print("    rescaled.level =", lvl(ct_rs))

    print("\n[Summary]")
    print("  base ct.level =", lvl(ct))
    print("  ct*int.level  =", lvl(ct_int))
    print("  ct*flt.level  =", lvl(ct_flt))
    print("  ct*pt.level   =", lvl(ct_pt))
    print("  ct*ct.level   =", lvl(ct_ct))
    print("  rescale.level =", lvl(ct_rs))


if __name__ == "__main__":
    main()
