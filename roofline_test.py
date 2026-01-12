import argparse
import json
import os
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import desilofhe as fhe
import pynvml

from src.block_matrix import BlockMatrix


# ---------------- NVML helpers ----------------
def nvml_init():
    pynvml.nvmlInit()


def nvml_shutdown():
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass


def gib(x_bytes: float) -> float:
    return float(x_bytes) / (1024 ** 3)


def nvml_used_bytes(dev: int) -> int:
    h = pynvml.nvmlDeviceGetHandleByIndex(dev)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    return int(info.used)


def nvml_util(dev: int) -> Tuple[int, int]:
    """
    returns: (gpu_util_percent, mem_util_percent)
    mem_util: "percentage of time over the past sample period during which global (device) memory was being read or written"
    """
    h = pynvml.nvmlDeviceGetHandleByIndex(dev)
    u = pynvml.nvmlDeviceGetUtilizationRates(h)
    return int(u.gpu), int(u.memory)


class NVMLSampler:
    """
    Sample:
      - used memory (bytes)
      - gpu util (%)
      - mem util (%)
    during inference window.
    """
    def __init__(self, dev: int, interval_s: float = 0.02):
        self.dev = dev
        self.interval_s = interval_s
        self.samples_used: List[int] = []
        self.samples_gpu: List[int] = []
        self.samples_mem: List[int] = []
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None

    def start(self):
        self.samples_used.clear()
        self.samples_gpu.clear()
        self.samples_mem.clear()
        self._stop.clear()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=1.0)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples_used.append(nvml_used_bytes(self.dev))
                g, m = nvml_util(self.dev)
                self.samples_gpu.append(g)
                self.samples_mem.append(m)
            except Exception:
                pass
            time.sleep(self.interval_s)

    def peak_used_gib(self) -> float:
        return gib(max(self.samples_used)) if self.samples_used else 0.0

    def mean_used_gib(self) -> float:
        return gib(sum(self.samples_used) / len(self.samples_used)) if self.samples_used else 0.0

    def mean_gpu_util(self) -> float:
        return float(sum(self.samples_gpu) / len(self.samples_gpu)) if self.samples_gpu else 0.0

    def mean_mem_util(self) -> float:
        return float(sum(self.samples_mem) / len(self.samples_mem)) if self.samples_mem else 0.0


# ---------------- timing ----------------
class CUDATimer:
    def __init__(self):
        self.s = torch.cuda.Event(enable_timing=True)
        self.e = torch.cuda.Event(enable_timing=True)

    def run(self, fn: Callable[[], Any]) -> Tuple[float, Any]:
        torch.cuda.synchronize()
        self.s.record()
        out = fn()
        self.e.record()
        torch.cuda.synchronize()
        return float(self.s.elapsed_time(self.e)), out


# ---------------- bound heuristic ----------------
def classify_bound(mean_gpu: float, mean_mem: float) -> str:
    """
    Heuristic classifier:
      - memory-bound: mem_util high, gpu_util modest
      - compute-bound: gpu_util high, mem_util modest
      - mixed: both high
      - overhead/latency-bound: both low
    Thresholds are adjustable; these are decent defaults for A/B testing.
    """
    if mean_mem >= 70 and mean_gpu <= 55:
        return "memory-bound (global memory busy dominates)"
    if mean_gpu >= 70 and mean_mem <= 55:
        return "compute-bound (ALU busy dominates)"
    if mean_gpu >= 65 and mean_mem >= 65:
        return "mixed-bound (both compute and memory heavily utilized)"
    return "overhead/latency-bound (neither compute nor memory highly utilized)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=16)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--shape0", type=int, default=256)
    ap.add_argument("--shape1", type=int, default=64)
    ap.add_argument("--shape2", type=int, default=64)
    ap.add_argument("--num_samples", type=int, default=128)
    ap.add_argument("--sample_interval", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--out", type=str, default="roofline_results.jsonl")
    args = ap.parse_args()

    nvml_init()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    torch.cuda.set_device(args.device)

    dev = args.device
    shape: Tuple[int, int, int] = (args.shape0, args.shape1, args.shape2)
    level = args.level

    print(f"[Config] level={level} | shape={shape} | num_samples={args.num_samples} | device={dev}")
    base_used = nvml_used_bytes(dev)
    print(f"[NVML] baseline used = {gib(base_used):.3f} GiB")

    # --- engine & keys ---
    print("[Init] Creating engine...")
    engine = fhe.GLEngine(shape=shape, mode="gpu")
    engine_max_level = getattr(engine, "max_level", None)
    if callable(engine_max_level):
        engine_max_level = engine_max_level()
    print(f"[Init] engine.max_level = {engine_max_level}")

    print("[Init] KeyGen...")
    sk = engine.create_secret_key()
    mult_key = engine.create_matrix_multiplication_key(sk)
    hadamard_key = engine.create_hadamard_multiplication_key(sk)

    print("[Init] Encrypting inputs...")
    dummy = [np.random.randn(128, 128).astype(np.float32) for _ in range(args.num_samples)]
    x_enc = BlockMatrix.encrypt_inputs(engine, dummy, sk, level=level)

    # weights on CPU (random)
    hidden = 128
    ffn = 512
    np_wq = np.random.randn(hidden, hidden).astype(np.float32)
    np_wk = np.random.randn(hidden, hidden).astype(np.float32)
    np_wv = np.random.randn(hidden, hidden).astype(np.float32)
    np_wo = np.random.randn(hidden, hidden).astype(np.float32)
    np_ff1 = np.random.randn(hidden, ffn).astype(np.float32)
    np_ff2 = np.random.randn(ffn, hidden).astype(np.float32)

    timer = CUDATimer()

    def full_layer():
        # encode weights inside stages to avoid OOM
        wq = BlockMatrix.encode_weights(engine, np_wq, level=level)
        q = x_enc.matmul(wq, mult_key)
        del wq
        torch.cuda.empty_cache()

        wk = BlockMatrix.encode_weights(engine, np_wk, level=level)
        k = x_enc.matmul(wk, mult_key)
        del wk
        torch.cuda.empty_cache()

        score = q.matmul(k, mult_key)
        del q, k
        torch.cuda.empty_cache()

        probs = score.square(hadamard_key)
        del score
        torch.cuda.empty_cache()

        wv = BlockMatrix.encode_weights(engine, np_wv, level=level)
        v = x_enc.matmul(wv, mult_key)
        del wv
        torch.cuda.empty_cache()

        context = probs.matmul(v, mult_key)
        del probs, v
        torch.cuda.empty_cache()

        wo = BlockMatrix.encode_weights(engine, np_wo, level=level)
        o = context.matmul(wo, mult_key)
        del wo, context
        torch.cuda.empty_cache()

        attn_out = x_enc.add(o)
        del o
        torch.cuda.empty_cache()

        w1 = BlockMatrix.encode_weights(engine, np_ff1, level=level)
        ff1 = attn_out.matmul(w1, mult_key)
        del w1
        torch.cuda.empty_cache()

        gelu = ff1.square(hadamard_key)
        del ff1
        torch.cuda.empty_cache()

        w2 = BlockMatrix.encode_weights(engine, np_ff2, level=level)
        ff2 = gelu.matmul(w2, mult_key)
        del w2, gelu
        torch.cuda.empty_cache()

        final = attn_out.add(ff2)
        return final

    # Warmup (optional)
    for i in range(args.warmup):
        print(f"[Warmup] {i+1}/{args.warmup}")
        _ms, _ = timer.run(full_layer)

    # Measure with NVML sampling during inference
    print("[Run] Sampling NVML util + used mem during inference...")
    sampler = NVMLSampler(dev, interval_s=args.sample_interval)
    sampler.start()
    inf_ms, out = timer.run(full_layer)
    sampler.stop()

    out_level = out.get_level() if hasattr(out, "get_level") else None

    peak_used = sampler.peak_used_gib()
    mean_used = sampler.mean_used_gib()
    mean_gpu = sampler.mean_gpu_util()
    mean_mem = sampler.mean_mem_util()
    bound = classify_bound(mean_gpu, mean_mem)

    print("\n=== Roofline-style Diagnosis ===")
    print(f"inference_time: {inf_ms/1000:.3f} s")
    print(f"nvml_used_peak: {peak_used:.3f} GiB")
    print(f"nvml_used_mean: {mean_used:.3f} GiB")
    print(f"gpu_util_mean:  {mean_gpu:.1f} %")
    print(f"mem_util_mean:  {mean_mem:.1f} %")
    print(f"bound:          {bound}")
    print(f"output_level:   {out_level}")

    rec: Dict[str, Any] = {
        "level": level,
        "shape": list(shape),
        "num_samples": args.num_samples,
        "engine_max_level": engine_max_level,
        "inference_ms": float(inf_ms),
        "nvml_used_peak_gib": float(peak_used),
        "nvml_used_mean_gib": float(mean_used),
        "gpu_util_mean_pct": float(mean_gpu),
        "mem_util_mean_pct": float(mean_mem),
        "bound": bound,
        "output_level": out_level,
        "sample_interval_s": args.sample_interval,
        "warmup": args.warmup,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\n[Saved] {args.out}")

    nvml_shutdown()


if __name__ == "__main__":
    main()
