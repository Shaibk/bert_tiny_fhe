
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
# 引入你写好的 Encoder 类
from src.bert_layers import FHEBertTinyEncoder


# ---------------- NVML helpers ----------------
def nvml_init():
    pynvml.nvmlInit()


def nvml_shutdown():
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass


def nvml_used_bytes(dev: int) -> int:
    h = pynvml.nvmlDeviceGetHandleByIndex(dev)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    return int(info.used)


def gib(x: float) -> float:
    return float(x) / (1024 ** 3)


class NVMLSampler:
    """全局采样：推理一次 run 的 peak/mean。"""
    def __init__(self, dev: int, interval_s: float = 0.02):
        self.dev = dev
        self.interval_s = interval_s
        self.samples: List[int] = []
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None

    def start(self):
        self.samples.clear()
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
                self.samples.append(nvml_used_bytes(self.dev))
            except Exception:
                pass
            time.sleep(self.interval_s)

    def peak_gib(self) -> float:
        return gib(max(self.samples)) if self.samples else 0.0

    def mean_gib(self) -> float:
        return gib(sum(self.samples) / len(self.samples)) if self.samples else 0.0


# ---------------- timing helpers ----------------
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


# ---------------- pretty table ----------------
def print_table(title: str, headers: List[str], rows: List[List[Any]]):
    print(f"\n=== {title} ===")
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))

    def fmt(r):
        return " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers)))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(fmt(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=16)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--shape0", type=int, default=256)
    ap.add_argument("--shape1", type=int, default=64)
    ap.add_argument("--shape2", type=int, default=64)
    ap.add_argument("--num_samples", type=int, default=128)
    ap.add_argument("--out", type=str, default="stage_encode.jsonl")
    ap.add_argument("--sample_interval", type=float, default=0.02)
    # 添加 Delta 参数控制 (可选)
    ap.add_argument("--ffn_w1_delta", type=int, default=5, help="Decrease level for FFN W1")
    ap.add_argument("--ffn_w2_delta", type=int, default=7, help="Decrease level for FFN W2")
    args = ap.parse_args()

    nvml_init()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    torch.cuda.set_device(args.device)

    dev = args.device
    shape: Tuple[int, int, int] = (args.shape0, args.shape1, args.shape2)
    level = args.level

    print(f"[Config] level={level} | shape={shape} | num_samples={args.num_samples} | device={dev}")
    print(f"[Config] FFN Delta: W1=-{args.ffn_w1_delta}, W2=-{args.ffn_w2_delta}")
    print(f"[NVML] initial used = {gib(nvml_used_bytes(dev)):.3f} GiB")

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

    # --- inputs ---
    print("[Init] Encrypting inputs...")
    dummy = [np.random.randn(128, 128).astype(np.float32) for _ in range(args.num_samples)]
    x_enc = BlockMatrix.encrypt_inputs(engine, dummy, sk, level=level)

    # --- 实例化 Encoder ---
    print("[Init] Instantiating FHEBertTinyEncoder...")
    encoder = FHEBertTinyEncoder(
        engine=engine,
        mult_key=mult_key,
        hadamard_key=hadamard_key,
        level=level,
        ffn_w1_delta=args.ffn_w1_delta,
        ffn_w2_delta=args.ffn_w2_delta
    )

    timer = CUDATimer()
    sampler = NVMLSampler(dev, interval_s=args.sample_interval)

    # 这里的 stage_recs 如果你需要和以前一样详细，需要改造 Encoder 类返回中间结果
    # 这里我们只记录整个 forward_one_layer 的作为一个大步骤
    stage_rows: List[List[Any]] = []
    stage_recs: List[Dict[str, Any]] = []
    headers = ["stage", "total(s)", "Δmem(GiB)", "mem_before", "mem_after"]

    # --- Run ---
    print("\n[Run] Starting Inference (Whole Layer)...")
    sampler.start()
    
    mem_before = nvml_used_bytes(dev)
    
    # 执行整个层
    total_ms, final_out = timer.run(lambda: encoder.forward_one_layer(x_enc))
    
    mem_after = nvml_used_bytes(dev)
    sampler.stop()

    # --- Collect Stats ---
    final_level = final_out.get_level() if hasattr(final_out, "get_level") else "Unknown"
    global_peak = sampler.peak_gib()
    global_mean = sampler.mean_gib()
    
    # Record the single big stage
    rec = {
        "name": "BertTiny_1Layer",
        "total_ms": float(total_ms),
        "before_gib": gib(mem_before),
        "after_gib": gib(mem_after),
        "delta_gib": gib(mem_after - mem_before),
    }
    stage_recs.append(rec)
    stage_rows.append([
        "BertTiny_1Layer",
        f"{total_ms/1000:.3f}",
        f"{rec['delta_gib']:+.3f}",
        f"{rec['before_gib']:.3f}",
        f"{rec['after_gib']:.3f}",
    ])

    # --- Print Summary ---
    print_table(
        title=f"Performance Summary (level={level})",
        headers=headers,
        rows=stage_rows,
    )

    print(f"\n=== Summary (level={level}) ===")
    print(f"Total Inference Time:   {total_ms/1000:.3f} s")
    print(f"NVML peak (this run):   {global_peak:.3f} GiB")
    print(f"NVML mean (this run):   {global_mean:.3f} GiB")
    print(f"Final output level:     {final_level}")

    # --- Save JSON ---
    out_rec = {
        "level": level,
        "engine_max_level": engine_max_level,
        "shape": list(shape),
        "num_samples": args.num_samples,
        "ffn_w1_delta": args.ffn_w1_delta,
        "ffn_w2_delta": args.ffn_w2_delta,
        "total_stage_time_ms": float(total_ms),
        "nvml_peak_gib": float(global_peak),
        "nvml_mean_gib": float(global_mean),
        "final_level": final_level,
        "stage_recs": stage_recs,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(out_rec) + "\n")
    print(f"\n[Saved] {args.out}")

    nvml_shutdown()


if __name__ == "__main__":
    main()

