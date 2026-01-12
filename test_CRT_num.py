import argparse
import json
import os
import time
import threading
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import desilofhe as fhe

from src.bert_layers import FHEBertTinyEncoder
from src.block_matrix import BlockMatrix


# ---------- NVML sampler (true GPU memory used) ----------
try:
    import pynvml
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


class NVMLMemorySampler:
    def __init__(self, device_index: int = 0, interval_s: float = 0.02):
        self.device_index = device_index
        self.interval_s = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self.handle = None

    def start(self):
        if not _NVML_AVAILABLE:
            return
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if not _NVML_AVAILABLE:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    def _run(self):
        while not self._stop.is_set():
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                self.samples.append(int(info.used))  # bytes
            except Exception:
                pass
            time.sleep(self.interval_s)

    def mean_bytes(self) -> float:
        return float(sum(self.samples) / len(self.samples)) if self.samples else 0.0

    def peak_bytes(self) -> int:
        return int(max(self.samples)) if self.samples else 0


def bytes_to_gib(x: float) -> float:
    return float(x) / (1024 ** 3)


def run_one_level(level: int, shape: Tuple[int, int, int], num_samples: int, device: int) -> Dict[str, Any]:
    torch.cuda.set_device(device)

    # 1) engine
    engine = fhe.GLEngine(shape=shape, mode="gpu")
    max_level = engine.max_level() if hasattr(engine, "max_level") else None

    # 2) keys
    print(f"[level={level}] keygen...")
    t0 = time.time()
    sk = engine.create_secret_key()
    mult_key = engine.create_matrix_multiplication_key(sk)
    hadamard_key = engine.create_hadamard_multiplication_key(sk)
    keygen_s = time.time() - t0

    # 3) model (level passed into encoder)
    bert = FHEBertTinyEncoder(engine, mult_key, hadamard_key, level=level)

    # 4) input encrypt (ensure level is used)
    print(f"[level={level}] encrypt inputs...")
    dummy_inputs = [np.random.randn(128, 128).astype(np.float32) for _ in range(num_samples)]
    x_enc = BlockMatrix.encrypt_inputs(engine, dummy_inputs, sk, level=level)

    # 5) measure (NVML for true GPU used memory)
    nvml = NVMLMemorySampler(device_index=device, interval_s=0.02)
    nvml.start()

    torch.cuda.reset_peak_memory_stats(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f"[level={level}] run 1-layer...")
    torch.cuda.synchronize()
    start_event.record()
    out = bert.forward_one_layer(x_enc)
    end_event.record()
    torch.cuda.synchronize()

    nvml.stop()

    elapsed_ms = float(start_event.elapsed_time(end_event))
    out_level = out.get_level() if hasattr(out, "get_level") else None

    # torch allocator peak (fallback / auxiliary)
    torch_peak_bytes = int(torch.cuda.max_memory_allocated(device))

    result = {
        "level": level,
        "engine_max_level": max_level,
        "shape": shape,
        "num_samples": num_samples,
        "keygen_s": keygen_s,
        "inference_ms": elapsed_ms,
        "output_level": out_level,
        "nvml_peak_gib": bytes_to_gib(nvml.peak_bytes()) if _NVML_AVAILABLE else None,
        "nvml_mean_gib": bytes_to_gib(nvml.mean_bytes()) if _NVML_AVAILABLE else None,
        "torch_peak_gib": bytes_to_gib(torch_peak_bytes),
        "nvml_available": _NVML_AVAILABLE,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=16)
    ap.add_argument("--shape0", type=int, default=256)
    ap.add_argument("--shape1", type=int, default=64)
    ap.add_argument("--shape2", type=int, default=64)
    ap.add_argument("--num_samples", type=int, default=128)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", type=str, default="level_bench_results.jsonl")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")

    shape = (args.shape0, args.shape1, args.shape2)

    res = run_one_level(level=args.level, shape=shape, num_samples=args.num_samples, device=args.device)

    print("\n=== RESULT ===")
    print(json.dumps(res, indent=2))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
