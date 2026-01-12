from __future__ import annotations
import os
import time
from typing import Dict, Any

def log_kv(step: int, metrics: Dict[str, Any]) -> None:
    msg = " | ".join([f"{k}={v}" for k, v in metrics.items()])
    print(f"[step {step}] {msg}", flush=True)

def stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")
