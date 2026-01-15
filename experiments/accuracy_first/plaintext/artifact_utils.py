import os
import re
from typing import Optional


_TAG_RE = re.compile(r"[^a-z0-9]+")


def _normalize_tag(value: str) -> str:
    tag = _TAG_RE.sub("_", value.strip().lower()).strip("_")
    if not tag:
        raise ValueError("Empty dataset tag after normalization.")
    return tag


def build_dataset_tag(dataset: str, version: Optional[str] = None) -> str:
    if version:
        return _normalize_tag(f"{dataset}_{version}")
    return _normalize_tag(dataset)


def build_ckpt_path(base_dir: str, dataset: str, version: Optional[str], prefix: str) -> str:
    tag = build_dataset_tag(dataset, version)
    return os.path.join(base_dir, f"{prefix}_{tag}.pt")


def build_weights_path(project_root: str, dataset: str, version: Optional[str], prefix: str) -> str:
    tag = build_dataset_tag(dataset, version)
    return os.path.join(project_root, f"{prefix}_{tag}.npz")
