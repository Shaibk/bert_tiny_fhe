from typing import Optional

from .data_clinc150 import build_clinc150_dataloaders, build_clinc150_dataloaders_with_test
from .data_goemotions import build_goemotions_dataloaders, build_goemotions_dataloaders_with_test
from .data_jigsaw_toxic import build_jigsaw_toxic_dataloaders, build_jigsaw_toxic_dataloaders_with_test


_ALIASES = {
    "clinc": "clinc150",
    "clinc150": "clinc150",
    "go_emotions": "goemotions",
    "goemotions": "goemotions",
    "jigsaw": "jigsaw_toxic",
    "jigsaw_toxic": "jigsaw_toxic",
    "jigsaw_toxicity": "jigsaw_toxic",
}


def normalize_dataset_name(name: str) -> str:
    key = name.strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(f"Unsupported dataset '{name}'. Supported: {sorted(set(_ALIASES.values()))}")


def add_dataset_args(parser, default_dataset: str = "clinc150"):
    parser.add_argument("--dataset", default=default_dataset, help="Dataset name (clinc150/goemotions/jigsaw_toxic).")
    parser.add_argument("--dataset-version", default=None, help="Optional dataset version suffix for artifacts.")
    parser.add_argument("--dataset-config", default=None, help="HF dataset config name (if applicable).")
    parser.add_argument("--dataset-source", default=None, help="HF dataset name override (if applicable).")


def build_dataloaders(
    dataset_name: str,
    tokenizer,
    max_len: int = 32,
    batch_size: int = 128,
    num_workers: int = 4,
    dataset_config: Optional[str] = None,
    dataset_source: Optional[str] = None,
):
    name = normalize_dataset_name(dataset_name)
    if name == "clinc150":
        return build_clinc150_dataloaders(
            tokenizer=tokenizer,
            max_len=max_len,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    if name == "goemotions":
        return build_goemotions_dataloaders(
            tokenizer=tokenizer,
            max_len=max_len,
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_config=dataset_config,
            dataset_source=dataset_source,
        )
    if name == "jigsaw_toxic":
        return build_jigsaw_toxic_dataloaders(
            tokenizer=tokenizer,
            max_len=max_len,
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_config=dataset_config,
            dataset_source=dataset_source,
        )
    raise ValueError(f"Unsupported dataset '{dataset_name}'.")


def build_dataloaders_with_test(
    dataset_name: str,
    tokenizer,
    max_len: int = 32,
    batch_size: int = 128,
    num_workers: int = 4,
    dataset_config: Optional[str] = None,
    dataset_source: Optional[str] = None,
):
    name = normalize_dataset_name(dataset_name)
    if name == "clinc150":
        return build_clinc150_dataloaders_with_test(
            tokenizer=tokenizer,
            max_len=max_len,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    if name == "goemotions":
        return build_goemotions_dataloaders_with_test(
            tokenizer=tokenizer,
            max_len=max_len,
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_config=dataset_config,
            dataset_source=dataset_source,
        )
    if name == "jigsaw_toxic":
        return build_jigsaw_toxic_dataloaders_with_test(
            tokenizer=tokenizer,
            max_len=max_len,
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_config=dataset_config,
            dataset_source=dataset_source,
        )
    raise ValueError(f"Unsupported dataset '{dataset_name}'.")
