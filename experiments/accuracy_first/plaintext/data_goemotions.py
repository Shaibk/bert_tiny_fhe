import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, DownloadConfig


def _build_goemotions_dataset(
    tokenizer,
    max_len=32,
    dataset_config=None,
    dataset_source=None,
):
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    source = dataset_source or "go_emotions"
    config = dataset_config or "simplified"
    ds = load_dataset(
        source,
        config,
        download_config=DownloadConfig(local_files_only=True),
    )

    text_field = "text"
    label_field = "labels"

    def has_label(example):
        return example[label_field] is not None and len(example[label_field]) > 0

    for split in ["train", "validation", "test"]:
        if split in ds:
            ds[split] = ds[split].filter(has_label)

    def to_single_label(example):
        labels = example[label_field]
        example[label_field] = int(labels[0])
        return example

    for split in ["train", "validation", "test"]:
        if split in ds:
            ds[split] = ds[split].map(to_single_label)

    labels = ds["train"][label_field]
    num_classes = int(max(labels)) + 1

    def tokenize_fn(batch):
        enc = tokenizer(
            batch[text_field],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )
        enc["labels"] = batch[label_field]
        return enc

    ds = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=ds["train"].column_names,
    )
    ds.set_format(type="torch")
    return ds, num_classes


def build_goemotions_dataloaders(
    tokenizer,
    max_len=32,
    batch_size=128,
    num_workers=4,
    dataset_config=None,
    dataset_source=None,
):
    """
    GoEmotions (single-label proxy): uses the first label in each example.
    """
    ds, num_classes = _build_goemotions_dataset(
        tokenizer=tokenizer,
        max_len=max_len,
        dataset_config=dataset_config,
        dataset_source=dataset_source,
    )

    train_loader = DataLoader(
        ds["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ds["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, num_classes


def build_goemotions_dataloaders_with_test(
    tokenizer,
    max_len=32,
    batch_size=128,
    num_workers=4,
    dataset_config=None,
    dataset_source=None,
):
    ds, num_classes = _build_goemotions_dataset(
        tokenizer=tokenizer,
        max_len=max_len,
        dataset_config=dataset_config,
        dataset_source=dataset_source,
    )

    train_loader = DataLoader(
        ds["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ds["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = None
    if "test" in ds:
        test_loader = DataLoader(
            ds["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader, num_classes
