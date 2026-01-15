import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, DownloadConfig


def _guess_text_field(features):
    for k in ["text", "comment_text", "sentence", "comment"]:
        if k in features:
            return k
    raise ValueError(f"Cannot find text field in {features.keys()}")


def _guess_label_field(features):
    for k in ["label", "labels", "toxic", "toxicity"]:
        if k in features:
            return k
    raise ValueError(f"Cannot find label field in {features.keys()}")


def _normalize_label(value, toxic_threshold: float):
    if isinstance(value, list):
        return int(value[0])
    if isinstance(value, (float, int)):
        return int(float(value) >= toxic_threshold)
    return int(value)


def _build_jigsaw_dataset(
    tokenizer,
    max_len=32,
    dataset_config=None,
    dataset_source=None,
    toxic_threshold=0.5,
):
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    source = dataset_source or "jigsaw_toxicity_pred"
    ds = load_dataset(
        source,
        dataset_config,
        download_config=DownloadConfig(local_files_only=True),
    )

    text_field = _guess_text_field(ds["train"].features)
    label_field = _guess_label_field(ds["train"].features)

    def normalize_example(example):
        example[label_field] = _normalize_label(example[label_field], toxic_threshold)
        return example

    for split in ["train", "validation", "test"]:
        if split in ds:
            ds[split] = ds[split].map(normalize_example)

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


def build_jigsaw_toxic_dataloaders(
    tokenizer,
    max_len=32,
    batch_size=128,
    num_workers=4,
    dataset_config=None,
    dataset_source=None,
    toxic_threshold=0.5,
):
    ds, num_classes = _build_jigsaw_dataset(
        tokenizer=tokenizer,
        max_len=max_len,
        dataset_config=dataset_config,
        dataset_source=dataset_source,
        toxic_threshold=toxic_threshold,
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


def build_jigsaw_toxic_dataloaders_with_test(
    tokenizer,
    max_len=32,
    batch_size=128,
    num_workers=4,
    dataset_config=None,
    dataset_source=None,
    toxic_threshold=0.5,
):
    ds, num_classes = _build_jigsaw_dataset(
        tokenizer=tokenizer,
        max_len=max_len,
        dataset_config=dataset_config,
        dataset_source=dataset_source,
        toxic_threshold=toxic_threshold,
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
