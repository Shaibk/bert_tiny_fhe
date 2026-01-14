import torch
from torch.utils.data import DataLoader
from datasets import load_dataset


def _guess_text_field(features):
    for k in ["text", "utterance", "sentence"]:
        if k in features:
            return k
    raise ValueError(f"Cannot find text field in {features.keys()}")


def _guess_label_field(features):
    for k in ["label", "intent", "labels"]:
        if k in features:
            return k
    raise ValueError(f"Cannot find label field in {features.keys()}")


def build_clinc150_dataloaders(
    tokenizer,
    max_len=32,
    batch_size=128,
    num_workers=4,
):
    """
    CLINC150 in-domain intent classification only (OOS filtered out)

    Returns:
        train_loader, val_loader, num_classes
    """

    # ===== Load dataset =====
    ds = load_dataset("DeepPavlov/clinc150")

    text_field = _guess_text_field(ds["train"].features)
    label_field = _guess_label_field(ds["train"].features)

    # ===== Filter out OOS samples (label == None) =====
    def is_in_domain(example):
        return example[label_field] is not None

    ds["train"] = ds["train"].filter(is_in_domain)
    ds["validation"] = ds["validation"].filter(is_in_domain)

    # ===== Infer num_classes safely =====
    labels = ds["train"][label_field]
    num_classes = int(max(labels)) + 1

    # ===== Tokenization =====
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

    # ===== DataLoaders =====
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