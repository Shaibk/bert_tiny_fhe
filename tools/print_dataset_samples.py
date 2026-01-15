import argparse
import os
from datasets import load_dataset

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from experiments.accuracy_first.plaintext.dataset_registry import add_dataset_args, normalize_dataset_name


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


def _default_source(dataset_name: str) -> str:
    if dataset_name == "clinc150":
        return "DeepPavlov/clinc150"
    if dataset_name == "goemotions":
        return "go_emotions"
    if dataset_name == "jigsaw_toxic":
        return "jigsaw_toxicity_pred"
    raise ValueError(f"Unsupported dataset '{dataset_name}'")


def main():
    parser = argparse.ArgumentParser(description="Print a few dataset samples with label IDs.")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--num", type=int, default=8)
    add_dataset_args(parser, default_dataset="clinc150")
    args = parser.parse_args()

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    dataset_name = normalize_dataset_name(args.dataset)
    source = args.dataset_source or _default_source(dataset_name)
    ds = load_dataset(source, args.dataset_config)
    text_field = _guess_text_field(ds["train"].features)
    label_field = _guess_label_field(ds["train"].features)

    def is_in_domain(example):
        return example[label_field] is not None

    for k in ["train", "validation", "test"]:
        if k in ds:
            if dataset_name == "clinc150":
                ds[k] = ds[k].filter(is_in_domain)

    split = ds[args.split]

    print(f"Split: {args.split} | size={len(split)}")
    print(f"Text field: {text_field} | Label field: {label_field}")

    num = min(args.num, len(split))
    for i in range(num):
        item = split[i]
        text = item[text_field]
        label = item[label_field]
        print("-" * 60)
        print(f"idx={i}")
        print(f"text: {text}")
        print(f"label_id: {label}")


if __name__ == "__main__":
    main()
