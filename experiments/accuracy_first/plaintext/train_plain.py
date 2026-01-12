import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from .model_plain_tinybert import PlainTinyBert

class SimpleTextClsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx],
                       truncation=True, padding="max_length", max_length=self.max_len,
                       return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 先用 HF tokenizer 跑通流程；后面你可以换成自己的词表/分词
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    max_len = 128
    vocab_size = len(tokenizer)
    num_classes = 2

    # toy data：你跑通后再换成你的真实数据集
    texts = ["a good movie", "bad film"] * 200
    labels = [1, 0] * 200

    ds = SimpleTextClsDataset(texts, labels, tokenizer, max_len=max_len)
    dl = DataLoader(ds, batch_size=16, shuffle=True)

    model = PlainTinyBert(
        vocab_size=vocab_size, max_len=max_len,
        hidden=128, layers=2, heads=2, intermediate=512,
        attn_type="2quad", attn_kwargs={"c": 3.0, "eps": 1e-6},
        act="gelu_poly_learnable", act_kwargs={"init_a": 0.125, "init_b": 0.25, "init_d": 0.5},
        num_classes=num_classes
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    ce = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(3):
        for step, batch in enumerate(dl):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids, attention_mask=attention_mask)
            loss = ce(out["logits"], labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % 20 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}")

    torch.save(model.state_dict(), "plain_tinybert_student.pt")
    print("saved: plain_tinybert_student.pt")

if __name__ == "__main__":
    main()
