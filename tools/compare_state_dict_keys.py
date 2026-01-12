import torch

def show_keys(path, n=60):
    sd = torch.load(path, map_location="cpu")
    keys = sorted(sd.keys())
    print(f"\n== {path} ==")
    print(f"num_keys = {len(keys)}")
    for k in keys[:n]:
        print(k)
    if len(keys) > n:
        print("...")

if __name__ == "__main__":
    show_keys("plain_tinybert_student.pt", n=80)
