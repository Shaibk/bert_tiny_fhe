import torch

# TODO: 改成你仓库里 FHE TinyBERT 的真实 import
# from your_module_path import TinyBertFHE

def main():
    # TODO: 用你 FHE 模型需要的初始化方式替换
    model = TinyBertFHE(...)  # noqa

    keys = sorted(model.state_dict().keys())
    print(f"num_keys = {len(keys)}")
    for k in keys[:120]:
        print(k)

if __name__ == "__main__":
    main()
