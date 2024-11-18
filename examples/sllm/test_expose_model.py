from vllm import LLM, SamplingParams
import argparse
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir",
                        help="Specify where the model is",
                        required=True)
    args = parser.parse_args()
    llm = LLM(model=args.model_dir, enforce_eager=True)
    model=llm.expose_model()

    state = model.state_dict()
    for key, value in state.items():
        # 检查是否是 Nested Tensor
        if isinstance(value, torch.Tensor) and value.is_nested:
            print(f"{key}: Nested Tensor (dtype: {value.dtype})")
            sub_tensors = value.unbind()
            # 按范数排序子张量
            sub_tensors_sorted = sorted(
                sub_tensors,
                key=lambda t: t.to(dtype=torch.float32).norm().item()  # 转换为浮点数后计算范数
            )
            for i, tensor in enumerate(sub_tensors_sorted):
                print(f"  Sub-tensor {i}: shape = {tensor.shape}, norm = {tensor.to(dtype=torch.float32).norm().item()}, dtype = {tensor.dtype}")
        elif isinstance(value, torch.Tensor):
            # 如果是普通张量，打印形状和数据类型
            print(f"{key}: shape = {value.shape}, dtype = {value.dtype}")
        else:
            # 如果既不是 Nested Tensor 也不是普通张量，跳过
            print(f"{key}: Skipped (type: {type(value)})")