import torch
from transformers import AutoModelForCausalLM
import argparse
from vllm import LLM, SamplingParams


# 直接比较两个模型之间的对应层，统计相同参数与小差异参数的数量
# python compare_models.py /path/to/llama-3.1-8b-instruct /path/to/llama-3.1-storm-8b --thresholds 1e-5 1e-4 1e-3

# 从命令行获取模型路径和多个阈值
parser = argparse.ArgumentParser(description="Compare two models and compute similarity with multiple thresholds.")
parser.add_argument("model_instruct_path", type=str, help="Path to the first model (e.g., Llama-3.1-8B-Instruct)")
parser.add_argument("model_storm_path", type=str, help="Path to the second model (e.g., Llama-3.1-Storm-8B)")
parser.add_argument("--thresholds", type=float, nargs='+', default=[1e-5, 1e-4, 1e-3], help="List of thresholds to compare (e.g., 1e-5 1e-4 1e-3)")
args = parser.parse_args()

# 加载两个模型
model_instruct = LLM(model=args.model_instruct_path, enforce_eager=True, max_model_len=60000).expose_model()
model_storm = LLM(model=args.model_storm_path, enforce_eager=True, max_model_len=60000).expose_model()

# 获取 state_dict
state_dict_instruct = model_instruct.state_dict()
state_dict_storm = model_storm.state_dict()


# 初始化统计信息
total_numel = 0
total_zero_diff_count = 0
total_within_threshold_counts = {threshold: 0 for threshold in args.thresholds}

def calculate_diff_metrics(instruct_param, storm_param, instruct_name, thresholds, total_numel, total_zero_diff_count, total_within_threshold_counts):
    # 计算差值矩阵
    diff = instruct_param - storm_param
    total_numel += instruct_param.numel()

    # 计算差值为零的元素数目
    zero_diff_count = (diff == 0).sum().item()
    total_zero_diff_count += zero_diff_count

    # 打印输出
    print(f"{instruct_name} {zero_diff_count} {instruct_param.numel()} {zero_diff_count/instruct_param.numel():.4f}")
    
    for threshold in thresholds:
        within_threshold_count = (diff.abs() <= threshold).sum().item()
        total_within_threshold_counts[threshold] += within_threshold_count
    
    return total_numel, total_zero_diff_count, total_within_threshold_counts

# 遍历两个模型的对应层参数
for (instruct_name, instruct_param), (storm_name, storm_param) in zip(state_dict_instruct.items(), state_dict_storm.items()):
    # 检查名称是否匹配
    if instruct_name != storm_name:
        print(f"Parameter mismatch: {instruct_name} vs {storm_name}")
        continue

    if isinstance(instruct_param, torch.Tensor) and isinstance(storm_param, torch.Tensor):
        if instruct_param.is_nested and storm_param.is_nested:
            sub_tensors_instruct = instruct_param.unbind()
            sub_tensors_storm = storm_param.unbind()

            sub_tensors_instruct_sorted = sorted(sub_tensors_instruct, key=lambda t: t.to(dtype=torch.float32).norm().item())
            sub_tensors_storm_sorted = sorted(sub_tensors_storm, key=lambda t: t.to(dtype=torch.float32).norm().item())

            sub_tensor_id = 0
            for sub_tensor_instruct, sub_tensor_storm in zip(sub_tensors_instruct_sorted, sub_tensors_storm_sorted):
                if sub_tensor_instruct.shape != sub_tensor_storm.shape:
                    print(f"Parameter mismatch: {instruct_name} vs {storm_name} sub-tensor")
                    continue
                total_numel, total_zero_diff_count, total_within_threshold_counts = calculate_diff_metrics(
                    sub_tensor_instruct, sub_tensor_storm, f"{instruct_name}.sub.{sub_tensor_id}", args.thresholds,
                    total_numel, total_zero_diff_count, total_within_threshold_counts
                )
                sub_tensor_id += 1
        else:
            if instruct_param.shape != storm_param.shape:
                print(f"Parameter mismatch: {instruct_name} vs {storm_name}")
                continue
            total_numel, total_zero_diff_count, total_within_threshold_counts = calculate_diff_metrics(
                instruct_param, storm_param, instruct_name, args.thresholds, total_numel, total_zero_diff_count, total_within_threshold_counts
            )
    else:
        print(f"Parameter mismatch: {instruct_name} vs {storm_name}")

# 输出结果
print(f"Total zero diff ratio: {total_zero_diff_count / total_numel:.4f}")
for threshold, within_threshold_count in total_within_threshold_counts.items():
    print(f"Total within threshold ratio at {threshold}: {within_threshold_count / total_numel:.4f}")
