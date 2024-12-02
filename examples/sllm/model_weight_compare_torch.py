import torch
from transformers import AutoModelForCausalLM
import argparse

# 直接比较两个模型之间的对应层，统计相同参数与小差异参数的数量
# python compare_models.py /path/to/llama-3.1-8b-instruct /path/to/llama-3.1-storm-8b --thresholds 1e-5 1e-4 1e-3



# 从命令行获取模型路径和多个阈值
parser = argparse.ArgumentParser(description="Compare two models and compute similarity with multiple thresholds.")
parser.add_argument("model_instruct_path", type=str, help="Path to the first model (e.g., Llama-3.1-8B-Instruct)")
parser.add_argument("model_storm_path", type=str, help="Path to the second model (e.g., Llama-3.1-Storm-8B)")
parser.add_argument("--thresholds", type=float, nargs='+', default=[1e-5, 1e-4, 1e-3], help="List of thresholds to compare (e.g., 1e-5 1e-4 1e-3)")
args = parser.parse_args()

# 加载两个模型
model_instruct = AutoModelForCausalLM.from_pretrained(args.model_instruct_path)
model_storm = AutoModelForCausalLM.from_pretrained(args.model_storm_path)

# 获取 state_dict
state_dict_instruct = model_instruct.state_dict()
state_dict_storm = model_storm.state_dict()

# 初始化统计信息
total_numel = 0
total_zero_diff_count = 0
total_within_threshold_counts = {threshold: 0 for threshold in args.thresholds}

# 遍历两个模型的对应层参数
for (instruct_name, instruct_param), (storm_name, storm_param) in zip(state_dict_instruct.items(), state_dict_storm.items()):
    # 检查名称是否匹配
    if instruct_param.shape != storm_param.shape:
        print(f"Parameter mismatch: {instruct_name} vs {storm_name}")
        continue

    # 计算差值矩阵
    diff = instruct_param - storm_param
    total_numel += instruct_param.numel()

    # 计算差值为零的元素数目
    zero_diff_count = (diff == 0).sum().item()
    total_zero_diff_count += zero_diff_count

    print(f"{instruct_name} {zero_diff_count} {instruct_param.numel()} {zero_diff_count/instruct_param.numel():.4f}")

    # 计算每个阈值下的元素数目
    for threshold in args.thresholds:
        within_threshold_count = (diff.abs() <= threshold).sum().item()
        total_within_threshold_counts[threshold] += within_threshold_count
    

# 输出结果
print(f"Total zero diff ratio: {total_zero_diff_count / total_numel:.4f}")
for threshold, within_threshold_count in total_within_threshold_counts.items():
    print(f"Total within threshold ratio at {threshold}: {within_threshold_count / total_numel:.4f}")
