import torch

# 创建两个形状不同的张量
a = torch.arange(3, dtype=torch.float)
b = torch.arange(5, dtype=torch.float)

# 将它们组合成一个嵌套张量
nt = torch.nested.nested_tensor([a, b])

print(nt.is_nested)  # 输出：True

padded_tensor = torch.nested.to_padded_tensor(nt, padding=0.0)
print(padded_tensor)
