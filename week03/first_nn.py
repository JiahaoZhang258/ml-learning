import torch
import torch.nn as nn

# 用 nn.Sequential 搭建神经网络
model = nn.Sequential(
    nn.Linear(2, 4),   # 全连接层：输入2维，输出4维
    nn.ReLU(),          # 激活函数
    nn.Linear(4, 1),   # 全连接层：输入4维，输出1维
)

print('模型结构：')
print(model)

# 做一次前向传播
x = torch.tensor([[1.0, 2.0]])  # 输入：1个样本，2个特征
output = model(x)
print(f'\n输入: {x}')
print(f'输出: {output}')
print(f'输入shape: {x.shape}')
print(f'输出shape: {output.shape}')

# 统计参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f'\n总参数数量：{total_params}')

# 手动算一下：
# 第一层 Linear(2,4)：权重 2×4=8 个，偏置 4 个，共 12 个
# 第二层 Linear(4,1)：权重 4×1=4 个，偏置 1 个，共 5 个
# 总共：12 + 5 = 17 个
print('手动计算：第一层12个 + 第二层5个 = 17个')