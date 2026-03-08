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

# 生成训练数据
torch.manual_seed(42)
X = torch.randn(100, 2)                          # 100个样本，2个特征
y = 2*X[:,0] + 3*X[:,1] + torch.randn(100)*0.1  # 真实标签
y = y.unsqueeze(1)                               # 变成 (100,1)

print(f'\nX shape: {X.shape}')
print(f'y shape: {y.shape}')
print(f'前3个样本的 X:\n{X[:3]}')
print(f'前3个样本的 y:\n{y[:3]}')

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练循环
losses = []
for epoch in range(200):
    optimizer.zero_grad()        # 1. 清空梯度
    output = model(X)            # 2. 前向传播
    loss = criterion(output, y)  # 3. 计算Loss
    loss.backward()              # 4. 反向传播
    optimizer.step()             # 5. 更新参数
    losses.append(loss.item())
    if epoch % 40 == 0:
        print(f'Epoch {epoch:3d}: Loss={loss.item():.4f}')

import matplotlib.pyplot as plt

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.savefig('week03/training_loss.png', dpi=100)
print('训练完成！图片已保存')

# 查看训练好的参数
for name, param in model.named_parameters():
    print(f'{name}: {param.data}')

# 用训练好的网络预测几个样本
model.eval()
with torch.no_grad():
    test_X = torch.tensor([[1.0, 0.0],   # 真实y = 2*1 + 3*0 = 2
                            [0.0, 1.0],   # 真实y = 2*0 + 3*1 = 3
                            [1.0, 1.0]])  # 真实y = 2*1 + 3*1 = 5
    pred = model(test_X)
    print("\n预测值 vs 真实值：")
    print(f"输入[1,0]：预测={pred[0].item():.2f}，真实=2.00")
    print(f"输入[0,1]：预测={pred[1].item():.2f}，真实=3.00")
    print(f"输入[1,1]：预测={pred[2].item():.2f}，真实=5.00")