import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 加载数据
transform = transforms.ToTensor()
train_data = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_data  = datasets.MNIST('./data', train=False, download=True, transform=transform)

print(f'训练集大小：{len(train_data)}')
print(f'测试集大小：{len(test_data)}')
print(f'图片shape：{train_data[0][0].shape}')
print(f'标签：{train_data[0][1]}')

# 2. 创建 DataLoader
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64)

print(f'\n每批大小：64')
print(f'训练集共 {len(train_loader)} 批')
print(f'测试集共 {len(test_loader)} 批')

# 3. 定义模型
model = nn.Sequential(
    nn.Flatten(),          # 把 (1,28,28) 拉平成 784 维向量
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),    # 输出10个类别（0-9）
)

print(f'\n模型结构：')
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f'总参数数量：{total_params:,}')

# 4. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. 训练
for epoch in range(5):
    model.train()
    for X, y in train_loader:
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    # 6. 测试准确率
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in test_loader:
            output = model(X)
            correct += (output.argmax(1) == y).sum().item()
    accuracy = correct / len(test_data)
    print(f'Epoch {epoch+1}: 测试准确率 = {accuracy:.4f}')