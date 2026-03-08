import torch
# 4. 自动求梯度（这是 Tensor 和 NumPy 最大的区别！）
x = torch.tensor(3.0, requires_grad=True)  # 告诉PyTorch要对x求梯度
y = x ** 2                                  # y = x²
y.backward()                                # 反向传播，自动求导
print(f'\nx={x.item()}, y={y.item()}')
print(f'dy/dx = {x.grad.item()}')           # 应该是 2*3 = 6

