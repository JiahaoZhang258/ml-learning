import numpy as np
import matplotlib.pyplot as plt

# 目标：找到 f(x) = x² 的最小值
def f(x):
    return x ** 2

def gradient(x):
    return 2 * x

# 梯度下降参数
x = 10.0      # 从 x=10 出发
lr = 0.1      # 学习率
n_steps = 30  # 迭代30次

# 记录每步的结果
history_x    = [x]
history_loss = [f(x)]

for step in range(n_steps):
    grad = gradient(x)
    x = x - lr * grad
    history_x.append(x)
    history_loss.append(f(x))
    if step % 5 == 0:
        print(f"Step {step:2d}: x={x:.4f}, loss={f(x):.6f}")

print(f"\n最终结果：x={x:.6f}, loss={f(x):.8f}")

# 画出 loss 下降曲线和 x 收敛过程
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history_loss, 'b-o', markersize=4)
plt.xlabel('迭代次数')
plt.ylabel('Loss')
plt.title('Loss 下降曲线')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_x, 'r-o', markersize=4)
plt.xlabel('迭代次数')
plt.ylabel('x 的值')
plt.title('x 收敛过程')
plt.axhline(0, color='gray', linestyle='--', label='最优解 x=0')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('week02/gradient_descent.png', dpi=100)
print("图片已保存到 week02/gradient_descent.png")

# 学习率对比实验
learning_rates = [0.001, 0.1, 1.5]
colors = ['blue', 'green', 'red']
labels = ['lr=0.001 (太小)', 'lr=0.1 (合适)', 'lr=1.5 (太大会震荡)']

plt.figure(figsize=(10, 5))
for lr, color, label in zip(learning_rates, colors, labels):
    x = 10.0
    losses = [f(x)]
    for _ in range(30):
        x = x - lr * gradient(x)
        losses.append(f(x))
    plt.plot(losses, color=color, label=label)

plt.xlabel('迭代次数')
plt.ylabel('Loss')
plt.title('不同学习率的收敛对比')
plt.legend()
plt.grid(True)
plt.ylim(-1, 110)
plt.savefig('week02/lr_comparison.png', dpi=100)
print("学习率对比图已保存")