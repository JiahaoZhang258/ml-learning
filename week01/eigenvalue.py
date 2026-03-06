import numpy as np

# 构造一个矩阵
A = np.array([[3, 1],
              [1, 3]])

# 求特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

print("特征值：", eigenvalues)
print("特征向量（每一列是一个特征向量）：\n", eigenvectors)
import matplotlib.pyplot as plt

# 画出特征向量方向
origin = [0, 0]

plt.figure(figsize=(6, 6))

# 第一个特征向量（红色）
v1 = eigenvectors[:, 0] * eigenvalues[0]
plt.quiver(*origin, *v1, scale=1, scale_units='xy', angles='xy', color='red', label=f'λ₁=4')

# 第二个特征向量（蓝色）
v2 = eigenvectors[:, 1] * eigenvalues[1]
plt.quiver(*origin, *v2, scale=1, scale_units='xy', angles='xy', color='blue', label=f'λ₂=2')

plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.title('特征向量可视化')
plt.savefig('week01/eigenvectors.png')
print("图片已保存到 week01/eigenvectors.png")