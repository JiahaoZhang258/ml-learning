import numpy as np

# 设置随机种子
np.random.seed(42) 

# 正态分布随机数（均值0，方差1）
a = np.random.randn(3, 4)
print("正态分布随机矩阵：\n", a)

# 0到1之间的均匀分布
b = np.random.rand(3, 4)
print("\n均匀分布随机矩阵：\n", b)

# 随机整数（0到10之间）
c = np.random.randint(0, 10, size=(3, 4))
print("\n随机整数矩阵：\n", c)