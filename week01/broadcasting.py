import numpy as np

# 情况一：(3,4) + (4,) → 可以广播
A = np.ones((3, 4))
b = np.array([1, 2, 3, 4])
print("情况一：(3,4) + (4,)")
print("结果形状：", (A + b).shape)
print(A + b)

# 情况二：(3,4) + (3,1) → 可以广播
A = np.ones((3, 4))
b = np.array([[1],
              [2],
              [3]])
print("情况二：(3,4) + (3,1)")
print("b 的形状：", b.shape)
print("结果形状：", (A + b).shape)
print(A + b)

# 情况三：(3,4) + (3,2) → 不能广播！
try:
    A = np.ones((3, 4))
    b = np.ones((3, 2))
    print(A + b)
except ValueError as e:
    print("报错了！原因：", e)