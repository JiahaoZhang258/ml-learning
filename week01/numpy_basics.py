import numpy as np
import time

A = np.random.randn(1000, 1000)
b = np.random.randn(1000)

# 方法一：for 循环
t0 = time.time()
result1 = np.zeros_like(A)
for i in range(1000):
    result1[i] = A[i] + b
t1 = time.time()
print(f"for循环耗时: {(t1-t0)*1000:.2f} ms")

# 方法二：广播
t0 = time.time()
result2 = A + b
t1 = time.time()
print(f"广播耗时:   {(t1-t0)*1000:.2f} ms")

print("两种方法结果相同：", np.allclose(result1, result2))