import numpy as np

# 余弦相似度公式：cos(θ) = (a·b) / (||a|| * ||b||)
def cosine_similarity(a, b):
    dot = np.dot(a, b)           # 点积
    norm_a = np.linalg.norm(a)   # ||a|| 向量长度
    norm_b = np.linalg.norm(b)   # ||b|| 向量长度
    return dot / (norm_a * norm_b)

# 测试
a = np.array([1, 0, 0])   # 指向 x 轴
b = np.array([1, 0, 0])   # 相同方向
c = np.array([0, 1, 0])   # 垂直
d = np.array([-1, 0, 0])  # 相反方向

print(f"相同方向: {cosine_similarity(a, b):.2f}")
print(f"垂直方向: {cosine_similarity(a, c):.2f}")
print(f"相反方向: {cosine_similarity(a, d):.2f}")

# 模拟词向量（假设每个单词用3维向量表示）
cat   = np.array([0.9, 0.1, 0.0])   # 猫
dog   = np.array([0.8, 0.2, 0.0])   # 狗
car   = np.array([0.0, 0.0, 1.0])   # 汽车

print(f"猫 vs 狗:  {cosine_similarity(cat, dog):.2f}")
print(f"猫 vs 汽车: {cosine_similarity(cat, car):.2f}")

# 批量计算余弦相似度（用矩阵运算，不用for循环）
def batch_cosine_similarity(A, B):
    # A, B 均为 (N, D) 矩阵
    norms_A = np.linalg.norm(A, axis=1, keepdims=True)  # (N, 1)
    norms_B = np.linalg.norm(B, axis=1, keepdims=True)  # (N, 1)
    A_norm = A / norms_A
    B_norm = B / norms_B
    return np.sum(A_norm * B_norm, axis=1)  # (N,)

# 测试
np.random.seed(42)
A = np.random.randn(5, 3)  # 5个向量，每个3维
B = np.random.randn(5, 3)

result = batch_cosine_similarity(A, B)
print("批量余弦相似度：", np.round(result, 2))