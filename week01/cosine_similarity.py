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