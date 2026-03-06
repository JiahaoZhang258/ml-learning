import numpy as np

# 模拟一批数据：5个样本，3个特征
np.random.seed(42)
X = np.random.randn(5, 3) * 10 + 5
print("原始数据：\n", np.round(X, 2))
print("每列均值：", np.round(X.mean(axis=0), 2))
print("每列标准差：", np.round(X.std(axis=0), 2))

# 标准化：(X - 均值) / 标准差
mean = X.mean(axis=0)
std = X.std(axis=0)
X_normalized = (X - mean) / std

print("\n标准化后的数据：\n", np.round(X_normalized, 2))
print("标准化后每列均值：", np.round(X_normalized.mean(axis=0), 2))
print("标准化后每列标准差：", np.round(X_normalized.std(axis=0), 2))