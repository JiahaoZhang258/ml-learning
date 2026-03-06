创建数组

np.array([1,2,3]) — 从列表创建数组
np.zeros_like(A) — 创建和 A 形状相同的全零矩阵
np.random.randn(3,4) — 正态分布随机矩阵
np.random.rand(3,4) — 均匀分布随机矩阵（0到1）
np.random.randint(0,10,size=(3,4)) — 随机整数矩阵
np.random.seed(42) — 设置随机种子

查看信息

.shape — 查看形状
type() — 查看类型

矩阵运算

A @ B — 矩阵乘法
np.dot(a, b) — 点积
A.T — 转置
np.linalg.det(A) — 行列式
np.linalg.inv(A) — 逆矩阵
np.linalg.eig(A) — 特征值和特征向量
np.linalg.norm(a) — 向量长度

索引与切片

A[0] — 取第0行
A[:, 1] — 取第1列
A[1, 2] — 取第1行第2列元素
A[0:2, :] — 取第0到1行
A[A > 6] — 条件索引

其他

np.allclose(A, B) — 判断两个矩阵是否近似相等
np.round(A, decimals=10) — 四舍五入
A.copy() — 复制矩阵
