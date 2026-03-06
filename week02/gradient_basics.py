import numpy as np

# f(x) = x²
def f(x):
    return x ** 2

# 导数 f'(x) = 2x
def df(x):
    return 2 * x

# 在几个点上计算导数
for x in [-3, -1, 0, 1, 3]:
    print(f"x={x:2d},  f(x)={f(x):2d},  导数={df(x):2d},  含义：{'向右走函数增大' if df(x)>0 else '向右走函数减小' if df(x)<0 else '最低点！'}")

    # 二元函数 f(x, y) = x² + y²
def f2(x, y):
    return x**2 + y**2

# 偏导数
# ∂f/∂x = 2x （把 y 当常数，对 x 求导）
# ∂f/∂y = 2y （把 x 当常数，对 y 求导）
def gradient_f2(x, y):
    df_dx = 2 * x
    df_dy = 2 * y
    return np.array([df_dx, df_dy])

# 在点 (3, 4) 计算梯度
x, y = 3.0, 4.0
grad = gradient_f2(x, y)
print(f"\n二元函数 f(x,y) = x² + y²")
print(f"在点 ({x}, {y}) 处：")
print(f"函数值 = {f2(x, y)}")
print(f"梯度 = {grad}  （这是一个向量，指向函数增大最快的方向）")
print(f"下降方向 = {-grad}  （梯度的反方向）")

# 链式法则例子
# f(x) = (x² + 1)²
# 令 g(x) = x² + 1，则 f = g²
# df/dx = df/dg × dg/dx = 2g × 2x = 2(x²+1) × 2x

def f_chain(x):
    return (x**2 + 1)**2

def df_chain(x):
    g = x**2 + 1       # 中间变量
    df_dg = 2 * g      # f 对 g 的导数
    dg_dx = 2 * x      # g 对 x 的导数
    return df_dg * dg_dx  # 链式法则

# 验证
for x in [1.0, 2.0, 3.0]:
    print(f"x={x}, 链式法则导数={df_chain(x):.2f}")

# 用数值方法验证链式法则
def numerical_gradient(f, x, eps=1e-5):
    return (f(x + eps) - f(x - eps)) / (2 * eps)

print("\n链式法则 vs 数值梯度验证：")
for x in [1.0, 2.0, 3.0]:
    analytic = df_chain(x)
    numeric  = numerical_gradient(f_chain, x)
    print(f"x={x}, 链式法则={analytic:.4f}, 数值梯度={numeric:.4f}, 误差={abs(analytic-numeric):.2e}")