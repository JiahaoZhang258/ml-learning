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