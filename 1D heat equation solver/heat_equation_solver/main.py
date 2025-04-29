import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# 定义热传导方程的参数
alpha = 0.2  # 已知的热传导系数
L = 1.0  # 空间域长度
T = 1.0  # 时间域长度


# 定义初始条件函数
def initial_condition(x):
    return np.sin(np.pi * x)


# 定义边界条件函数
def boundary_condition(t):
    return 0.0


# 空间离散化
Nx = 100
x = np.linspace(0, L, Nx)
dx = x[1] - x[0]

# 时间离散化
Nt = 100
t = np.linspace(0, T, Nt)


# 构造有限差分矩阵
def heat_equation(t, u):
    d2u = np.zeros_like(u)
    # 内部节点的二阶导数
    d2u[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2
    # 应用边界条件
    d2u[0] = 0
    d2u[-1] = 0
    return alpha * d2u


# 初始条件向量
u0 = initial_condition(x)

# 求解热传导方程
sol = solve_ivp(heat_equation, [0, T], u0, t_eval=t, method='RK45')

# 提取解
u_sol = sol.y


# 生成用于PINN训练的采样点
def generate_collocation_points(N_collocation):
    x_col = np.random.uniform(0, L, N_collocation)
    t_col = np.random.uniform(0, T, N_collocation)
    return np.column_stack((x_col, t_col))


# 生成初始条件采样点
def generate_initial_points(N_initial):
    x_init = np.random.uniform(0, L, N_initial)
    t_init = np.zeros(N_initial)
    u_init = initial_condition(x_init)
    return np.column_stack((x_init, t_init)), u_init


def generate_boundary_points(N_boundary):
    # 确保左右边界点数相等
    if N_boundary % 2 != 0:
        N_boundary += 1  # 确保偶数个点

    # 左边界
    x_left = np.zeros(N_boundary // 2)
    t_left = np.random.uniform(0, T, N_boundary // 2)
    u_left = boundary_condition(t_left)

    # 如果u_left是标量，转换为数组
    if np.isscalar(u_left):
        u_left = np.full_like(t_left, u_left)

    # 右边界
    x_right = L * np.ones(N_boundary // 2)
    t_right = np.random.uniform(0, T, N_boundary // 2)
    u_right = boundary_condition(t_right)

    # 如果u_right是标量，转换为数组
    if np.isscalar(u_right):
        u_right = np.full_like(t_right, u_right)

    x_bc = np.concatenate([x_left, x_right])
    t_bc = np.concatenate([t_left, t_right])
    u_bc = np.concatenate([u_left, u_right])

    return np.column_stack((x_bc, t_bc)), u_bc


# 生成已知解的采样点
def generate_solution_points(N_solution):
    idx_x = np.random.randint(0, Nx, N_solution)
    idx_t = np.random.randint(0, Nt, N_solution)
    x_sol = x[idx_x]
    t_sol = t[idx_t]
    u_sol_points = u_sol[idx_x, idx_t]
    return np.column_stack((x_sol, t_sol)), u_sol_points


# 可视化结果
def plot_solution(x, t, u):
    plt.figure(figsize=(10, 6))
    plt.imshow(u, extent=[0, T, 0, L], origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Temperature')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Solution of the Heat Equation')
    plt.show()


# 保存数据用于PINN
def save_data_for_pinn():
    # 生成采样点
    N_collocation = 10000
    N_initial = 100
    N_boundary = 100
    N_solution = 1000

    X_col = generate_collocation_points(N_collocation)
    X_init, u_init = generate_initial_points(N_initial)
    X_bc, u_bc = generate_boundary_points(N_boundary)
    X_sol, u_sol_points = generate_solution_points(N_solution)

    # 保存数据
    np.savez('heat_equation_data.npz',
             X_col=X_col, X_init=X_init, u_init=u_init,
             X_bc=X_bc, u_bc=u_bc, X_sol=X_sol, u_sol_points=u_sol_points)

    return X_col, X_init, u_init, X_bc, u_bc, X_sol, u_sol_points


if __name__ == "__main__":
    plot_solution(x, t, u_sol)
    X_col, X_init, u_init, X_bc, u_bc, X_sol, u_sol_points = save_data_for_pinn()
    print(f"真实热传导系数 alpha = {alpha}")  