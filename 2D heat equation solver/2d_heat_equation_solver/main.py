import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import os


def solve_2d_heat_equation(alpha=0.2, source_strength=1.0, nx=100, ny=100, nt=1000,
                           Lx=1.0, Ly=1.0, total_time=0.1, save_path='.'):
    """
    求解二维热传导方程: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²) + Q

    参数:
    - alpha: 热传导系数
    - source_strength: 恒定热源强度
    - nx, ny: x和y方向网格数
    - nt: 时间步数
    - Lx, Ly: 计算域大小
    - total_time: 总计算时间
    - save_path: 结果保存路径
    """
    # 计算网格间距和时间步长
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    dt = total_time / nt

    # 确保满足CFL条件
    cfl = alpha * dt / (dx ** 2 + dy ** 2)
    if cfl > 0.1:  # 降低CFL条件阈值以提高稳定性
        print(f"Warning: CFL condition not satisfied (CFL = {cfl:.4f} > 0.1)")
        dt = 0.1 * (dx ** 2 + dy ** 2) / alpha  # 更保守的时间步长
        nt = int(total_time / dt) + 1
        print(f"Adjusting time step: dt = {dt:.6f}, nt = {nt}")

    # 创建网格
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # 初始化温度场
    u = np.zeros((ny, nx))

    # 设置初始条件 (示例: 高斯分布)
    u_initial = np.exp(-((X - Lx / 2) ** 2 + (Y - Ly / 2) ** 2) / 0.05)
    u = u_initial.copy()

    # 创建记录数组
    u_history = np.zeros((nt + 1, ny, nx))
    u_history[0] = u.copy()

    # 时间循环
    for t in range(1, nt + 1):
        # 内部点更新 (使用有限差分法)
        un = u.copy()

        # 计算二阶导数
        d2u_dx2 = (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) / dx ** 2
        d2u_dy2 = (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]) / dy ** 2

        # 更新内部点，添加数值稳定性检查
        u_update = un[1:-1, 1:-1] + alpha * dt * (d2u_dx2 + d2u_dy2) + dt * source_strength

        # 限制温度的最大值，防止溢出
        max_temp = 1e10
        u_update = np.clip(u_update, -max_temp, max_temp)

        u[1:-1, 1:-1] = u_update

        # 应用边界条件 - 使用更稳定的实现方式
        # 左边界 (x=0): 黎曼条件 ∂u/∂x + h*u = g
        h_left = 0.5
        g_left = 0.0
        u[:, 0] = (u[:, 1] + dx * g_left) / (1 + dx * h_left)
        u[:, 0] = np.nan_to_num(u[:, 0])  # 处理可能的NaN值

        # 右边界 (x=Lx): 黎曼条件 ∂u/∂x + h*u = g
        h_right = 0.5
        g_right = 0.0
        u[:, -1] = (u[:, -2] + dx * g_right) / (1 + dx * h_right)
        u[:, -1] = np.nan_to_num(u[:, -1])  # 处理可能的NaN值

        # 上边界 (y=Ly): 黎曼条件 ∂u/∂y + h*u = g
        h_top = 0.5
        g_top = 0.0
        u[-1, :] = (u[-2, :] + dy * g_top) / (1 + dy * h_top)
        u[-1, :] = np.nan_to_num(u[-1, :])  # 处理可能的NaN值

        # 下边界 (y=0): 黎曼条件 ∂u/∂y + h*u = g
        h_bottom = 0.5
        g_bottom = 0.0
        u[0, :] = (u[1, :] + dy * g_bottom) / (1 + dy * h_bottom)
        u[0, :] = np.nan_to_num(u[0, :])  # 处理可能的NaN值

        # 检查是否有NaN或inf值
        if np.isnan(u).any() or np.isinf(u).any():
            print(f"Warning: NaN or Inf values detected at time step {t}")
            # 重置为上一时间步的值
            u = un.copy()

        # 记录当前时间步
        u_history[t] = u.copy()

        # 打印进度
        if t % (nt // 10) == 0:
            print(f"Completed time step: {t}/{nt}")

    # 保存数据
    os.makedirs(save_path, exist_ok=True)
    data_path = os.path.join(save_path, f'heat_equation_2d_data_alpha_{alpha}.npz')
    np.savez(data_path,
             X=X, Y=Y, u_history=u_history,
             alpha=alpha, source_strength=source_strength,
             dx=dx, dy=dy, dt=dt)
    print(f"Data saved to: {data_path}")

    # 可视化初始和最终状态
    fig = plt.figure(figsize=(15, 6))

    # 初始条件
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, u_initial, cmap='viridis',
                             norm=colors.Normalize(vmin=0, vmax=np.max(u_history)))
    ax1.set_title('Initial Temperature Distribution')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Temperature')

    # 最终状态
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, u, cmap='viridis',
                             norm=colors.Normalize(vmin=0, vmax=np.max(u_history)))
    ax2.set_title('Final Temperature Distribution')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Temperature')

    fig.colorbar(surf2, ax=[ax1, ax2], shrink=0.5, aspect=10)
    plt.savefig(os.path.join(save_path, f'temperature_distribution_alpha_{alpha}.png'), dpi=300)
    plt.close()

    return data_path


if __name__ == "__main__":
    # 测试不同的alpha值
    for alpha in [0.2, 0.8]:
        print(f"\nSolving 2D heat equation for alpha = {alpha}")
        solve_2d_heat_equation(alpha=alpha, source_strength=0.1,  # 降低热源强度
                               nx=50, ny=50, nt=1000,  # 减少网格数量
                               Lx=1.0, Ly=1.0, total_time=0.1,
                               save_path='./data')