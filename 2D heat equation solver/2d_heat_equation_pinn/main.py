import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import os

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 确保CUDA上下文正确初始化
if device.type == 'cuda':
    torch.cuda.init()
    torch.cuda.current_device()


# 定义二维PINN模型
class PINN2D(nn.Module):
    def __init__(self):
        super(PINN2D, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 32),  # 输入: x, y, t
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)  # 输出: 温度 u
        )

    def forward(self, x):
        return self.layers(x)


def load_data(data_path, sample_size=10000):
    """加载数据并准备训练样本"""
    data = np.load(data_path)
    X = data['X']
    Y = data['Y']
    u_history = data['u_history']
    alpha = data['alpha']
    source_strength = data['source_strength']

    print(f"\nLoading data: {data_path}")
    print(f"Data dimensions: X.shape={X.shape}, Y.shape={Y.shape}, u_history.shape={u_history.shape}")

    # 展平网格
    x_flat = X.flatten()
    y_flat = Y.flatten()

    # 随机选择时间点
    time_indices = np.random.choice(u_history.shape[0], size=10, replace=False)

    # 采样内部点
    interior_points = []
    interior_temperatures = []

    for t_idx in time_indices:
        t = t_idx * data['dt']
        u = u_history[t_idx].flatten()

        points_per_time = sample_size // 10
        indices = np.random.choice(len(x_flat), size=points_per_time, replace=False)
        interior_points.extend([(x_flat[i], y_flat[i], t) for i in indices])
        interior_temperatures.extend(u[indices])

    print(f"Interior points sampling: Total samples={len(interior_points)}, Samples per time point={points_per_time}")

    # 准备边界点
    boundary_points = []
    boundary_temperatures = []

    for t_idx in time_indices:
        t = t_idx * data['dt']

        # 左边界 (x=0)
        boundary_points.extend([(0, y, t) for y in Y[:, 0]])
        boundary_temperatures.extend(u_history[t_idx, :, 0])

        # 右边界 (x=Lx)
        boundary_points.extend([(X[0, -1], y, t) for y in Y[:, -1]])
        boundary_temperatures.extend(u_history[t_idx, :, -1])

        # 上边界 (y=Ly)
        boundary_points.extend([(x, Y[-1, 0], t) for x in X[-1, :]])
        boundary_temperatures.extend(u_history[t_idx, -1, :])

        # 下边界 (y=0)
        boundary_points.extend([(x, 0, t) for x in X[0, :]])
        boundary_temperatures.extend(u_history[t_idx, 0, :])

    print(f"Boundary points sampling: Total samples={len(boundary_points)}")

    # 准备初始条件点 - 修正采样逻辑
    initial_points = []
    initial_temperatures = []

    # 生成2D网格的索引
    rows, cols = X.shape
    row_indices = np.random.randint(0, rows, size=2000)
    col_indices = np.random.randint(0, cols, size=2000)

    t = 0.0
    for i in range(2000):
        r, c = row_indices[i], col_indices[i]
        initial_points.append((X[r, c], Y[r, c], t))
        initial_temperatures.append(u_history[0, r, c])

    # 转换为numpy数组
    interior_points = np.array(interior_points)
    interior_temperatures = np.array(interior_temperatures)
    boundary_points = np.array(boundary_points)
    boundary_temperatures = np.array(boundary_temperatures)
    initial_points = np.array(initial_points)
    initial_temperatures = np.array(initial_temperatures)

    # 转换为张量并移至设备
    interior_points = torch.tensor(interior_points, dtype=torch.float32, device=device)
    interior_temperatures = torch.tensor(interior_temperatures, dtype=torch.float32, device=device).view(-1, 1)
    boundary_points = torch.tensor(boundary_points, dtype=torch.float32, device=device)
    boundary_temperatures = torch.tensor(boundary_temperatures, dtype=torch.float32, device=device).view(-1, 1)
    initial_points = torch.tensor(initial_points, dtype=torch.float32, device=device)
    initial_temperatures = torch.tensor(initial_temperatures, dtype=torch.float32, device=device).view(-1, 1)

    print(f"Initial condition points sampling: Total points={len(initial_points)}")

    return {
        'interior_points': interior_points,
        'interior_temperatures': interior_temperatures,
        'boundary_points': boundary_points,
        'boundary_temperatures': boundary_temperatures,
        'initial_points': initial_points,
        'initial_temperatures': initial_temperatures,
        'source_strength': source_strength,
        'true_alpha': alpha,
        'grid_data': {
            'X': X, 'Y': Y, 'u_history': u_history,
            'dt': data['dt'], 'dx': data['dx'], 'dy': data['dy']
        }
    }


def loss_function(model, alpha_param, data, source_strength):
    """计算二维热传导方程的损失函数"""
    # 数据项
    # 内部点损失
    u_pred_interior = model(data['interior_points'])
    mse_interior = torch.mean((u_pred_interior - data['interior_temperatures']) ** 2)

    # 边界条件损失
    u_pred_boundary = model(data['boundary_points'])
    mse_boundary = torch.mean((u_pred_boundary - data['boundary_temperatures']) ** 2)

    # 初始条件损失
    u_pred_initial = model(data['initial_points'])
    mse_initial = torch.mean((u_pred_initial - data['initial_temperatures']) ** 2)

    # 物理约束项
    xyt = data['interior_points'].clone().requires_grad_(True)

    # 使用autograd计算所有需要的导数
    u = model(xyt)

    # 计算一阶导数
    grads = torch.autograd.grad(u, xyt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dx = grads[:, 0:1]
    du_dy = grads[:, 1:2]
    du_dt = grads[:, 2:3]

    # 计算二阶导数
    grads_dx = torch.autograd.grad(du_dx, xyt, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
    grads_dy = torch.autograd.grad(du_dy, xyt, grad_outputs=torch.ones_like(du_dy), create_graph=True)[0]

    d2u_dx2 = grads_dx[:, 0:1]
    d2u_dy2 = grads_dy[:, 1:2]

    # 确保所有张量在同一设备上
    if alpha_param.device != d2u_dx2.device:
        alpha_param = alpha_param.to(d2u_dx2.device)

    # 将source_strength转换为张量并移至相同设备
    if not isinstance(source_strength, torch.Tensor):
        source_strength = torch.tensor(source_strength, dtype=torch.float32, device=d2u_dx2.device)
    else:
        source_strength = source_strength.to(d2u_dx2.device)

    # 二维热传导方程: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²) + Q
    residual = du_dt - alpha_param * (d2u_dx2 + d2u_dy2) - source_strength
    mse_phys = torch.mean(residual ** 2)

    # 总损失
    total_loss = mse_interior + mse_boundary + mse_initial + 100 * mse_phys

    return total_loss, mse_interior, mse_boundary, mse_initial, mse_phys


def train_pinn(data_path, save_path='.', epochs=5000):
    """训练二维热传导方程的PINN模型"""
    # 加载数据
    data = load_data(data_path)
    source_strength = data['source_strength']
    true_alpha = data['true_alpha']

    # 创建模型
    model = PINN2D().to(device)

    # 将热传导系数α作为可训练参数
    alpha_param = nn.Parameter(torch.tensor(0.9, dtype=torch.float32, device=device, requires_grad=True))

    # 定义优化器
    model_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    alpha_optimizer = torch.optim.Adam([alpha_param], lr=0.001)

    # 学习率调度器
    model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model_optimizer, mode='min', factor=0.8, patience=500
    )

    alpha_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        alpha_optimizer, mode='min', factor=0.6, patience=500
    )

    # 记录训练过程
    alpha_history = []
    loss_history = []
    model_lr_history = []
    alpha_lr_history = []

    # 训练模型
    for epoch in range(epochs):
        # 前向传播和损失计算
        total_loss, mse_interior, mse_boundary, mse_initial, mse_phys = loss_function(
            model, alpha_param, data, source_strength)

        # 反向传播和优化
        model_optimizer.zero_grad()
        alpha_optimizer.zero_grad()
        total_loss.backward()


        model_optimizer.step()
        alpha_optimizer.step()

        # 对alpha施加约束（确保为正数）
        with torch.no_grad():
            alpha_param.clamp_(min=1e-6)

        # 更新学习率
        model_scheduler.step(total_loss)
        alpha_scheduler.step(mse_phys)

        # 记录训练过程
        alpha_history.append(alpha_param.item())
        loss_history.append(total_loss.item())
        model_lr_history.append(model_optimizer.param_groups[0]['lr'])
        alpha_lr_history.append(alpha_optimizer.param_groups[0]['lr'])

        # 打印训练信息
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item():.8f}, '
                  f'α: {alpha_param.item():.8f}, '
                  f'Interior Loss: {mse_interior.item():.8f}, '
                  f'Boundary Loss: {mse_boundary.item():.8f}, '
                  f'Initial Loss: {mse_initial.item():.8f}, '
                  f'Physics Loss: {mse_phys.item():.8f}')

    # 可视化训练过程
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')

    plt.subplot(1, 4, 2)
    plt.plot(alpha_history)
    plt.axhline(y=true_alpha, color='r', linestyle='--', label=f'True α = {true_alpha}')
    plt.title('Estimated α')
    plt.xlabel('Epoch')
    plt.ylabel('α')
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(model_lr_history)
    plt.title('Model Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.yscale('log')

    plt.subplot(1, 4, 4)
    plt.plot(alpha_lr_history)
    plt.title('α Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.yscale('log')

    # 使用constrained_layout替代tight_layout
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, f'training_history_alpha_{true_alpha}.png'), dpi=300)
    plt.close()

    # 评估模型
    model.eval()

    # 选择一个时间点进行可视化
    grid_data = data['grid_data']
    t_idx = len(grid_data['u_history']) // 2
    t = t_idx * grid_data['dt']

    # 创建评估网格
    x_flat = grid_data['X'].flatten()
    y_flat = grid_data['Y'].flatten()
    t_flat = np.full_like(x_flat, t)

    eval_points = torch.tensor(np.column_stack((x_flat, y_flat, t_flat)),
                               dtype=torch.float32, device=device)

    with torch.no_grad():
        u_pred = model(eval_points).cpu().numpy().reshape(grid_data['X'].shape)

    u_true = grid_data['u_history'][t_idx]

    # 计算误差
    error = np.abs(u_pred - u_true)
    max_error = np.max(error)
    mean_error = np.mean(error)

    # 可视化结果 - 使用constrained_layout
    fig = plt.figure(figsize=(18, 5), constrained_layout=True)

    # 真实解
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(grid_data['X'], grid_data['Y'], u_true, cmap='viridis',
                             norm=colors.Normalize(vmin=0, vmax=np.max(u_true)))
    ax1.set_title('True Temperature Distribution')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Temperature')

    # 预测解
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(grid_data['X'], grid_data['Y'], u_pred, cmap='viridis',
                             norm=colors.Normalize(vmin=0, vmax=np.max(u_true)))
    ax2.set_title('PINN Predicted Temperature Distribution')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Temperature')

    # 误差分布
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(grid_data['X'], grid_data['Y'], error, cmap='Reds',
                             norm=colors.Normalize(vmin=0, vmax=max_error))
    ax3.set_title(f'Error Distribution (Max: {max_error:.4f}, Mean: {mean_error:.4f})')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Error')

    fig.colorbar(surf2, ax=[ax1, ax2, ax3], shrink=0.5, aspect=10)


    plt.savefig(os.path.join(save_path, f'temperature_prediction_alpha_{true_alpha}.png'), dpi=300)
    plt.close()

    # 打印最终结果
    print(f"\nTrue thermal conductivity alpha = {true_alpha}")
    print(f"PINN estimated thermal conductivity alpha = {alpha_param.item():.8f}")
    print(f"Relative error: {abs(alpha_param.item() - true_alpha) / true_alpha * 100:.2f}%")

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'alpha': alpha_param.item(),
        'true_alpha': true_alpha,
        'source_strength': source_strength,
    }, os.path.join(save_path, f'pinn_2d_model_alpha_{true_alpha}.pth'))

    return alpha_param.item()


if __name__ == "__main__":
    alpha = 0.8
    print(f"\n=== Training PINN model for alpha = {alpha} ===")
    data_path = f'./data/heat_equation_2d_data_alpha_{alpha}.npz'
    train_pinn(data_path, save_path='./results', epochs=2000)