import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
data = np.load('heat_equation_data.npz')
X_col = data['X_col']
X_init = data['X_init']
u_init = data['u_init']
X_bc = data['X_bc']
u_bc = data['u_bc']
X_sol = data['X_sol']
u_sol_points = data['u_sol_points']

# 转换为PyTorch张量并移至设备
X_col_torch = torch.tensor(X_col, dtype=torch.float32, device=device)
X_init_torch = torch.tensor(X_init, dtype=torch.float32, device=device)
u_init_torch = torch.tensor(u_init, dtype=torch.float32, device=device)
X_bc_torch = torch.tensor(X_bc, dtype=torch.float32, device=device)
u_bc_torch = torch.tensor(u_bc, dtype=torch.float32, device=device)
X_sol_torch = torch.tensor(X_sol, dtype=torch.float32, device=device)
u_sol_torch = torch.tensor(u_sol_points, dtype=torch.float32, device=device)


# 定义PINN模型
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.layers(x)


# 创建模型并移至设备
model = PINN().to(device)

# 将热传导系数α作为可训练参数
alpha_param = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=device, requires_grad=True))


# 定义损失函数
def loss_function():
    # 预测初始条件
    model.eval()
    with torch.no_grad():
        u_init_pred = model(X_init_torch).squeeze()
    init_loss = torch.mean((u_init_pred - u_init_torch) ** 2)

    # 预测边界条件
    with torch.no_grad():
        u_bc_pred = model(X_bc_torch).squeeze()
    bc_loss = torch.mean((u_bc_pred - u_bc_torch) ** 2)

    # 预测已知解
    with torch.no_grad():
        u_sol_pred = model(X_sol_torch).squeeze()
    sol_loss = torch.mean((u_sol_pred - u_sol_torch) ** 2)

    # 计算物理残差
    model.train()
    x = X_col_torch[:, 0:1].requires_grad_(True)
    t = X_col_torch[:, 1:2].requires_grad_(True)

    # 使用autograd计算导数
    u = model(torch.cat([x, t], dim=1))

    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dt = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]

    # 热传导方程: du/dt = α * d²u/dx²
    residual = du_dt - alpha_param * d2u_dx2
    phys_loss = torch.mean(residual ** 2)

    # 总损失，增加物理损失的权重
    total_loss = init_loss + bc_loss + sol_loss + 5000.0 * phys_loss

    return total_loss, init_loss, bc_loss, sol_loss, phys_loss


# 为神经网络参数设置优化器和调度器
model_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    model_optimizer, mode='min', factor=0.5, patience=500
)

# 为alpha参数设置优化器和调度器
alpha_optimizer = torch.optim.Adam([alpha_param], lr=0.02)
alpha_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    alpha_optimizer, mode='min', factor=0.8, patience=200, threshold=0.05, min_lr=0.005
)

# 记录训练过程
alpha_history = []
loss_history = []
model_lr_history = []
alpha_lr_history = []
phys_loss_history = []


# 训练函数，添加参数约束
def train_step():
    # 前向传播和损失计算
    total_loss, init_loss, bc_loss, sol_loss, phys_loss = loss_function()

    # 反向传播和优化
    model_optimizer.zero_grad()
    alpha_optimizer.zero_grad()
    total_loss.backward()
    model_optimizer.step()
    alpha_optimizer.step()

    # 对alpha施加约束（确保为正数）
    with torch.no_grad():
        alpha_param.clamp_(min=0.005)  # 防止alpha变为负数或零

    # 记录训练过程
    alpha_history.append(alpha_param.item())
    loss_history.append(total_loss.item())
    phys_loss_history.append(phys_loss.item())
    model_lr_history.append(model_optimizer.param_groups[0]['lr'])
    alpha_lr_history.append(alpha_optimizer.param_groups[0]['lr'])

    return total_loss.item(), init_loss.item(), bc_loss.item(), sol_loss.item(), phys_loss.item(), alpha_param.item()


# 训练模型
epochs = 5000
for epoch in range(epochs):
    total_loss, init_loss, bc_loss, sol_loss, phys_loss, alpha_val = train_step()

    # 更新模型学习率
    old_model_lr = model_optimizer.param_groups[0]['lr']
    model_scheduler.step(total_loss)
    new_model_lr = model_optimizer.param_groups[0]['lr']

    # 更新alpha学习率
    old_alpha_lr = alpha_optimizer.param_groups[0]['lr']
    alpha_scheduler.step(phys_loss)  # 基于物理损失调整alpha
    new_alpha_lr = alpha_optimizer.param_groups[0]['lr']

    # 手动打印学习率变化
    if old_model_lr != new_model_lr:
        print(f'Epoch {epoch + 1}: Model LR reduced from {old_model_lr:.8f} to {new_model_lr:.8f}')
    if old_alpha_lr != new_alpha_lr:
        print(f'Epoch {epoch + 1}: Alpha LR reduced from {old_alpha_lr:.8f} to {new_alpha_lr:.8f}')

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.8f}, '
              f'α: {alpha_val:.8f}, '
              f'Init Loss: {init_loss:.8f}, BC Loss: {bc_loss:.8f}, '
              f'Sol Loss: {sol_loss:.8f}, Phys Loss: {phys_loss:.8f}')

# 可视化训练过程（增加物理损失子图）
plt.figure(figsize=(25, 5))
plt.subplot(1, 5, 1)
plt.plot(loss_history)
plt.title('Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')

plt.subplot(1, 5, 2)
plt.plot(phys_loss_history)
plt.title('Physics Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')

plt.subplot(1, 5, 3)
plt.plot(alpha_history)
plt.axhline(y=0.8, color='r', linestyle='--', label='True α')
plt.title('Estimated α')
plt.xlabel('Epoch')
plt.ylabel('α')
plt.legend()

plt.subplot(1, 5, 4)
plt.plot(model_lr_history)
plt.title('Model Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('LR')
plt.yscale('log')

plt.subplot(1, 5, 5)
plt.plot(alpha_lr_history)
plt.title('Alpha Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('LR')
plt.yscale('log')

plt.tight_layout()
plt.show()

# 打印最终结果
true_alpha = 0.8
print(f"真实热传导系数 alpha = {true_alpha}")
print(f"PINN估计的热传导系数 alpha = {alpha_param.item():.8f}")
print(f"相对误差: {abs(alpha_param.item() - true_alpha) / true_alpha * 100:.2f}%")

# 可选：保存模型和估计的alpha值
torch.save({
    'model_state_dict': model.state_dict(),
    'alpha': alpha_param.item(),
    'optimizer_state_dict': model_optimizer.state_dict(),
}, 'pinn_heat_equation_model.pth')
print("模型已保存至 pinn_heat_equation_model.pth")