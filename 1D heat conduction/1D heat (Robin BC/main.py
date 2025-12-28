import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# 定义热传导方程PINN模型
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(2, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义热传导方程的残差损失
def pde_loss(model, x, t, alpha):
    x.requires_grad = True
    t.requires_grad = True
    inputs = torch.cat([x, t], dim=1)
    u = model(inputs)
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dt = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
    residual = du_dt - alpha * d2u_dx2
    return torch.mean(residual ** 2)


# 定义初始条件和边界条件损失
def icbc_loss(model, x_ic, t_ic, u_ic, x_bc1, t_bc1, x_bc2, t_bc2, h, q0, q1, alpha):
    # 初始条件损失
    inputs_ic = torch.cat([x_ic, t_ic], dim=1)
    u_pred_ic = model(inputs_ic)
    loss_ic = torch.mean((u_pred_ic - u_ic) ** 2)

    # 边界条件损失（罗宾边界条件）
    x_bc1.requires_grad = True
    x_bc2.requires_grad = True
    inputs_bc1 = torch.cat([x_bc1, t_bc1], dim=1)
    inputs_bc2 = torch.cat([x_bc2, t_bc2], dim=1)
    u_bc1 = model(inputs_bc1)
    u_bc2 = model(inputs_bc2)
    du_dx_bc1 = torch.autograd.grad(u_bc1, x_bc1, grad_outputs=torch.ones_like(u_bc1), create_graph=True)[0]
    du_dx_bc2 = torch.autograd.grad(u_bc2, x_bc2, grad_outputs=torch.ones_like(u_bc2), create_graph=True)[0]

    # 罗宾边界条件损失
    loss_bc1 = torch.mean((alpha * du_dx_bc1 + h * u_bc1 - q0) ** 2)
    loss_bc2 = torch.mean((alpha * du_dx_bc2 + h * u_bc2 - q1) ** 2)

    return loss_ic + loss_bc1 + loss_bc2


# 生成训练数据
def generate_data():
    # 空间和时间范围
    x_min, x_max = 0, 1
    t_min, t_max = 0, 1
    # 初始条件数据
    x_ic = torch.linspace(x_min, x_max, 100).view(-1, 1)
    t_ic = torch.zeros_like(x_ic)
    u_ic = torch.sin(np.pi * x_ic)
    # 边界条件数据
    t_bc = torch.linspace(t_min, t_max, 100).view(-1, 1)
    t_bc1 = t_bc
    t_bc2 = t_bc
    x_bc1 = x_min * torch.ones_like(t_bc)
    x_bc2 = x_max * torch.ones_like(t_bc)
    # 罗宾边界条件参数
    h = 0.1  # 热交换系数
    q0 = torch.full_like(t_bc,0.4)
    q1 = torch.full_like(t_bc,0.6)
    # 内部数据
    x = torch.rand(1000, 1) * (x_max - x_min) + x_min
    t = torch.rand(1000, 1) * (t_max - t_min) + t_min

    return x, t, x_ic, t_ic, u_ic, x_bc1, t_bc1, x_bc2, t_bc2, h, q0, q1


# 训练模型
def train(model, alpha, epochs=1000, lr_model=0.01, lr_alpha=0.00001):
    alpha = torch.tensor(alpha, requires_grad=True)
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': lr_model},
        {'params': [alpha], 'lr': lr_alpha}
    ])
    x, t, x_ic, t_ic, u_ic, x_bc1, t_bc1, x_bc2, t_bc2, h, q0, q1 = generate_data()

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_pde = pde_loss(model, x, t, alpha)
        loss_icbc = icbc_loss(model, x_ic, t_ic, u_ic, x_bc1, t_bc1, x_bc2, t_bc2, h, q0, q1, alpha)
        loss = loss_pde + loss_icbc
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Alpha: {alpha.item():.4f}')

    return model, alpha


# 主函数
if __name__ == "__main__":
    # 真实的阿尔法值
    alpha_true = 0.1
    # 初始化PINN模型
    model = PINN()
    # 训练模型
    model, alpha_pred = train(model, alpha_true)

    print(f'True alpha: {alpha_true}, Predicted alpha: {alpha_pred.item():.4f}')

    # 可视化结果
    x_test = torch.linspace(0, 1, 100).view(-1, 1)
    t_test = torch.ones_like(x_test) * 0.5
    inputs_test = torch.cat([x_test, t_test], dim=1)
    u_pred = model(inputs_test).detach().numpy()
    plt.plot(x_test.numpy(), u_pred)
    plt.xlabel('x')
    plt.ylabel('u(x, t=0.5)')
    plt.title('PINN solution of the heat equation with Robin BC')
    plt.show()

