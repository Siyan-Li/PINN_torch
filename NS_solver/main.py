import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义神经网络
class NSModel(nn.Module):
    def __init__(self):
        super(NSModel, self).__init__()
        self.fc1 = nn.Linear(3, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 3)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x


# 定义参数
rho = 1.0
nu = 0.1
model = NSModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002)
num_epochs = 1500

# 提示用户输入测试数据
print("请输入 20 组 (x, y, t) 及其对应的 (u, v, p) 数据，每组数据用空格分隔，每组之间用换行分隔：")
data = []
for _ in range(20):
    line = input().strip().split()
    x, y, t, u, v, p = map(float, line)
    data.append([x, y, t, u, v, p])

data = np.array(data)
XYT_exp = torch.tensor(data[:, :3], dtype=torch.float32).to(device)
u_exp = torch.tensor(data[:, 3:4], dtype=torch.float32).to(device)
v_exp = torch.tensor(data[:, 4:5], dtype=torch.float32).to(device)
p_exp = torch.tensor(data[:, 5:6], dtype=torch.float32).to(device)

# 生成训练用的网格点
x_train = torch.linspace(0, 1, 20)
y_train = torch.linspace(0, 1, 20)
t_train = torch.linspace(0, 1, 20)
X_train, Y_train, T_train = torch.meshgrid(x_train, y_train, t_train, indexing='ij')
X_train = X_train.reshape(-1, 1)
Y_train = Y_train.reshape(-1, 1)
T_train = T_train.reshape(-1, 1)
XYT_train = torch.cat([X_train, Y_train, T_train], dim=1).to(device)

# 训练模型
for epoch in range(num_epochs):
    # 根据epoch调整学习率
    if epoch < 500:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01
    if 500 <= epoch < 1000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.005
    else :
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.002

    # 计算物理方程损失
    XYT_train.requires_grad = True
    output_train = model(XYT_train)
    u_train = output_train[:, 0:1]
    v_train = output_train[:, 1:2]
    p_train = output_train[:, 2:3]

    # 计算偏导数
    du_dXYT = torch.autograd.grad(u_train, XYT_train, grad_outputs=torch.ones_like(u_train), create_graph=True)[0]
    dv_dXYT = torch.autograd.grad(v_train, XYT_train, grad_outputs=torch.ones_like(v_train), create_graph=True)[0]
    dp_dXYT = torch.autograd.grad(p_train, XYT_train, grad_outputs=torch.ones_like(p_train), create_graph=True)[0]

    du_dx = du_dXYT[:, 0:1]
    du_dy = du_dXYT[:, 1:2]
    du_dt = du_dXYT[:, 2:3]

    dv_dx = dv_dXYT[:, 0:1]
    dv_dy = dv_dXYT[:, 1:2]
    dv_dt = dv_dXYT[:, 2:3]

    dp_dx = dp_dXYT[:, 0:1]
    dp_dy = dp_dXYT[:, 1:2]

    du_dxx = torch.autograd.grad(du_dx, XYT_train, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0][:, 0:1]
    du_dyy = torch.autograd.grad(du_dy, XYT_train, grad_outputs=torch.ones_like(du_dy), create_graph=True)[0][:, 1:2]
    dv_dxx = torch.autograd.grad(dv_dx, XYT_train, grad_outputs=torch.ones_like(dv_dx), create_graph=True)[0][:, 0:1]
    dv_dyy = torch.autograd.grad(dv_dy, XYT_train, grad_outputs=torch.ones_like(dv_dy), create_graph=True)[0][:, 1:2]

    # 连续性方程损失
    continuity_loss = torch.mean((du_dx + dv_dy) ** 2)

    # 动量方程损失
    momentum_x_loss = torch.mean(
        (du_dt + u_train * du_dx + v_train * du_dy + (1 / rho) * dp_dx - nu * (du_dxx + du_dyy)) ** 2)
    momentum_y_loss = torch.mean(
        (dv_dt + u_train * dv_dx + v_train * dv_dy + (1 / rho) * dp_dy - nu * (dv_dxx + dv_dyy)) ** 2)

    # 实验数据损失
    output_exp = model(XYT_exp)
    u_pred_exp = output_exp[:, 0:1]
    v_pred_exp = output_exp[:, 1:2]
    p_pred_exp = output_exp[:, 2:3]
    data_loss = nn.MSELoss()(u_pred_exp, u_exp) + nn.MSELoss()(v_pred_exp, v_exp) + nn.MSELoss()(p_pred_exp, p_exp)

    # 总损失
    loss = continuity_loss + momentum_x_loss + momentum_y_loss + data_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 可视化结果
x_test = torch.linspace(0, 1, 50)
y_test = torch.linspace(0, 1, 50)
t_test = torch.tensor([0.5]).repeat(50 * 50)
X_test, Y_test = torch.meshgrid(x_test, y_test, indexing='ij')
X_test = X_test.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)
XYT_test = torch.cat([X_test, Y_test, t_test.view(-1, 1)], dim=1).to(device)
output_test = model(XYT_test).detach().cpu().numpy()
u_test = output_test[:, 0].reshape(50, 50)
v_test = output_test[:, 1].reshape(50, 50)
p_test = output_test[:, 2].reshape(50, 50)

# 绘制速度场 u
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(u_test, extent=[0, 1, 0, 1], origin='lower')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Velocity u at t = 0.5')

# 绘制速度场 v
plt.subplot(132)
plt.imshow(v_test, extent=[0, 1, 0, 1], origin='lower')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Velocity v at t = 0.5')

# 绘制压力场 p
plt.subplot(133)
plt.imshow(p_test, extent=[0, 1, 0, 1], origin='lower')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Pressure p at t = 0.5')

plt.tight_layout()
plt.show()

# 让用户输入任意 (x, y, t) 进行预测
while True:
    try:
        input_str = input("请输入范围内的 (x, y, t) 数据（用空格分隔），输入 'q' 退出：")
        if input_str.lower() == 'q':
            break
        x, y, t = map(float, input_str.split())
        if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= t <= 1:
            input_tensor = torch.tensor([[x, y, t]], dtype=torch.float32).to(device)
            output = model(input_tensor).detach().cpu().numpy()[0]
            u_pred, v_pred, p_pred = output
            print(f"预测的 (u, v, p) 值为: ({u_pred:.4f}, {v_pred:.4f}, {p_pred:.4f})")
        else:
            print("输入的数据不在 [0, 1] 范围内，请重新输入。")
    except ValueError:

        print("输入格式错误，请输入三个用空格分隔的数字。")
