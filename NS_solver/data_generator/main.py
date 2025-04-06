import torch

# 生成 20 个随机的 (x, y, t) 点
x = torch.rand(20)
y = torch.rand(20)
t = torch.rand(20)

# 生成对应的 (u, v, p) 数据，这里简单随机生成
u = torch.rand(20)
v = torch.rand(20)
p = torch.rand(20)

# 保存数据到文件
torch.save({
    'x': x,
    'y': y,
    't': t,
    'u': u,
    'v': v,
    'p': p
}, 'experimental_data.pt')

# 加载文件
data = torch.load('experimental_data.pt')

# 输出提示信息
print("输出为 20 组 (x, y, t) 及其对应的 (u, v, p) 数据，每组数据用空格分隔，每组之间用换行分隔：")
# 按格式输出数据
for i in range(20):
    print(f"{data['x'][i].item():.4f} {data['y'][i].item():.4f} {data['t'][i].item():.4f} {data['u'][i].item():.4f} {data['v'][i].item():.4f} {data['p'][i].item():.4f}")