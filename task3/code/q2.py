import torch
import imageio as imio
import os

from model import CNN

model = CNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型参数，注意将文件名改为你训练时保存的模型文件名
model.load_state_dict(torch.load("/mnt/f/大学/A大二上/2025焦糖工作室招新/task3/code/q1_model.pt", weights_only=True))

model.eval()

# TODO：获取 conv1 layer 的权重
conv_weights = model.conv1.weight.data

os.makedirs("q2_filters", exist_ok=True)

for i in range(conv_weights.shape[0]):
    # TODO: 获取第 i 个卷积核
    f = conv_weights[i]  # 获取第 i 个卷积核，形状为 (3, 7, 7)

    # TODO: 将卷积核归一化到 [0, 255] 并转换为 uint8 类型
    # 归一化到 [0, 1] 范围
    f_min = f.min()
    f_max = f.max()
    if f_max > f_min:  # 避免除零错误
        f = (f - f_min) / (f_max - f_min)
    else:
        f = torch.zeros_like(f)
    
    # 转换到 [0, 255] 范围并转为 uint8
    f = (f * 255).clamp(0, 255).byte()
    
    # 将 tensor 转换为 numpy 数组并调整维度顺序 (C, H, W) -> (H, W, C)
    f_numpy = f.permute(1, 2, 0).numpy()

    imio.imwrite(f"q2_filters/filter_{i}.png", f_numpy)
