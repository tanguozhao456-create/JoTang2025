import torch
import os

from PIL import Image
from glob import glob

from torchvision import transforms
from torchvision.transforms import functional

from model import CNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
weights_path = os.path.join(script_dir, "q1_model.pt")
test_glob_img = os.path.join(script_dir, "custom_image_dataset", "test_unlabeled", "img_*.png")
test_glob_image = os.path.join(script_dir, "custom_image_dataset", "test_unlabeled", "image_*.png")
output_path = os.path.join(project_root, "q1_test.txt")

# 构建模型并加载权重（映射到当前设备）
model = CNN()
model.to(device)
model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
model.eval()

def _extract_index(file_path: str) -> int:
    name = os.path.splitext(os.path.basename(file_path))[0]  # e.g. img_123 or image_0456
    # 取最后的数字部分
    for part in reversed(name.split("_")):
        if part.isdigit():
            return int(part)
    return 0

# 合并两类通配符的结果并去重
all_candidates = glob(test_glob_img) + glob(test_glob_image)
seen = set()
deduped = []
for p in all_candidates:
    if p not in seen:
        seen.add(p)
        deduped.append(p)

# 按数字索引排序，确保与 image_0001.png... 对应顺序一致
test_images = sorted(deduped, key=_extract_index)

# TODO: 创建测试时的图像 transformations.
test_tf = transforms.Compose([
    transforms.Resize(224),
    transforms.Lambda(lambda img: functional.crop(img, 0, 0, 134, 134)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])

with open(output_path, "w", encoding="utf-8", newline="\n") as f:
    for idx, imgfile in enumerate(test_images, start=1):
        # 统一输出为 image_XXXX.png（1-9999 使用 4 位零填充，10000 显示为 10000）
        filename = f"image_{idx:04d}.png"
        img = Image.open(imgfile).convert("RGB")
        img = test_tf(img)
        img = img.unsqueeze(0).to(device)
        # TODO: 使模型进行前向传播并获取预测标签，predicted 是一个 PyTorch 张量，包含预测的标签，值为 0 到 9 之间的单个整数（包含 0 和 9）
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)

        f.write(f"{filename},{int(predicted.item())}\n")