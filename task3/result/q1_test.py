import torch
import os

from PIL import Image
from glob import glob

from torchvision import transforms
from torchvision.transforms import functional

from model import CNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN()
model.to(device)
model.load_state_dict(torch.load("q1_model.pt", weights_only=True))
model.eval()

test_images = sorted(glob("custom_image_dataset/test_unlabeled/*.png"))

# TODO: 创建测试时的图像 transformations.
test_tf = transforms.Compose([
    transforms.Resize(224),
    transforms.Lambda(lambda img: functional.crop(img, 0, 0, 134, 134)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])

test_write = open("q1_test.txt", "w")
for imgfile in test_images:
    filename = os.path.basename(imgfile)
    img = Image.open(imgfile)
    img = test_tf(img)
    img = img.unsqueeze(0).to(device)

    # TODO: 使模型进行前向传播并获取预测标签，predicted 是一个 PyTorch 张量，包含预测的标签，值为 0 到 9 之间的单个整数（包含 0 和 9）
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    test_write.write(f"{filename},{predicted.item()}\n")
test_write.close()