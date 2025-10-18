import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import functional

torch.manual_seed(123)


def create_dataloaders():
    train_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.Lambda(lambda img: functional.crop(img, 0, 0, 134, 134)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ]) # TODO: 定义训练集的数据预处理与增强
    val_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.Lambda(lambda img: functional.crop(img, 0, 0, 134, 134)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])  # TODO: 定义验证集的数据预处理

    train_dataset = ImageFolder(root= "custom_image_dataset/train", transform=train_tf) # TODO: 加载训练集，并确保应用训练集的 transform
    val_dataset = ImageFolder(root= "custom_image_dataset/val", transform=val_tf) # TODO: 加载验证集

    train_loader = DataLoader(
        train_dataset, 
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    ) # TODO: 创建训练集 dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,  
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    ) # TODO: 创建验证集 dataloader

    return train_loader, val_loader

