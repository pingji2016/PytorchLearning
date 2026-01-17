import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
import numpy as np
import copy

import os

# 复用 1.py 中的 Net 定义
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def set_seed(seed=42):
    """设置随机种子以保证结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保证 CUDA 卷积操作的确定性 (会牺牲一点性能)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(device, seed=None):
    # 如果指定了 seed，则设置它
    if seed is not None:
        set_seed(seed)
        
    # 初始化模型 (权重是随机初始化的，除非设置了 seed)
    model = Net().to(device)
    
    # 准备数据 (为了速度，只用一小部分数据训练)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 数据集目录在上一级
    data_root = os.path.join(script_dir, '..', 'data')
    
    # 仅使用前 1000 张图片训练，以加快演示速度
    dataset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    subset_indices = list(range(1000)) 
    subset = torch.utils.data.Subset(dataset, subset_indices)
    
    # DataLoader 的 shuffle 也会受 seed 影响
    loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=True)
    
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
    return model

def compare_models(model1, model2):
    """比较两个模型的权重相似度"""
    print("\nComparing models...")
    diff_sum = 0
    param_count = 0
    
    dict1 = model1.state_dict()
    dict2 = model2.state_dict()
    
    for key in dict1:
        w1 = dict1[key]
        w2 = dict2[key]
        diff = (w1 - w2).abs().sum().item()
        diff_sum += diff
        param_count += w1.numel()
        
        if diff > 0.0001:
            print(f"  Layer {key} differs! Sum diff: {diff:.6f}")
    
    if diff_sum == 0:
        print("=> Result: Models are IDENTICAL (Perfect Match).")
    else:
        print(f"=> Result: Models are DIFFERENT. Total param difference: {diff_sum:.6f}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("-" * 60)
    print("Experiment 1: Training twice WITHOUT fixed seed (Default behavior)")
    print("-" * 60)
    print("Training Model A...")
    model_a = train_one_epoch(device, seed=None)
    print("Training Model B...")
    model_b = train_one_epoch(device, seed=None)
    compare_models(model_a, model_b)

    print("\n" + "-" * 60)
    print("Experiment 2: Training twice WITH fixed seed (seed=42)")
    print("-" * 60)
    print("Training Model C (Seed 42)...")
    model_c = train_one_epoch(device, seed=42)
    print("Training Model D (Seed 42)...")
    model_d = train_one_epoch(device, seed=42)
    compare_models(model_c, model_d)

if __name__ == '__main__':
    main()
