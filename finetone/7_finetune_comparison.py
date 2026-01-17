import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import copy
import time
from torch.utils.data import DataLoader, Subset
import numpy as np

# 复用 Net 定义
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

def train_and_evaluate(model, dataloader, strategy_name, epochs=3):
    device = next(model.parameters()).device
    # 策略 A: 冻结卷积层
    if strategy_name == "Strategy A (Frozen Conv)":
        print("  -> Freezing Conv layers...")
        for param in model.conv1.parameters(): param.requires_grad = False
        for param in model.conv2.parameters(): param.requires_grad = False
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, momentum=0.9)
    else:
        # 策略 B: 全量微调
        print("  -> Fine-tuning all layers...")
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.train()
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        total_loss = 0
        correct = 0
        total = 0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        avg_loss = total_loss / len(dataloader)
        acc = 100. * correct / total
        print(f"  Epoch {epoch}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")
        
    duration = time.time() - start_time
    print(f"  Training time: {duration:.2f}s")
    return acc, duration

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 1. 准备数据 (少量数据模拟微调场景)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(script_dir, '..', 'data')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_dataset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    
    # 随机选 200 张作为“极少样本”场景，看哪种策略好
    indices = np.random.choice(len(full_dataset), 200, replace=False)
    new_dataset = Subset(full_dataset, indices)
    new_dataloader = DataLoader(new_dataset, batch_size=16, shuffle=True)
    print(f"Fine-tuning on {len(new_dataset)} samples...")

    # 2. 加载基础模型权重
    pretrained_path = os.path.join(script_dir, '..', 'mnist_cnn.pt')
    if not os.path.exists(pretrained_path):
        print("Please run 1.py first to generate mnist_cnn.pt")
        return
        
    base_state_dict = torch.load(pretrained_path, map_location=device)

    # ==========================================
    # 对比实验
    # ==========================================
    
    # 实验 1: 策略 A (冻结卷积层，只训练 FC)
    print("\n--- Strategy A: Freeze Conv Layers (Train FC only) ---")
    model_a = Net().to(device)
    model_a.load_state_dict(base_state_dict)
    acc_a, time_a = train_and_evaluate(model_a, new_dataloader, "Strategy A (Frozen Conv)")

    # 实验 2: 策略 B (微调所有层，低学习率)
    print("\n--- Strategy B: Fine-tune All Layers (Low LR) ---")
    model_b = Net().to(device)
    model_b.load_state_dict(base_state_dict)
    acc_b, time_b = train_and_evaluate(model_b, new_dataloader, "Strategy B (Full Fine-tune)")

    print("\n" + "="*40)
    print("Summary:")
    print(f"Strategy A (Frozen): Acc={acc_a:.2f}%, Time={time_a:.2f}s")
    print(f"Strategy B (Full)  : Acc={acc_b:.2f}%, Time={time_b:.2f}s")
    print("="*40)
    print("Analysis:")
    print("1. Frozen strategy is usually faster and less prone to overfitting on very small data.")
    print("2. Full fine-tuning usually yields higher accuracy if data is sufficient.")

if __name__ == '__main__':
    main()
