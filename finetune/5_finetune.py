import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader, Subset
import numpy as np

# 复用 1.py 中的 Net 定义 (为了保证结构一致，实际项目中可以 import)
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

def fine_tune():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 1. 加载预训练模型
    model = Net().to(device)
    
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 假设模型在上一级目录
    pretrained_path = os.path.join(script_dir, '..', 'mnist_cnn.pt')
    
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}...")
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
    else:
        print(f"Pretrained model not found at {pretrained_path}! Please run 1.py first.")
        return

    # 2. 准备“新数据”
    # 为了模拟新数据，我们随机选取 MNIST 的一小部分 (例如 1000 张图片)
    # 假设这是我们收集到的新场景下的手写数字
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 数据集目录也在上一级
    data_root = os.path.join(script_dir, '..', 'data')
    full_dataset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    
    # 随机选 1000 张作为“新数据集”
    indices = np.random.choice(len(full_dataset), 1000, replace=False)
    new_dataset = Subset(full_dataset, indices)
    new_dataloader = DataLoader(new_dataset, batch_size=32, shuffle=True)
    
    print(f"Fine-tuning on {len(new_dataset)} new samples...")

    # 3. 设置微调策略
    # 策略 A: 冻结卷积层，只训练全连接层 (适合数据量很少的情况)
    # for param in model.conv1.parameters(): param.requires_grad = False
    # for param in model.conv2.parameters(): param.requires_grad = False
    
    # 策略 B: 训练所有层，但使用很小的学习率 (适合数据量适中，希望整体微调)
    # 这里我们使用策略 B
    
    # 注意学习率 lr 比从头训练时 (通常 0.01 或 1.0) 要小很多
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 4. 开始微调训练
    model.train()
    epochs = 2
    for epoch in range(1, epochs + 1):
        total_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(new_dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
        avg_loss = total_loss / len(new_dataloader)
        accuracy = 100. * correct / len(new_dataset)
        print(f"Fine-tune Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")

    # 5. 保存微调后的模型
    save_path = "mnist_cnn_finetuned.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Fine-tuned model saved to {save_path}")

if __name__ == '__main__':
    fine_tune()
