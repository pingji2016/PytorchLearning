import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 如果是 CUDA，打印显卡信息
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# --------------------------
# 1. 数据准备
# --------------------------
def load_data(batch_size=64):
    # 定义数据转换：转换为 Tensor 并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST 的均值和标准差
    ])

    # 下载并加载训练集
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                             shuffle=True)

    # 下载并加载测试集
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000,
                                            shuffle=False)
    
    return train_loader, test_loader

# --------------------------
# 2. 定义网络结构 (经典 CNN)
# --------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 第一层卷积：输入 1 通道，输出 32 通道，卷积核 3x3
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 第二层卷积：输入 32 通道，输出 64 通道，卷积核 3x3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Dropout 防止过拟合
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # 全连接层 1
        self.fc1 = nn.Linear(9216, 128)
        # 全连接层 2 (输出层，10 个类别)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Conv1 -> ReLU -> Conv2 -> ReLU -> MaxPool -> Dropout
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # Flatten 平铺
        x = torch.flatten(x, 1)
        
        # FC1 -> ReLU -> Dropout -> FC2
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # LogSoftmax 输出概率对数
        output = F.log_softmax(x, dim=1)
        return output

# --------------------------
# 3. 训练函数
# --------------------------
def train(model, device, train_loader, optimizer, epoch, train_losses):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad() # 清空梯度
        output = model(data)  # 前向传播
        loss = F.nll_loss(output, target) # 计算损失
        loss.backward()       # 反向传播
        optimizer.step()      # 更新参数
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            train_losses.append(loss.item())

# --------------------------
# 4. 测试函数
# --------------------------
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(): # 测试时不需要计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # 累加损失
            pred = output.argmax(dim=1, keepdim=True) # 获取最大概率的索引
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')

# --------------------------
# 5. 辅助功能：绘图与图片保存
# --------------------------
def plot_losses(losses):
    if not losses:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss over Time')
    plt.xlabel('Iterations (x100 batches)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    print("Loss curve saved to training_loss.png")
    # plt.show() # 如果在非 GUI 环境下运行，可以注释掉这行

def save_and_predict_images(model, device, test_loader, num_images=2):
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    print(f"\nExtracting {num_images} images for testing...")
    
    for i in range(num_images):
        img_tensor = images[i]
        label = labels[i].item()
        
        # 反归一化并保存图片
        img_np = img_tensor.squeeze().numpy()
        img_np = img_np * 0.3081 + 0.1307 # 反归一化
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        
        img_pil = Image.fromarray(img_np)
        img_filename = f'test_image_{i+1}.png'
        img_pil.save(img_filename)
        print(f"Saved {img_filename} (True Label: {label})")
        
        # 预测
        with torch.no_grad():
            img_input = img_tensor.unsqueeze(0).to(device)
            output = model(img_input)
            pred = output.argmax(dim=1).item()
            print(f"Prediction for {img_filename}: {pred}")


# --------------------------
# 主函数
# --------------------------
def main():
    # 超参数
    BATCH_SIZE = 64
    EPOCHS = 3 # 训练轮数
    LEARNING_RATE = 1.0
    GAMMA = 0.7
    MODEL_PATH = "mnist_cnn.pt"

    # 加载数据
    print("Loading MNIST data...")
    train_loader, test_loader = load_data(BATCH_SIZE)

    # 初始化模型
    model = Net().to(device)
    
    # 检查模型是否存在
    if os.path.exists(MODEL_PATH):
        print(f"\nFound existing model: {MODEL_PATH}")
        print("Loading model and skipping training...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
        # 直接进行测试
        test(model, device, test_loader)
    else:
        print("\nNo existing model found. Starting training...")
        # 优化器
        optimizer = optim.Adadelta(model.parameters(), lr=LEARNING_RATE)
        # 学习率调度器
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=GAMMA)

        # 记录损失
        train_losses = []

        # 循环训练
        for epoch in range(1, EPOCHS + 1):
            train(model, device, train_loader, optimizer, epoch, train_losses)
            test(model, device, test_loader)
            scheduler.step()

        # 绘制损失曲线
        plot_losses(train_losses)

        # 保存模型
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    # 保存并预测测试图片 (无论是否训练，都执行这一步)
    save_and_predict_images(model, device, test_loader, num_images=2)

if __name__ == '__main__':
    main()
