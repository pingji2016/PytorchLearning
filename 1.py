import os
# 解决 Windows + 某些科学计算库的 OpenMP 运行时重复加载问题（否则可能直接报错退出）
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

# 选择运行设备：如果当前 PyTorch 构建支持 CUDA 且检测到显卡，则用 cuda；否则用 cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 如果是 CUDA，打印显卡信息（方便确认确实在用 GPU）
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# --------------------------
# 1. 数据准备
# --------------------------
def load_data(batch_size=64):
    # 定义数据转换：
    # 1) ToTensor: 把 [0,255] 的灰度图转成 [0,1] 的 float Tensor，形状为 [1, 28, 28]
    # 2) Normalize: 做标准化 x' = (x - mean) / std，使训练更稳定
    #    这里 mean=0.1307、std=0.3081 是 MNIST 数据集统计出来的常用值
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST 的均值和标准差
    ])

    # MNIST 数据集对象：第一次会下载到 ./data，之后直接读取本地缓存
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                             download=True, transform=transform)
    # DataLoader 负责：
    # - 每次给你一个 batch 的 (images, labels)
    # - shuffle=True 会在每个 epoch 打乱训练集顺序
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
        # 第一层卷积：输入 1 通道（灰度图），输出 32 通道
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 第二层卷积：输入 32 通道，输出 64 通道
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Dropout：训练时随机丢弃部分神经元，降低过拟合风险
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # 经过两层卷积 + 池化后，特征图展平为 9216 维（64 * 12 * 12）
        self.fc1 = nn.Linear(9216, 128)
        # 输出层：10 类（0~9）
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 前向传播：
        # Conv -> ReLU -> Conv -> ReLU -> MaxPool -> Dropout -> Flatten -> FC -> ReLU -> Dropout -> FC -> LogSoftmax
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        # 2x2 最大池化：把空间尺寸缩小一半（降低计算量，并保留最强特征）
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # Flatten：从 [N, C, H, W] 变成 [N, C*H*W]
        x = torch.flatten(x, 1)
        
        # 全连接分类头
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # LogSoftmax：输出每个类别的对数概率（配合 nll_loss 使用）
        output = F.log_softmax(x, dim=1)
        return output

# --------------------------
# 3. 训练函数
# --------------------------
def train(model, device, train_loader, optimizer, epoch, train_losses):
    # 训练模式：启用 Dropout 等训练行为
    model.train()
    # train_loader 会“批量”产出数据：
    # - batch_idx: 第几个 batch
    # - data: 这一批图片张量，形状通常是 [batch_size, 1, 28, 28]
    # - target: 这一批图片标签，形状通常是 [batch_size]，值是 0~9
    for batch_idx, (data, target) in enumerate(train_loader):
        # 把数据搬到 device（GPU/CPU）
        data, target = data.to(device), target.to(device)
        
        # 训练一步的标准流程：清梯度 -> 前向 -> 算 loss -> 反向 -> 更新参数
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            # 这里每 100 个 batch 记录一次 loss（所以 loss 曲线的 x 轴不是 epoch）
            train_losses.append(loss.item())

# --------------------------
# 4. 测试函数
# --------------------------
def test(model, device, test_loader):
    # eval 模式：关闭 Dropout 等随机性，保证推理稳定
    model.eval()
    test_loss = 0
    correct = 0
    # no_grad：测试/推理时不需要梯度，可节省显存和加速
    with torch.no_grad():
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
    # x 轴：第几次“记录 loss”的点。因为我们是每 100 个 batch 记录一次，所以这里标成 x100 batches
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
        
        # 反归一化并保存图片：
        # 训练/加载数据时做了 Normalize: x' = (x - mean) / std
        # 现在要保存成“正常可看的图片”，需要做逆变换：x = x' * std + mean
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

def save_train_samples(train_loader, num_images=8, out_dir="train_samples"):
    # 从训练集取一个 batch，导出 num_images 张，方便快速看“训练数据长什么样”
    os.makedirs(out_dir, exist_ok=True)
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    count = min(num_images, images.size(0))
    for i in range(count):
        img_tensor = images[i].cpu()
        label = labels[i].item()
        img_np = img_tensor.squeeze().numpy()
        # 这里同样是“反归一化”，把标准化后的像素值还原成接近原图的 [0,1] 范围
        img_np = img_np * 0.3081 + 0.1307
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        img_filename = os.path.join(out_dir, f"train_{i+1}_label_{label}.png")
        img_pil.save(img_filename)

def save_one_per_digit(train_loader, out_dir="train_samples_0to9"):
    # 在训练集里“扫一遍”，直到找到 0~9 每个数字各一张，然后保存出来
    os.makedirs(out_dir, exist_ok=True)
    found = {}
    for images, labels in train_loader:
        for i in range(images.size(0)):
            label = int(labels[i].item())
            if label in found:
                continue
            img_tensor = images[i].cpu()
            img_np = img_tensor.squeeze().numpy()
            img_np = img_np * 0.3081 + 0.1307
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            Image.fromarray(img_np).save(os.path.join(out_dir, f"train_digit_{label}.png"))
            found[label] = True
            if len(found) == 10:
                return


# --------------------------
# 主函数
# --------------------------
def main():
    # 超参数（训练时需要手动设定/可调的参数）
    BATCH_SIZE = 64
    EPOCHS = 3 # 训练轮数：完整遍历训练集 EPOCHS 次
    LEARNING_RATE = 1.0
    GAMMA = 0.7
    MODEL_PATH = "mnist_cnn.pt"

    # 加载数据
    print("Loading MNIST data...")
    train_loader, test_loader = load_data(BATCH_SIZE)
    save_train_samples(train_loader, num_images=8, out_dir="train_samples")
    save_one_per_digit(train_loader, out_dir="train_samples_0to9")

    # 初始化模型
    model = Net().to(device)
    
    # 检查模型是否存在
    if os.path.exists(MODEL_PATH):
        print(f"\nFound existing model: {MODEL_PATH}")
        print("Loading model and skipping training...")
        # map_location=device：即使 .pt 是在别的设备上保存的，也能加载到当前 device
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
        # 直接进行测试
        test(model, device, test_loader)
    else:
        print("\nNo existing model found. Starting training...")
        # 优化器：Adadelta 对学习率相对不那么敏感，适合作为入门示例
        optimizer = optim.Adadelta(model.parameters(), lr=LEARNING_RATE)
        # 学习率调度器：每个 epoch 结束把学习率乘以 GAMMA（逐步减小步长）
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
