import os
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入为 MNIST 单通道图像 [1, 28, 28]
        # 第一层卷积：提取低级边缘/纹理特征
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 第二层卷积：在更高层次上组合特征
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Dropout：训练时随机丢弃部分神经元，缓解过拟合
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # 经过两次 3x3 卷积和一次 2x2 池化后
        # 特征图大小变为 64 x 12 x 12 = 9216
        self.fc1 = nn.Linear(9216, 128)
        # 输出 10 类（0-9）
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 形状: [batch, 1, 28, 28] -> [batch, 32, 26, 26]
        x = F.relu(self.conv1(x))
        # -> [batch, 64, 24, 24]
        x = F.relu(self.conv2(x))
        # -> [batch, 64, 12, 12]
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        # 展平为向量供全连接层使用
        x = torch.flatten(x, 1)
        # 全连接层提取全局特征
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        # 输出未归一化 logits，后续用交叉熵损失计算
        x = self.fc2(x)
        return x


def get_device():
    # 优先使用 GPU，提高训练速度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        # 打印显卡名称，方便确认环境
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def build_loader(data_root, batch_size, train, subset_size, seed):
    # MNIST 标准化：均值 0.1307，标准差 0.3081
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    # 下载并构建 MNIST 数据集
    dataset = torchvision.datasets.MNIST(
        root=data_root, train=train, download=True, transform=transform
    )
    # subset_size > 0 时，随机抽取部分样本用于快速实验
    if subset_size and subset_size < len(dataset):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(dataset), subset_size, replace=False)
        dataset = Subset(dataset, indices)
    # 训练集需要打乱，测试集保持顺序
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def train_one_epoch(model, device, train_loader, optimizer, epoch):
    # 切换到训练模式：启用 Dropout、BatchNorm 的训练行为
    model.train()
    total_loss = 0.0
    for batch_index, (data, target) in enumerate(train_loader, start=1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # 前向传播得到 logits
        output = model(data)
        # 交叉熵损失，内部自带 softmax
        loss = F.cross_entropy(output, target)
        # 反向传播与参数更新
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_index % 100 == 0:
            print(
                f"Epoch {epoch} Batch {batch_index}/{len(train_loader)} "
                f"Loss {loss.item():.4f}"
            )
    # 计算平均损失便于观察收敛趋势
    average_loss = total_loss / max(1, len(train_loader))
    print(f"Epoch {epoch} Avg Loss {average_loss:.4f}")
    return average_loss


def evaluate(model, device, test_loader):
    # 切换到评估模式：关闭 Dropout 等随机性
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # 统计总损失，最后除以样本数得到平均值
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            # 取最大 logits 的索引作为预测类别
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"Test set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)"
    )
    return test_loss, accuracy


def parse_args():
    # 命令行参数，便于控制训练超参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--model-path", type=str, default="")
    return parser.parse_args()


def main():
    # 读取参数并设置随机种子，保证可复现
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device()

    # 约定数据目录为项目根目录下的 data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_root = os.path.join(project_root, "data")
    # 默认模型保存到当前脚本目录
    model_path = (
        args.model_path if args.model_path else os.path.join(script_dir, "mnist_cnn_9.pt")
    )

    # 构建训练与测试 DataLoader
    train_loader = build_loader(
        data_root=data_root,
        batch_size=args.batch_size,
        train=True,
        subset_size=args.subset,
        seed=args.seed,
    )
    test_loader = build_loader(
        data_root=data_root,
        batch_size=1000,
        train=False,
        subset_size=0,
        seed=args.seed,
    )

    # 初始化模型并移动到设备
    model = Net().to(device)

    # 若模型已存在且未强制训练，则直接加载并评估
    if os.path.exists(model_path) and not args.force_train:
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        evaluate(model, device, test_loader)
        return

    # 采用 Adam 优化器进行训练
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, device, train_loader, optimizer, epoch)
        evaluate(model, device, test_loader)

    # 训练完成后保存模型权重
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    # 作为脚本运行时执行主逻辑
    main()
