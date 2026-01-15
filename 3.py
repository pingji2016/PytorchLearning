import os
# 设置环境变量以避免 OpenMP 重复初始化错误
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

# ================================================================== #
#                         1. 配置与超参数                             #
# ================================================================== #

# 设备配置：如果有 GPU 则使用 GPU，否则使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")


# 超参数设置
latent_size = 64      # 潜在向量（随机噪声）的维度
hidden_size = 256     # 隐藏层神经元数量
image_size = 784      # 图像展平后的大小 (28*28 = 784)
num_epochs = 200      # 训练轮数（增加轮数以获得更好效果）
batch_size = 100      # 批次大小
sample_dir = 'samples' # 生成样本的保存目录

# 创建目录用于保存生成的图片
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# ================================================================== #
#                         2. 数据准备                                 #
# ================================================================== #

# 图像预处理：
# 1. ToTensor: 将 PIL Image 或 numpy.ndarray 转换为 tensor，并归一化到 [0, 1]
# 2. Normalize: 将 [0, 1] 的数据归一化到 [-1, 1]。公式：image = (image - mean) / std
#    这里 mean=[0.5], std=[0.5]，所以 (0-0.5)/0.5 = -1, (1-0.5)/0.5 = 1
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# MNIST 数据集
# root: 数据存放路径
# train: True 表示加载训练集
# transform: 应用上面的预处理
# download: 如果数据不存在则自动下载
mnist = torchvision.datasets.MNIST(root='./data', 
                                   train=True, 
                                   transform=transform, 
                                   download=True)

# 数据加载器：用于批量加载数据，支持打乱 (shuffle)
data_loader = torch.utils.data.DataLoader(dataset=mnist, 
                                          batch_size=batch_size, 
                                          shuffle=True)

# ================================================================== #
#                         3. 模型定义                                 #
# ================================================================== #

# 判别器 (Discriminator)
# 作用：接收一张图片，判断它是真实的（来自数据集）还是假的（由生成器生成）。
# 改进：增加了 LeakyReLU 和 Dropout 以提高泛化能力，防止过拟合。
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),   # 输入层 -> 隐藏层
    nn.LeakyReLU(0.2),                    # 激活函数：LeakyReLU 允许小的负值通过，防止梯度消失
    nn.Linear(hidden_size, hidden_size),  # 隐藏层 -> 隐藏层
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),            # 隐藏层 -> 输出层 (输出一个标量)
    nn.Sigmoid()                          # Sigmoid 将输出压缩到 [0, 1]，表示“是真实图片”的概率
).to(device)

# 生成器 (Generator)
# 作用：接收一个随机噪声向量，生成一张假图片。
# 改进：使用了 BatchNorm (批归一化) 来稳定训练，防止梯度爆炸或消失。
#       使用了 ReLU 作为中间层的激活函数，Tanh 作为输出层激活函数。
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),  # 输入噪声 -> 隐藏层
    nn.BatchNorm1d(hidden_size),          # 批归一化
    nn.ReLU(),                            # ReLU 激活
    nn.Linear(hidden_size, hidden_size),  # 隐藏层 -> 隐藏层
    nn.BatchNorm1d(hidden_size),          # 批归一化
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),   # 隐藏层 -> 输出层 (生成 784 维向量)
    nn.Tanh()                             # Tanh 将输出压缩到 [-1, 1]，与预处理后的真实图片范围一致
).to(device)

# ================================================================== #
#                         4. 训练过程                                 #
# ================================================================== #

# 损失函数：二元交叉熵损失 (Binary Cross Entropy Loss)
# 用于衡量二分类任务的误差
criterion = nn.BCELoss()

# 优化器：Adam 算法
# lr (learning rate): 学习率
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

# 辅助函数：将 [-1, 1] 的数据还原回 [0, 1]，以便保存为图片
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# 辅助函数：清空梯度
def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

total_step = len(data_loader)
print(f"Start training on {device}...")

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        # 将图片展平为 (batch_size, 784)
        images = images.reshape(batch_size, -1).to(device)
        
        # 创建标签
        # real_labels: 真实图片的标签为 1
        # fake_labels: 假图片的标签为 0
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ------------------------------------------------------------------ #
        #                      训练判别器 (Discriminator)                     #
        # ------------------------------------------------------------------ #
        # 目标：最大化 log(D(x)) + log(1 - D(G(z)))
        # 即：让判别器正确识别真实图片 (输出 1) 和 假图片 (输出 0)

        # 1. 计算真实图片的损失
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        
        # 2. 计算假图片的损失
        z = torch.randn(batch_size, latent_size).to(device) # 生成随机噪声
        fake_images = G(z)                                  # 生成假图片
        outputs = D(fake_images)                            # 判别器判断假图片
        d_loss_fake = criterion(outputs, fake_labels)       # 假图片的标签应该是 0
        fake_score = outputs
        
        # 3. 反向传播和优化
        d_loss = d_loss_real + d_loss_fake # 总损失
        reset_grad()
        d_loss.backward()
        d_optimizer.step()

        # ------------------------------------------------------------------ #
        #                        训练生成器 (Generator)                       #
        # ------------------------------------------------------------------ #
        # 目标：最大化 log(D(G(z)))
        # 即：让判别器认为生成的图片是真的 (输出 1)

        # 1. 生成新的假图片
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        
        # 2. 计算损失
        # 注意：这里标签使用的是 real_labels (1)，因为生成器希望骗过判别器
        g_loss = criterion(outputs, real_labels)
        
        # 3. 反向传播和优化
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        
        # 每 200 步打印一次日志
        if (i+1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))
    
    # 每个 epoch 结束后保存生成的图片
    # 保存真实图片作为对比 (只保存一次)
    if (epoch+1) == 1:
        save_image(denorm(images.reshape(images.size(0), 1, 28, 28)), os.path.join(sample_dir, 'real_images.png'))
    
    # 保存生成的假图片
    save_image(denorm(fake_images.reshape(fake_images.size(0), 1, 28, 28)), 
               os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))
    print(f"Saved generated images for epoch {epoch+1}")

# 保存模型权重
torch.save(G.state_dict(), 'generator.pth')
torch.save(D.state_dict(), 'discriminator.pth')
print("Model saved to generator.pth and discriminator.pth")

print("Training finished.")
