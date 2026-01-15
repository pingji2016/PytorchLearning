import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
latent_size = 64
hidden_size = 256
image_size = 784 # 28x28
num_epochs = 50 # 演示用，可以改小
batch_size = 100
sample_dir = 'samples'

# 创建目录用于保存生成的图片
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# 图像处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]) # 归一化到 [-1, 1]
])

# MNIST 数据集
# 注意：这里假设数据已经下载或会自动下载到 ./data 目录
mnist = torchvision.datasets.MNIST(root='./data', 
                                   train=True, 
                                   transform=transform, 
                                   download=True)

# 数据加载器
data_loader = torch.utils.data.DataLoader(dataset=mnist, 
                                          batch_size=batch_size, 
                                          shuffle=True)

# 判别器 (Discriminator)
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid()
).to(device)

# 生成器 (Generator)
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh()
).to(device)

# 损失函数和优化器
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

# 开始训练
total_step = len(data_loader)
print(f"Start training on {device}...")

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(batch_size, -1).to(device)
        
        # 创建标签
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ================================================================== #
        #                      训练判别器 (Discriminator)                     #
        # ================================================================== #

        # 计算真实图片的损失
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        
        # 计算假图片的损失
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        # 反向传播和优化
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()

        # ================================================================== #
        #                        训练生成器 (Generator)                       #
        # ================================================================== #

        # 计算生成器的损失
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        
        # 生成器希望判别器认为这些图片是真的，所以这里用 real_labels
        g_loss = criterion(outputs, real_labels)
        
        # 反向传播和优化
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if (i+1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))
    
    # 每个 epoch 结束后保存一张生成的示例图
    if (epoch+1) == 1:
        save_image(denorm(images.reshape(images.size(0), 1, 28, 28)), os.path.join(sample_dir, 'real_images.png'))
    
    save_image(denorm(fake_images.reshape(fake_images.size(0), 1, 28, 28)), 
               os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

print("Training finished.")
