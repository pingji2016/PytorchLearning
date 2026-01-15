import os
# 设置环境变量以避免 OpenMP 重复初始化错误
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
# from torch.cuda.amp import autocast, GradScaler # 已废弃，改用 torch.amp

# 启用 cuDNN 自动调优，加速训练
torch.backends.cudnn.benchmark = True
# 启用 TF32 (TensorFloat-32) 加速矩阵乘法 (Ampere 架构特性)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main():
    # ================================================================== #
    #                         1. 配置与超参数                             #
    # ================================================================== #

    # 设备配置：如果有 GPU 则使用 GPU，否则使用 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        # 打印当前显存使用情况
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")


    # 超参数设置
    # 优化：针对 A10 (24GB) 极致优化
    latent_size = 64      # 潜在向量（随机噪声）的维度
    hidden_size = 2048    # 隐藏层神经元数量
    image_size = 784      # 图像展平后的大小 (28*28 = 784)
    num_epochs = 200      # 训练轮数
    batch_size = 4096     # 批次大小
    sample_dir = 'samples' # 生成样本的保存目录
    
    # 混合精度 Scaler (更新为新 API)
    scaler = torch.amp.GradScaler('cuda')

    # 创建目录用于保存生成的图片
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # ================================================================== #
    #                         2. 数据准备                                 #
    # ================================================================== #

    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # MNIST 数据集
    mnist = torchvision.datasets.MNIST(root='./data', 
                                       train=True, 
                                       transform=transform, 
                                       download=True)

    # 数据加载器
    data_loader = torch.utils.data.DataLoader(dataset=mnist, 
                                              batch_size=batch_size, 
                                              shuffle=True,
                                              num_workers=8,
                                              pin_memory=True)

    # ================================================================== #
    #                         3. 模型定义                                 #
    # ================================================================== #

    # 判别器 (Discriminator)
    # 修改：移除 Sigmoid，配合 BCEWithLogitsLoss 使用，以支持混合精度训练
    D = nn.Sequential(
        nn.Linear(image_size, hidden_size),   
        nn.LeakyReLU(0.2),                    
        nn.Linear(hidden_size, hidden_size),  
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_size, 1)
        # nn.Sigmoid()  <-- 已移除，输出 logits
    ).to(device)

    # 生成器 (Generator)
    G = nn.Sequential(
        nn.Linear(latent_size, hidden_size),  
        nn.BatchNorm1d(hidden_size),          
        nn.ReLU(),                            
        nn.Linear(hidden_size, hidden_size),  
        nn.BatchNorm1d(hidden_size),          
        nn.ReLU(),
        nn.Linear(hidden_size, image_size),   
        nn.Tanh()                             
    ).to(device)

    # ================================================================== #
    #                         4. 训练过程                                 #
    # ================================================================== #

    # 使用 BCEWithLogitsLoss 代替 BCELoss，数值更稳定且支持 AMP
    criterion = nn.BCEWithLogitsLoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

    def denorm(x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def reset_grad():
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()

    total_step = len(data_loader)
    print(f"Start training on {device} with AMP...")

    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(data_loader):
            curr_batch_size = images.size(0)
            images = images.reshape(curr_batch_size, -1).to(device)
            
            # 创建标签
            real_labels = torch.ones(curr_batch_size, 1).to(device)
            fake_labels = torch.zeros(curr_batch_size, 1).to(device)

            # ------------------------------------------------------------------ #
            #                      训练判别器 (Discriminator)                     #
            # ------------------------------------------------------------------ #

            # 1. 计算真实图片的损失 (使用 autocast)
            with torch.amp.autocast('cuda'):
                outputs = D(images)
                d_loss_real = criterion(outputs, real_labels)
                real_score = outputs # 这里是 logits
                
                # 2. 计算假图片的损失
                z = torch.randn(curr_batch_size, latent_size).to(device)
                fake_images = G(z)
                outputs = D(fake_images)
                d_loss_fake = criterion(outputs, fake_labels)
                fake_score = outputs # 这里是 logits
                
                d_loss = d_loss_real + d_loss_fake
            
            # 3. 反向传播和优化 (使用 scaler)
            reset_grad()
            scaler.scale(d_loss).backward()
            scaler.step(d_optimizer)
            scaler.update()

            # ------------------------------------------------------------------ #
            #                        训练生成器 (Generator)                       #
            # ------------------------------------------------------------------ #

            with torch.amp.autocast('cuda'):
                # 1. 生成新的假图片
                z = torch.randn(curr_batch_size, latent_size).to(device)
                fake_images = G(z)
                outputs = D(fake_images)
                
                # 2. 计算损失
                g_loss = criterion(outputs, real_labels)
            
            # 3. 反向传播和优化
            reset_grad()
            scaler.scale(g_loss).backward()
            scaler.step(g_optimizer)
            scaler.update()
            
            # 日志打印
            if (i+1) % 5 == 0 or (i+1) == total_step:
                # 将 logits 转换为概率以便打印
                real_score_prob = torch.sigmoid(real_score).mean().item()
                fake_score_prob = torch.sigmoid(fake_score).mean().item()
                
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                      .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                              real_score_prob, fake_score_prob))
        
        # 每个 epoch 结束后保存生成的图片
        if (epoch+1) == 1:
            save_image(denorm(images.reshape(curr_batch_size, 1, 28, 28)), os.path.join(sample_dir, 'real_images.png'))
        
        # 保存一张示例图 (取第一张)
        save_image(denorm(fake_images.reshape(curr_batch_size, 1, 28, 28)), 
                   os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))
        
        # 偶尔打印一下显存
        if (epoch+1) % 50 == 0:
             print(f"Epoch {epoch+1} GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated")

    # 保存模型权重
    torch.save(G.state_dict(), 'generator.pth')
    torch.save(D.state_dict(), 'discriminator.pth')
    print("Model saved to generator.pth and discriminator.pth")

    print("Training finished.")

if __name__ == '__main__':
    main()
