import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

# ================================================================== #
#                         1. 配置与模型定义                           #
# ================================================================== #

# 确保输出目录存在
# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
sample_dir = os.path.join(script_dir, 'generated_samples')

if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 超参数 (必须与训练时一致)
latent_size = 64
hidden_size = 256
image_size = 784

# 生成器模型定义 (适配 generator.pth 文件结构)
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
#                         2. 加载模型                                 #
# ================================================================== #

# 自动寻找模型文件 (优先找脚本同级目录，其次找上级目录)
model_name = 'generator.pth'
possible_paths = [
    os.path.join(script_dir, model_name),       # 同级目录
    os.path.join(script_dir, '..', model_name), # 上级目录
    model_name                                  # 当前工作目录
]

model_path = None
for p in possible_paths:
    if os.path.exists(p):
        model_path = p
        break

if model_path:
    # 加载权重
    try:
        # map_location 确保即使在没有 GPU 的机器上也能加载 GPU 训练的模型
        G.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Successfully loaded model from {model_path}")
    except RuntimeError as e:
        print(f"\nError loading model: {e}")
        print("\n[!] 架构不匹配 detected!")
        print(f"    saved model path: {model_path}")
        print("    原因: 代码中的模型结构已更新（变大了），但 generator.pth 还是旧的（小的）。")
        print("    解决: 请先运行 'python 3.py' 重新训练新模型。")
        exit()
else:
    print(f"Error: Model file '{model_name}' not found in {[os.path.abspath(p) for p in possible_paths]}.")
    exit()

# 设置为评估模式 (对于包含 BatchNorm/Dropout 的模型很重要)
G.eval()

# ================================================================== #
#                         3. 生成图片                                 #
# ================================================================== #

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# 生成 64 张图片
num_samples = 64
z = torch.randn(num_samples, latent_size).to(device)

# 不需要计算梯度
with torch.no_grad():
    fake_images = G(z)
    
    # 调整形状并去归一化
    fake_images = fake_images.reshape(num_samples, 1, 28, 28)
    fake_images = denorm(fake_images)
    
    # 保存结果
    output_path = os.path.join(sample_dir, 'result.png')
    save_image(fake_images, output_path)
    print(f"Generated {num_samples} images saved to {output_path}")
