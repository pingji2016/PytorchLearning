import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import numpy as np
import os

# ==========================================
# 1. 基础设置与辅助函数
# ==========================================

# 设置随机种子，保证结果可复现 (Reproducibility)
# 在深度学习中，固定随机种子非常重要，这样每次运行代码得到的结果才是一样的。
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 保证 CUDA 的确定性行为，可能会牺牲一点性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# ==========================================
# 2. 模型定义 (Model Definition)
# ==========================================

class PositionalEncoding(nn.Module):
    """
    位置编码 (Positional Encoding)
    
    Transformer 模型完全基于注意力机制 (Attention Mechanism)，不像 RNN 那样按顺序处理数据，
    也不像 CNN 那样有局部感受野。因此，模型本身无法知道单词在句子中的先后顺序。
    
    我们需要显式地将位置信息注入到输入中。
    这里使用正弦和余弦函数生成的固定位置编码。
    
    公式:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 创建一个 max_len x d_model 的矩阵来存储位置编码
        pe = torch.zeros(max_len, d_model)
        
        # 生成位置索引: [0, 1, 2, ..., max_len-1] -> [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算分母项: 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 调整形状为 [max_len, 1, d_model]，以便和输入张量相加
        # 输入张量形状通常是 [seq_len, batch_size, d_model]
        pe = pe.unsqueeze(0).transpose(0, 1) 
        
        # register_buffer 告诉 PyTorch 'pe' 是模型状态的一部分，但在训练时不需要更新（不是参数）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: 输入张量，形状 [seq_len, batch_size, d_model]
        """
        # 将位置编码加到输入 embedding 上
        # self.pe[:x.size(0), :] 截取当前序列长度对应的位置编码
        x = x + self.pe[:x.size(0), :]
        return x

class SimpleTransformer(nn.Module):
    """
    一个简单的 Sequence-to-Sequence Transformer 模型
    """
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        self.d_model = d_model
        
        # 1. 词嵌入层 (Embedding Layer)
        # 将离散的单词索引 (Integer) 映射为稠密向量 (Vector)
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. 位置编码层
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer 核心模块
        # PyTorch 提供了封装好的 nn.Transformer，包含了 Encoder 和 Decoder
        # batch_first=False: 默认输入形状是 (seq_len, batch, feature)
        self.transformer = nn.Transformer(
            d_model=d_model,           # 嵌入维度
            nhead=nhead,               # 多头注意力的头数
            num_encoder_layers=num_layers, # Encoder 层数
            num_decoder_layers=num_layers, # Decoder 层数
            dropout=dropout            # Dropout 比率
        )
        
        # 4. 输出层
        # 将 Decoder 的输出映射回词表大小，用于预测下一个词的概率
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        """
        前向传播
        src: 源序列 (Encoder 输入), 形状 [src_len, batch_size]
        tgt: 目标序列 (Decoder 输入), 形状 [tgt_len, batch_size]
        """
        
        # --- 1. 生成掩码 (Mask) ---
        # 因果掩码 (Causal Mask) / 后续掩码 (Subsequent Mask)
        # 作用: 防止 Decoder 在 t 时刻看到 t+1 及以后的真实标签 (防止"偷看答案")
        # 生成一个上三角矩阵，上三角部分为 -inf (被遮挡)，对角线及下三角为 0 (可见)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(src.device)
        
        # --- 2. Embedding + Positional Encoding ---
        # 乘以 sqrt(d_model) 是 Transformer 论文中的技巧，用于缩放 Embedding 的值
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # --- 3. Transformer Forward ---
        # src: 传入 Encoder
        # tgt: 传入 Decoder
        # tgt_mask: 应用于 Decoder 的 Self-Attention，确保因果性
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        
        # --- 4. Output Linear ---
        # 得到每个位置的词表概率分布 logits
        return self.fc_out(output)

# ==========================================
# 3. 数据生成 (Data Generation)
# ==========================================

def generate_random_batch(batch_size, seq_len, vocab_size):
    """
    生成随机序列及其翻转作为训练数据
    任务: 序列翻转 (Sequence Reversal)
    输入: [1, 5, 2, 9] -> 目标: [9, 2, 5, 1]
    """
    # 随机生成 1 到 vocab_size-1 的整数 (0 通常留作 padding 或特殊标记，这里简单处理)
    data = torch.randint(1, vocab_size, (seq_len, batch_size))
    
    # 目标是输入的翻转
    # dims=[0] 表示沿序列长度维度翻转
    target = torch.flip(data, dims=[0])
    return data, target

# ==========================================
# 4. 训练流程 (Training Loop)
# ==========================================

def train():
    # --- 超参数设置 (Hyperparameters) ---
    VOCAB_SIZE = 50   # 词表大小 (数字 0-49)
    D_MODEL = 128     # 嵌入维度 (Embedding Dimension)
    EPOCHS = 300      # 训练轮数 (增加轮数，因为现在是真学习)
    BATCH_SIZE = 64   # 批大小
    # SEQ_LEN = 10    # (移除固定长度，改为动态长度)
    
    # 检测是否有 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化模型
    model = SimpleTransformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=8, num_layers=3).to(device)
    
    # 优化器: Adam
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    # 损失函数: CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    print("Start Training (Dynamic Data Generation)...")
    n_samples = 3000 # 每个 Epoch 生成的样本数
    n_batches = n_samples // BATCH_SIZE
    
    for epoch in range(EPOCHS):
        # --- 动态生成数据 (Dynamic Data Generation) ---
        # 1. 每个 Epoch 生成全新的随机数据，防止死记硬背
        # 2. 随机改变序列长度 (5 到 20)，让模型适应不同长度的输入
        current_seq_len = random.randint(5, 20)
        
        train_src, train_tgt = generate_random_batch(n_samples, current_seq_len, VOCAB_SIZE)
        train_src, train_tgt = train_src.to(device), train_tgt.to(device)
        
        total_loss = 0
        
        for i in range(n_batches):
            # 1. 准备 Batch 数据
            start = i * BATCH_SIZE
            end = start + BATCH_SIZE
            src = train_src[:, start:end] # [seq_len, batch_size]
            tgt = train_tgt[:, start:end] # [seq_len, batch_size]
            
            # 2. 构造 Decoder 输入 (Teacher Forcing)
            # Decoder 的输入应该是: <SOS> + Target Sequence[:-1]
            # 这里我们用 0 作为 <SOS> (Start of Sentence) 标记
            sos_token = torch.zeros((1, BATCH_SIZE), dtype=torch.long, device=device)
            decoder_input = torch.cat([sos_token, tgt[:-1]], dim=0)
            
            # 3. 前向传播
            optimizer.zero_grad() # 清空梯度
            output = model(src, decoder_input)
            
            # 4. 计算损失
            # output: [seq_len, batch, vocab_size] -> reshape -> [seq_len*batch, vocab_size]
            # tgt:    [seq_len, batch]             -> reshape -> [seq_len*batch]
            loss = criterion(output.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
            
            # 5. 反向传播与优化
            loss.backward()
            
            # 梯度裁剪 (Gradient Clipping): 防止梯度爆炸，稳定训练
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        # 打印日志
        if (epoch + 1) % 50 == 0:
            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    print("Training finished!")
    
    # --- 保存模型 (Save Model) ---
    # 保存模型的权重 (state_dict)，而不是整个模型对象
    save_path = 'transformer_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    return model, device

# ==========================================
# 5. 推理/评估流程 (Inference/Evaluation)
# ==========================================

def evaluate(model, device):
    model.eval() # 切换到评估模式 (关闭 Dropout)
    print("\n--- Evaluation (Inference) ---")
    
    # 准备测试输入
    seq_len = 10
    # 一个手动构造的序列: 1 到 10
    src = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).T  # 形状变为 [seq_len, 1]
    src = src.to(device)
    
    print(f"Input Sequence: {src.flatten().tolist()}")
    
    # --- 自回归解码 (Autoregressive Decoding) ---
    # Transformer 生成时，需要一个词一个词地生成，
    # 每次生成的词都会作为下一次的输入。
    
    # 1. 编码器 (Encoder) 前向计算一次，得到 memory
    # memory 包含了输入序列的上下文信息
    with torch.no_grad():
        src_emb = model.pos_encoder(model.embedding(src) * math.sqrt(model.d_model))
        memory = model.transformer.encoder(src_emb)
    
    # 2. 准备 Decoder 的初始输入: <SOS> (这里是 0)
    sos_token = 0
    generated = torch.tensor([[sos_token]], device=device) # [1, 1]
    
    print("Generating:", end=" ")
    
    # 3. 逐步生成
    for i in range(seq_len):
        with torch.no_grad():
            # 构造 Decoder 输入
            tgt_input = model.pos_encoder(model.embedding(generated) * math.sqrt(model.d_model))
            
            # 生成 Mask (虽然推理时是逐个生成，但为了匹配 API 格式还是加上)
            tgt_mask = model.transformer.generate_square_subsequent_mask(generated.size(0)).to(device)
            
            # Decoder 前向计算
            out = model.transformer.decoder(tgt_input, memory, tgt_mask=tgt_mask)
            
            # 通过线性层映射到词表
            out = model.fc_out(out)
            
            # 取最后一个时间步的输出 (最新的预测)
            last_token_logits = out[-1, :]
            
            # 贪婪解码 (Greedy Search): 直接取概率最大的词
            predicted_token = last_token_logits.argmax(dim=-1).item()
        
        print(predicted_token, end=" ")
        
        # 将预测的词拼接到 generated 序列中，作为下一步的输入
        generated = torch.cat([generated, torch.tensor([[predicted_token]], device=device)], dim=0)
    
    print("\nExpected (Reverse): 10 9 8 7 6 5 4 3 2 1")

if __name__ == "__main__":
    # 训练并保存模型
    model, device = train()
    
    # 评估模型
    evaluate(model, device)
