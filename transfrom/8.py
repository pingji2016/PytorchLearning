import torch
import torch.nn as nn
import math
import os

# ==========================================
# 1. 模型定义 (需与训练时完全一致)
# ==========================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,           
            nhead=nhead,               
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers, 
            dropout=dropout            
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(src.device)
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return self.fc_out(output)

# ==========================================
# 2. 推理功能
# ==========================================

def load_model(model_path, device):
    """加载模型"""
    print(f"Loading model from {model_path}...")
    
    # 必须使用与训练时相同的超参数
    VOCAB_SIZE = 50
    D_MODEL = 128
    # 注意：7.py 中训练时使用了 nhead=8, num_layers=3
    model = SimpleTransformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=8, num_layers=3)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model

def predict(model, input_sequence, device):
    """
    对输入序列进行预测 (翻转任务)
    input_sequence: list of integers, e.g. [1, 2, 3]
    """
    model.eval()
    
    # 转换为 tensor 并增加 batch 维度 -> [seq_len, 1]
    src = torch.tensor([input_sequence]).T.to(device)
    seq_len = len(input_sequence)
    
    # 1. Encoder 前向计算
    with torch.no_grad():
        src_emb = model.pos_encoder(model.embedding(src) * math.sqrt(model.d_model))
        memory = model.transformer.encoder(src_emb)
    
    # 2. Decoder 自回归生成
    sos_token = 0
    generated = torch.tensor([[sos_token]], device=device) # [1, 1]
    
    predicted_seq = []
    
    print(f"Input: {input_sequence}")
    print("Generating:", end=" ")
    
    # 生成长度通常与输入长度一致
    for i in range(seq_len):
        with torch.no_grad():
            tgt_input = model.pos_encoder(model.embedding(generated) * math.sqrt(model.d_model))
            tgt_mask = model.transformer.generate_square_subsequent_mask(generated.size(0)).to(device)
            out = model.transformer.decoder(tgt_input, memory, tgt_mask=tgt_mask)
            out = model.fc_out(out)
            last_token_logits = out[-1, :]
            predicted_token = last_token_logits.argmax(dim=-1).item()
        
        predicted_seq.append(predicted_token)
        print(predicted_token, end=" ")
        
        # 下一步输入
        generated = torch.cat([generated, torch.tensor([[predicted_token]], device=device)], dim=0)
    
    print()
    return predicted_seq

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 定位模型文件
    # 假设 8.py 在 transfrom 目录下，而 model 在根目录下
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # 上一级目录
    model_path = os.path.join(project_root, 'transformer_model.pth')
    
    if not os.path.exists(model_path):
        # 尝试在当前目录下找 (如果用户把模型移过来了)
        model_path = 'transformer_model.pth'
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return

    try:
        model = load_model(model_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\n--- Inference Demo ---")
    
    # 测试案例 1
    seq1 = [1, 2, 3, 4, 5]
    predict(model, seq1, device)
    
    # 测试案例 2
    seq2 = [10, 20, 30, 40]
    predict(model, seq2, device)
    
    # 交互式输入
    while True:
        user_input = input("\nEnter a sequence of numbers (space separated) or 'q' to quit: ")
        if user_input.lower() == 'q':
            break
        
        try:
            seq = [int(x) for x in user_input.split()]
            if not seq:
                continue
            predict(model, seq, device)
        except ValueError:
            print("Invalid input. Please enter numbers separated by space.")

if __name__ == "__main__":
    main()
