# In file: models/transformer_architecture.py (UPGRADED & FIXED)
import torch
import torch.nn as nn
import math


# PositionalEncoding 类保持不变，因为它现在会接收一个偶数的d_model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
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


class Transformer_Classifier(nn.Module):
    """
    一个基于Transformer Encoder的分类器 (升级版: 带有输入投影层)。
    """

    def __init__(self, feature_dim, d_model=32, nhead=4, num_encoder_layers=2, dim_feedforward=128, dropout=0.3):
        super(Transformer_Classifier, self).__init__()

        # ✅ 核心修复 1: 定义一个内部模型维度 d_model，它是一个偶数
        self.d_model = d_model

        # ✅ 核心修复 2: 添加一个输入投影层
        # 这个线性层将原始的23维特征，投影到Transformer友好的32维空间
        self.input_projection = nn.Linear(feature_dim, self.d_model)

        # 位置编码现在使用d_model，不再有奇偶问题
        self.pos_encoder = PositionalEncoding(self.d_model)

        # Transformer编码器层，现在nhead可以被d_model整除
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # 堆叠多个编码器层
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 分类器头部，输入维度现在是d_model
        self.classifier_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        # 初始化权重，对Transformer有益
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.classifier_head[0].weight.data.uniform_(-initrange, initrange)
        self.classifier_head[3].weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # x的初始维度是 (N, feature_dim), e.g., (256, 23)

        # ✅ 核心修复 3: 首先通过投影层
        # (N, 23) -> (N, 32)
        x = self.input_projection(x)

        # Transformer期望的输入格式是 (N, L, E) for batch_first=True
        # (N, 32) -> (N, 1, 32)
        x = x.unsqueeze(1)

        # 此时序列长度为1，位置编码作用不大，但我们保留这个结构以备将来扩展
        # x = self.pos_encoder(x)

        # 通过Transformer编码器
        # (N, 1, 32) -> (N, 1, 32)
        transformer_out = self.transformer_encoder(x)

        # 取序列的平均输出用于分类 (对于长度为1的序列，等同于取第一个)
        # (N, 1, 32) -> (N, 32)
        pooled_output = transformer_out.mean(dim=1)

        # 送入分类器头部
        logits = self.classifier_head(pooled_output)

        return logits

    def predict(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits)