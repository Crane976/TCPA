# models/lstm_finetuner.py
import torch
import torch.nn as nn

class LSTMFinetuner(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        # LSTM层: 输入是13维的 KNOWLEDGE_SET
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # 激活函数
        self.relu = nn.ReLU()
        # 全连接层: 输出是9维的 ACTION_SET
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x 形状: [batch, seq_len=1, features=13]
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_step_out = lstm_out[:, -1, :]
        out = self.relu(last_step_out)
        out = self.fc(out)
        return out