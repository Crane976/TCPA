# models/lstm_predictor.py
import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        # LSTM层: 输入是9维的 ACTION_SET
        # 我们使用2层LSTM来增强其捕捉复杂关系的能力
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)

        # 全连接层: 将LSTM的隐藏状态映射到14维的衍生特征
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x 形状: [batch_size, seq_len=1, features=9]
        lstm_out, _ = self.lstm(x)

        # 取最后一个时间步的输出
        last_step_out = lstm_out[:, -1, :]

        # 经过全连接层得到最终预测
        prediction = self.fc(last_step_out)
        return prediction