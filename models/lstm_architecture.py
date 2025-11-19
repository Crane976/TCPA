# In a new file: models/lstm_architecture.py
import torch
import torch.nn as nn


class LSTM_Classifier(nn.Module):
    """一个基于LSTM的分类器，用于流量特征序列分析"""

    def __init__(self, feature_dim, hidden_dim=128, num_layers=2, dropout_rate=0.5):
        super(LSTM_Classifier, self).__init__()

        # LSTM层
        # LSTM期望的输入格式是 (N, L_in, H_in)
        # 我们的数据是 (N, feature_dim)，我们会把它看作是 L_in=1, H_in=feature_dim 的序列
        # N = Batch Size, L = Sequence Length, H = Hidden Size
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # 让Batch维度在第一位，方便处理
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True  # 使用双向LSTM以捕捉更丰富的上下文信息
        )

        # 分类器头部 (全连接层)
        # 双向LSTM的输出维度是 hidden_dim * 2
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x的初始维度是 (N, feature_dim)，例如 (256, 23)
        # 我们需要增加一个“序列长度”维度，这里序列长度为1
        # (N, 23) -> (N, 1, 23)
        x = x.unsqueeze(1)

        # LSTM的输出包括 output, (h_n, c_n)
        # output 的维度是 (N, L, D * H_out)，这里是 (N, 1, 2 * hidden_dim)
        # 我们只需要最后一个时间步的输出
        lstm_out, _ = self.lstm(x)

        # 获取最后一个时间步的输出
        # lstm_out[:, -1, :] 的维度是 (N, 2 * hidden_dim)
        last_step_output = lstm_out[:, -1, :]

        # 将其送入分类器头部
        logits = self.classifier_head(last_step_output)

        return logits

    def predict(self, x):
        # 用于最终预测的便捷方法
        logits = self.forward(x)
        return torch.sigmoid(logits)