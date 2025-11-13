# models/bot_pattern_lstm.py (Full Feature Generation Version)
import torch
import torch.nn as nn

class BotPatternLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, condition_dim):
        super(BotPatternLSTM, self).__init__()
        # 核心修改: 将条件向量与每个时间步的输入拼接
        self.lstm = nn.LSTM(input_dim + condition_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_seq, c):
        # x_seq: (batch, seq_len, input_dim)
        # c: (batch, condition_dim)

        # 将条件向量c扩展，以便与序列中的每个时间步拼接
        # c_expanded: (batch, seq_len, condition_dim)
        c_expanded = c.unsqueeze(1).expand(-1, x_seq.size(1), -1)

        # 拼接输入
        # combined_input: (batch, seq_len, input_dim + condition_dim)
        combined_input = torch.cat([x_seq, c_expanded], dim=2)

        # LSTM处理
        lstm_out, _ = self.lstm(combined_input)

        # 我们只关心序列的最后一个时间步的输出，因为它包含了整个序列的信息
        # last_output: (batch, hidden_dim)
        last_output = lstm_out[:, -1, :]

        # 全连接层输出最终预测
        # prediction: (batch, output_dim)
        prediction = self.fc(last_output)

        return prediction