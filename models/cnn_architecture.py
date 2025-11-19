# In a new file: models/cnn_architecture.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Classifier(nn.Module):
    """一个基于一维卷积的分类器，用于流量特征分析"""

    def __init__(self, feature_dim):
        super(CNN_Classifier, self).__init__()

        # 1D-CNN期望的输入格式是 (N, C_in, L_in)
        # 我们的数据是 (N, feature_dim)，所以我们会把它看作是 C_in=1, L_in=feature_dim
        # N = Batch Size, C = Channels, L = Length

        # 第一个卷积层
        # (N, 1, 23) -> (N, 32, 21)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm1d(32)

        # 第二个卷积层
        # (N, 32, 21) -> (N, 64, 19)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm1d(64)

        # 池化层，将长度减半
        # (N, 64, 19) -> (N, 64, 9)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 展平层，为全连接层做准备
        # (N, 64, 9) -> (N, 64 * 9) = (N, 576)
        self.flatten = nn.Flatten()

        # 全连接层
        self.fc1 = nn.Linear(64 * 9, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x的初始维度是 (N, feature_dim)，例如 (256, 23)
        # 我们需要增加一个“通道”维度
        # (N, 23) -> (N, 1, 23)
        x = x.unsqueeze(1)

        # 通过卷积和池化层
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.pool1(out)

        # 展平并送入全连接层
        out = self.flatten(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        # 输出Logits
        logits = self.fc2(out)

        return logits

    def predict(self, x):
        # 用于最终预测的便捷方法
        logits = self.forward(x)
        return torch.sigmoid(logits)