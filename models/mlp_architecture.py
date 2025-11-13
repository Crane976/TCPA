# In models/mlp_architecture.py (Final ResNet-style Version)
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """一个包含残差连接的基本块"""

    def __init__(self, size, dropout_rate=0.5):
        super(ResidualBlock, self).__init__()
        self.norm1 = nn.BatchNorm1d(size)
        self.layer1 = nn.Linear(size, size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 残差路径
        residual = x

        # 主路径
        out = self.norm1(x)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.dropout(out)

        # 添加残差连接
        out += residual

        return out


class MLP_Classifier(nn.Module):
    """一个基于残差块的、更深、更强大的MLP分类器"""

    def __init__(self, feature_dim):
        super(MLP_Classifier, self).__init__()

        # 输入层
        self.input_layer = nn.Linear(feature_dim, 256)  # ✅ 更宽的输入层

        # 一系列的残差块
        self.residual_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),  # ✅ 堆叠多个残差块以加深网络
            ResidualBlock(256),
        )

        # 输出层
        self.output_norm = nn.BatchNorm1d(256)
        self.output_layer = nn.Linear(256, 1)

    def get_features(self, x):  # ✅ 新增方法
        """返回输出层之前的特征表示"""
        out = self.input_layer(x)
        out = self.residual_blocks(out)
        out = self.output_norm(out)
        features = F.relu(out)
        return features

    def forward(self, x):
        # 完整的网络流程
        out = self.input_layer(x)
        out = self.residual_blocks(out)
        out = self.output_norm(out)
        out = F.relu(out)

        # 返回logits（用于BCEWithLogitsLoss）
        logits = self.output_layer(out)

        return logits

    def predict(self, x):
        # 用于最终预测的便捷方法
        logits = self.forward(x)
        return torch.sigmoid(logits)