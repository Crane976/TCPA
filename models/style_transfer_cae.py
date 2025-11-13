# models/style_transfer_cae.py (Final Refactored Version for Style Transfer)
"""
此文件定义了用于学习流量“风格”的条件自编码器（Conditional Autoencoder）模型结构。
版本已重构，以支持独立的编码和解码调用，从而实现风格迁移。
"""
import torch
import torch.nn as nn


class ConditionalAutoencoder(nn.Module):
    """
    条件自编码器模型。
    通过将类别标签(condition)与输入数据和潜在表示拼接，
    学习特定于类别的数据压缩和重构方式。
    """

    def __init__(self, feature_dim, latent_dim, num_classes):
        super().__init__()

        # --- 编码器网络 ---
        # 目标: 将 (特征 + 类别标签) 压缩为低维的潜在表示 Z
        self.encoder_net = nn.Sequential(
            nn.Linear(feature_dim + num_classes, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )

        # --- 解码器网络 ---
        # 目标: 将 (潜在表示 Z + 类别标签) 解码，重构为原始特征
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, feature_dim),
            nn.Sigmoid()  # ✅ 确保解码器的输出值被约束在[0, 1]范围内，这对于归一化的数据至关重要
        )

    def encode(self, x, c):
        """
        独立的编码器方法。
        Args:
            x: 输入特征张量
            c: 条件标签张量 (one-hot)
        Returns:
            encoded: 潜在空间表示 Z
        """
        # 将输入特征 x 和条件标签 c 在维度1上拼接
        x_with_condition = torch.cat([x, c], dim=1)
        encoded = self.encoder_net(x_with_condition)
        return encoded

    def decode(self, z, c):
        """
        独立的解码器方法。
        Args:
            z: 潜在空间表示 Z
            c: 条件标签张量 (one-hot)
        Returns:
            decoded: 重构的特征张量
        """
        # 将潜在表示 z 和条件标签 c 拼接
        z_with_condition = torch.cat([z, c], dim=1)
        decoded = self.decoder_net(z_with_condition)
        return decoded

    def forward(self, x, c):
        """
        完整的前向传播，用于标准的自编码器训练（STEP 1）。
        """
        # 编码过程
        encoded_z = self.encode(x, c)
        # 解码过程
        decoded_x = self.decode(encoded_z, c)

        return decoded_x, encoded_z