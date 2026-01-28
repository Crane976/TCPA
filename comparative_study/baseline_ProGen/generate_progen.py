# comparative_study/baseline_ProGen/generate_progen.py
# Baseline Implementation: ProGen (IEEE TIFS 2024)
# Adapted for Flow Feature Space (FS-ProGen)
# Core Logic: WGAN-GP + Weighted MSE Constraint

import pandas as pd
import numpy as np
import os
import sys
import joblib
import torch
import torch.nn as nn
import torch.optim as optim

# --- 路径适配 ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path: sys.path.append(project_root)

import config
from config import DEFENDER_SET, set_seed
from models.wgan_gp import ProGenGenerator, ProGenDiscriminator, compute_gradient_penalty

# ==========================================================
# --- ProGen 参数配置 ---
# ==========================================================
BATCH_SIZE = 256
EPOCHS = 50  # WGAN 需要训练
LR = 0.0001
LAMBDA_GP = 10  # WGAN-GP 梯度惩罚系数
LAMBDA_WMSE = 10  # ProGen Constraint 2 系数 (论文中叫 a1, a2)

# 特征权重 (Weights w_i in Eq. 10)
# ProGen 强调保护 "Functional Features"。
# 对于诱饵生成 (Benign -> Bot)，我们要保护那些定义了"基本通信能力"的特征
# 比如：不要把包数量改成0，不要把窗口大小改得离谱
# 我们给 "Unmodifiable" 特征高权重，给 "Modifiable" 特征低权重
WEIGHT_HIGH = 10.0
WEIGHT_LOW = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_feature_weights(feature_names):
    """
    根据 ProGen 思想，为不同特征分配 WMSE 权重。
    """
    weights = []
    # 简单的启发式规则：
    # 时间、大小、数量 = 可改 (Low Weight)
    # 协议、标志位、比例 = 尽量少改 (High Weight)

    modifiable_keywords = ['Duration', 'IAT', 'Packets', 'Bytes', 'Length']

    for feat in feature_names:
        if any(k in feat for k in modifiable_keywords):
            weights.append(WEIGHT_LOW)
        else:
            weights.append(WEIGHT_HIGH)

    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_progen(df_train, scaler, num_features):
    print("Training ProGen (WGAN-GP + WMSE)...")

    # 1. 准备数据
    # Source: Benign (作为 G 的输入，模拟 Projection)
    # Target: Bot (作为 D 的真实样本，指导 G 向 Bot 分布映射)
    X_benign = df_train[df_train['label'] == 0][DEFENDER_SET].values
    X_bot = df_train[df_train['label'] == 1][DEFENDER_SET].values

    # 采样对齐 (让每个 Batch 都有数据)
    min_len = min(len(X_benign), len(X_bot))
    # 为了训练充分，我们随机采样

    # 转 Tensor
    X_benign_scaled = scaler.transform(X_benign)
    X_bot_scaled = scaler.transform(X_bot)

    # 2. 初始化模型
    generator = ProGenGenerator(num_features).to(device)
    discriminator = ProGenDiscriminator(num_features).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.9))

    feature_weights = get_feature_weights(DEFENDER_SET)

    # 3. 训练循环
    for epoch in range(EPOCHS):
        # 简单的 Batch 采样逻辑
        n_batches = int(min_len / BATCH_SIZE)

        for i in range(n_batches):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Random sample real bots
            idx_bot = np.random.randint(0, X_bot.shape[0], BATCH_SIZE)
            real_bot = torch.tensor(X_bot_scaled[idx_bot], dtype=torch.float32).to(device)

            # Random sample benign (Source)
            idx_benign = np.random.randint(0, X_benign.shape[0], BATCH_SIZE)
            real_benign = torch.tensor(X_benign_scaled[idx_benign], dtype=torch.float32).to(device)

            optimizer_D.zero_grad()

            # Generate fake bots (Projection: Benign -> Bot)
            fake_bot = generator(real_benign)

            # WGAN Loss: -Mean(D(Real)) + Mean(D(Fake))
            real_validity = discriminator(real_bot)
            fake_validity = discriminator(fake_bot)

            # Gradient Penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_bot, fake_bot, device)

            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + LAMBDA_GP * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            # Train G every n_critic steps (Standard WGAN usually 5, here 1 for simplicity or 5)
            if i % 5 == 0:
                optimizer_G.zero_grad()

                # Generate fake bots again
                fake_bot = generator(real_benign)

                # Adversarial Loss (Fool D)
                fake_validity = discriminator(fake_bot)
                g_adv_loss = -torch.mean(fake_validity)

                # Constraint 2: Perturbation Loss (Weighted MSE)
                # Ensure Generated Bot is not too far from Original Benign (Structural preservation)
                # Eq. 10 in paper: sum(w * (x_hat - x)^2)
                wmse_loss = torch.mean(feature_weights * (fake_bot - real_benign) ** 2)

                g_loss = g_adv_loss + LAMBDA_WMSE * wmse_loss

                g_loss.backward()
                optimizer_G.step()

        print(
            f"   [Epoch {epoch}/{EPOCHS}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] [WMSE: {wmse_loss.item():.4f}]")

    return generator


def main():
    set_seed(2025)
    print("=" * 60)
    print(f"🚀 [Baseline Reproduction] ProGen: Projection-based GAN")
    print(f"   Dataset: {config.CURRENT_DATASET}")
    print("=" * 60)

    # 1. 动态配置
    if config.CURRENT_DATASET == 'CIC-IDS2017':
        NUM_TO_GENERATE = 39300
    else:
        NUM_TO_GENERATE = 100000

    OUTPUT_PATH = os.path.join(project_root, 'data', 'generated', f'baseline_ProGen_{config.CURRENT_DATASET}.csv')

    # 2. 加载训练集
    train_path = os.path.join(config.SPLITS_DIR, 'training_set.csv')
    df_train = pd.read_csv(train_path)
    print(f"Loaded Training Data: {len(df_train)}")

    # 3. 加载 Scaler
    print(f"Loading Scaler from {config.SCALER_PATH}...")
    scaler = joblib.load(config.SCALER_PATH)

    # 4. 训练 ProGen Generator
    # ProGen 是需要训练的，不像 TSAF 是直接生成的
    # 我们现场训练一个 (耗时应该在几分钟内，因为是 MLP)
    generator = train_progen(df_train, scaler, len(DEFENDER_SET))
    generator.eval()

    # 5. 生成诱饵
    print(f"Generating {NUM_TO_GENERATE} decoys from Benign samples...")

    # 采样良性载体
    df_benign_source = df_train[df_train['label'] == 0].sample(n=NUM_TO_GENERATE, replace=True, random_state=2025)
    X_source = scaler.transform(df_benign_source[DEFENDER_SET])
    X_source_tensor = torch.tensor(X_source, dtype=torch.float32).to(device)

    with torch.no_grad():
        # 批量生成以防显存溢出
        BATCH = 1000
        X_gen_list = []
        for i in range(0, NUM_TO_GENERATE, BATCH):
            batch = X_source_tensor[i:i + BATCH]
            gen_batch = generator(batch)
            X_gen_list.append(gen_batch.cpu().numpy())

    X_gen_np = np.concatenate(X_gen_list, axis=0)

    # 6. 反归一化 & 保存
    print("Inverse scaling...")
    X_gen_original = scaler.inverse_transform(X_gen_np)

    df_gen = pd.DataFrame(X_gen_original, columns=DEFENDER_SET)
    df_gen['Label'] = 1

    # 7. 公平性后处理 (Rounding)
    # ProGen 也是软约束 (Loss function)，所以出来的包数量肯定是小数
    # 我们帮它取整，公平对比
    print("Applying Integer Rounding (Fairness)...")
    integer_cols = [
        'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
        'Fwd Header Length', 'Bwd Header Length',
        'Subflow Fwd Packets', 'Subflow Fwd Bytes',
        'Subflow Bwd Packets', 'Subflow Bwd Bytes',
        'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
        'act_data_pkt_fwd', 'min_seg_size_forward',
        'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
        'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio'
    ]
    for col in integer_cols:
        if col in df_gen.columns:
            df_gen[col] = df_gen[col].clip(lower=0).round().astype(int)

    df_gen.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ ProGen Baseline Generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()