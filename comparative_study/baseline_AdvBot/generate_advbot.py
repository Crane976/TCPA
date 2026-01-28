# comparative_study/baseline_AdvBot/generate_advbot.py
# Baseline Implementation: Adv-Bot (Computers & Security 2023)
# Adapted for Decoy Generation (Benign -> Bot)
# Core Logic: Mean Difference Method (Eq. 6) + Projection Function

import pandas as pd
import numpy as np
import os
import sys
import joblib
import torch
import torch.nn as nn

# --- 路径适配 ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path: sys.path.append(project_root)

import config
from config import DEFENDER_SET, set_seed
# Adv-Bot 是黑盒迁移攻击，通常需要一个替身模型来判断是否攻击成功
# 这里我们使用 MLP Hunter 作为替身模型 (Surrogate Model)
from models.mlp_architecture import MLP_Classifier

# ==========================================================
# --- Adv-Bot 参数配置 ---
# ==========================================================
# Coefficient 'c' in Eq. 6 (Step size regulator)
C_COEFF = 0.05
# Max iterations (T)
MAX_ITER = 50

# 特征分组 (参考原文 Table 2 & 5)
# 绿色组 (Modifiable): 攻击者直接修改的特征
MODIFIABLE_FEATURES = [
    'Flow Duration',
    'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets'
]

# 黄色组 (Dependent): 必须由绿色组计算得出 (Proj Function)
DEPENDENT_FEATURES = [
    'Flow Bytes/s', 'Flow Packets/s',
    'Packet Length Mean',  # 近似计算
    'Down/Up Ratio'
]

# 红色组 (Unmodifiable): 保持不变 (Mask=0)
# 除了上述两组，剩下的都在这一组。

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_stats(df_train):
    """
    计算 Target (Bot) 的均值向量，以及 Source (Benign) 的均值向量。
    对应公式中的 mean_ratio 或 mean_diff。
    """
    print("Computing statistical means for Adv-Bot heuristic...")

    # 提取 Modifiable 特征的均值
    # 注意：这里需要在原始空间计算，因为硬约束(Proj)是在物理空间运作的
    # Adv-Bot 原文是在 Feature Space 操作然后 Proj，我们遵循此逻辑

    df_bot = df_train[df_train['label'] == 1][DEFENDER_SET]
    df_benign = df_train[df_train['label'] == 0][DEFENDER_SET]

    mu_bot = df_bot.mean()
    mu_benign = df_benign.mean()

    # 计算差异向量 (Mean Difference)
    # diff = |mu_bot - mu_benign|
    mean_diff = (mu_bot - mu_benign).abs()

    return mu_bot, mean_diff


def proj_function(df_batch):
    """
    [Adv-Bot Algorithm 1] Procedure Proj(x_adv)
    强制执行语义约束 (Semantic Constraints)
    """
    # 1. 语法约束 (Syntactic): 非负性、取整
    # Adv-Bot 强调这一点
    int_cols = ['Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets',
                'Total Length of Bwd Packets']
    for col in int_cols:
        if col in df_batch.columns:
            df_batch[col] = df_batch[col].clip(lower=0).round()

    if 'Flow Duration' in df_batch.columns:
        df_batch['Flow Duration'] = df_batch['Flow Duration'].clip(lower=1)  # 至少1微秒

    # 2. 语义约束 (Semantic): 重算依赖特征 (Yellow Group)
    # Rate = Total / Duration
    duration_sec = df_batch['Flow Duration'] / 1e6
    epsilon = 1e-9

    if 'Flow Bytes/s' in df_batch.columns:
        total_bytes = df_batch['Total Length of Fwd Packets'] + df_batch.get('Total Length of Bwd Packets', 0)
        df_batch['Flow Bytes/s'] = total_bytes / (duration_sec + epsilon)

    if 'Flow Packets/s' in df_batch.columns:
        total_pkts = df_batch['Total Fwd Packets'] + df_batch.get('Total Backward Packets', 0)
        df_batch['Flow Packets/s'] = total_pkts / (duration_sec + epsilon)

    if 'Packet Length Mean' in df_batch.columns:
        # 近似: Total Bytes / Total Pkts
        # 注意: 真实的 Packet Length Mean 涉及每个包的大小，这里只能做宏观近似
        # Adv-Bot 原文没有细说这个具体的公式，只说了 "recalculated"
        total_bytes = df_batch['Total Length of Fwd Packets'] + df_batch.get('Total Length of Bwd Packets', 0)
        total_pkts = df_batch['Total Fwd Packets'] + df_batch.get('Total Backward Packets', 0)
        df_batch['Packet Length Mean'] = total_bytes / (total_pkts + epsilon)

    return df_batch


def generate_adv_samples(df_source, mu_bot, mean_diff, surrogate_model, scaler):
    """
    Adv-Bot 核心迭代生成逻辑
    """
    # 转换为 DataFrame 以便进行列操作 (Adv-Bot 是基于列名的)
    # 但输入是 Tensor/Numpy? 不，我们直接在 DataFrame 上操作最方便，最后再 transform 进模型检测

    x_adv = df_source.copy().reset_index(drop=True)
    x_initial = x_adv.copy()  # x^0

    # 转换 mu_bot 和 mean_diff 为 numpy 以便广播计算
    # 只取 Modifiable features
    modifiable_cols = [c for c in MODIFIABLE_FEATURES if c in x_adv.columns]

    vec_mu_bot = mu_bot[modifiable_cols].values
    vec_mean_diff = mean_diff[modifiable_cols].values
    vec_x0 = x_initial[modifiable_cols].values

    print(f"Starting Iterative Attack (Max T={MAX_ITER})...")

    # 记录哪些样本已经成功欺骗了
    success_mask = np.zeros(len(x_adv), dtype=bool)

    for t in range(1, MAX_ITER + 1):
        # 1. 检查当前状态
        # 需要先 Proj，再 Scale，再 Predict
        x_adv = proj_function(x_adv)

        # Scale & Predict
        x_scaled = scaler.transform(x_adv[DEFENDER_SET])
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            # 假设 MLP 输出是 Logits
            preds = (torch.sigmoid(surrogate_model(x_tensor)) > 0.5).cpu().numpy().flatten()

        # 目标是 Label 1 (Bot)。如果 pred == 1，说明成功。
        current_success = (preds == 1)
        success_mask = success_mask | current_success

        success_rate = np.mean(success_mask)
        if t % 5 == 0 or t == 1:
            print(f"   Iter {t}: Success Rate = {success_rate * 100:.2f}%")

        if success_rate >= 0.99:
            break

        # 2. 更新未成功的样本 (Eq. 6)
        # x^{t} = x^{t-1} + sign(mu_bot - x^0) * (c * t) * mean_diff
        # 只更新 modifiable features

        # 获取当前未成功的样本索引
        not_done_indices = ~success_mask
        if not np.any(not_done_indices):
            break

        # 计算更新步长
        # sign(target - initial)
        direction = np.sign(vec_mu_bot - vec_x0[not_done_indices])
        step_mag = (C_COEFF * t) * vec_mean_diff

        perturbation = direction * step_mag

        # 更新 x_adv (在 Modifiable 列上)
        current_vals = x_adv.loc[not_done_indices, modifiable_cols].values
        new_vals = current_vals + perturbation

        x_adv.loc[not_done_indices, modifiable_cols] = new_vals

        # 下一轮循环开始前会调用 proj_function 修正依赖特征

    # 最终再 Proj 一次确保万无一失
    x_adv = proj_function(x_adv)
    return x_adv


def main():
    set_seed(2025)
    print("=" * 60)
    print(f"🚀 [Baseline Reproduction] Adv-Bot: Statistic-based Attack")
    print(f"   Dataset: {config.CURRENT_DATASET}")
    print("=" * 60)

    # 1. 动态配置
    if config.CURRENT_DATASET == 'CIC-IDS2017':
        NUM_TO_GENERATE = 39300
    else:
        NUM_TO_GENERATE = 100000

    OUTPUT_PATH = os.path.join(project_root, 'data', 'generated', f'baseline_AdvBot_{config.CURRENT_DATASET}.csv')

    # 2. 加载训练集 (计算统计量)
    train_path = os.path.join(config.SPLITS_DIR, 'training_set.csv')
    df_train = pd.read_csv(train_path)

    mu_bot, mean_diff = get_stats(df_train)

    # 3. 准备载体 (Benign)
    print(f"Loading Benign samples (Source)... Target: {NUM_TO_GENERATE}")
    # 注意：Adv-Bot 需要原始物理数值进行计算，所以我们取原始 csv，不要 scale
    df_benign = df_train[df_train['label'] == 0].sample(n=NUM_TO_GENERATE, replace=True, random_state=2025)

    # 4. 加载替身模型 & Scaler (仅用于引导迭代判定)
    print("Loading Surrogate Model (MLP) & Scaler...")
    scaler = joblib.load(config.SCALER_PATH)

    mlp_path = os.path.join(config.MODEL_SAVE_DIR, 'mlp_hunter.pt')
    surrogate_model = MLP_Classifier(feature_dim=len(DEFENDER_SET)).to(device)
    surrogate_model.load_state_dict(torch.load(mlp_path, map_location=device))
    surrogate_model.eval()

    # 5. 执行 Adv-Bot 生成
    df_adv = generate_adv_samples(df_benign, mu_bot, mean_diff, surrogate_model, scaler)

    # 6. 保存
    # 标记为 Bot
    df_adv['Label'] = 1

    # 只保留 DEFENDER_SET 列 + Label
    save_cols = DEFENDER_SET + ['Label']
    # 确保列存在 (防止某些列缺失)
    for c in DEFENDER_SET:
        if c not in df_adv.columns:
            df_adv[c] = 0

    df_adv[save_cols].to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Adv-Bot Baseline Generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()