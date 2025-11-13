# models/STEP1_train_style_transfer_cae.py (FINAL 3-TIER COMPATIBLE VERSION)
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os
import sys
import joblib

# ==========================================================
# --- 路径修正与模块导入 ---
# ==========================================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.style_transfer_cae import ConditionalAutoencoder
# ✅✅✅ 核心修改: 导入新的、为CAE量身定制的特征集 ✅✅✅
from config import ATTACKER_KNOWLEDGE_SET, set_seed

# ==========================================================
# --- 1. 配置区 ---
# ==========================================================
# --- 输入 ---
TRAIN_SET_PATH = os.path.join(project_root, 'data', 'splits', 'training_set.csv')
SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')

# --- 输出 ---
MODELS_DIR = os.path.join(project_root, 'models')
CAE_MODEL_PATH = os.path.join(MODELS_DIR, 'style_transfer_cae.pt')

# --- 模型参数 ---
# ✅✅✅ 核心修改: 特征维度现在由ATTACKER_KNOWLEDGE_SET决定 ✅✅✅
FEATURE_DIM = len(ATTACKER_KNOWLEDGE_SET)
LATENT_DIM = 5
NUM_CLASSES = 2  # Benign (0) 和 Bot (1)

# --- 训练参数 ---
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================================
# --- 2. 主训练函数 ---
# ==========================================================
def main():
    set_seed(2025)
    print("==========================================================")
    print("🚀 STEP 1 (Final): 训练上下文提取CAE引擎 (基于攻击者认知集)...")
    print(f"   >>> 攻击者认知边界 (输入维度): {FEATURE_DIM} 维 <<<")
    print(f"   (目标: 学习如何将 {FEATURE_DIM}维 特征压缩到 {LATENT_DIM}维 潜在空间)")
    print("==========================================================")
    print(f"使用设备: {device}")

    # --- 1. 加载数据和Scaler ---
    print("正在加载训练集和全局Scaler...")
    try:
        df_train_full = pd.read_csv(TRAIN_SET_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError as e:
        print(f"错误: 找不到核心文件 - {e}");
        return

    # --- 2. 准备特征(X)和条件标签(C) ---
    # ✅✅✅ 核心修改: 只提取ATTACKER_KNOWLEDGE_SET对应的列 ✅✅✅
    # 注意：我们先提取DEFENDER_SET的全部列，再用scaler转换，然后再选择子集
    # 这是一个更稳妥的做法，确保scaler应用在正确的维度上
    full_feature_names = scaler.feature_names_in_
    X_full_scaled = scaler.transform(df_train_full[full_feature_names].values)
    df_full_scaled = pd.DataFrame(X_full_scaled, columns=full_feature_names)

    X_scaled = df_full_scaled[ATTACKER_KNOWLEDGE_SET].values
    y_labels = df_train_full['label'].values

    print("数据准备完毕。")

    C_one_hot = np.zeros((len(y_labels), NUM_CLASSES))
    C_one_hot[np.arange(len(y_labels)), y_labels] = 1

    # --- 3. 划分训练/验证集 ---
    X_train, X_val, C_train, C_val = train_test_split(
        X_scaled, C_one_hot, test_size=VALIDATION_SPLIT, random_state=2025,
        stratify=C_one_hot.argmax(axis=1)
    )

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(C_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_tensor_x = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_tensor_c = torch.tensor(C_val, dtype=torch.float32).to(device)

    # --- 4. 初始化模型并开始训练 ---
    model = ConditionalAutoencoder(
        feature_dim=FEATURE_DIM,
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print("\n开始训练CAE模型...")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x_batch, c_batch in train_loader:
            x_batch, c_batch = x_batch.to(device), c_batch.to(device)
            recon, _ = model(x_batch, c_batch)
            loss = criterion(recon, x_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # --- 验证阶段 ---
        model.eval()
        with torch.no_grad():
            recon_val, _ = model(val_tensor_x, val_tensor_c)
            val_loss = criterion(recon_val, val_tensor_x).item()
            if (epoch + 1) % 10 == 0:
                print(
                    f"  -> Epoch {epoch + 1:3d}/{EPOCHS}, Train Loss: {total_loss / len(train_loader):.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CAE_MODEL_PATH)

    print("\n--- 训练完成 ---")
    print(f"表现最好的'上下文提取'CAE引擎已保存在: {CAE_MODEL_PATH}")
    print(f"(Final Best Validation Loss: {best_val_loss:.6f})")


if __name__ == "__main__":
    main()