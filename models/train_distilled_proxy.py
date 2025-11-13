# models/train_distilled_proxy.py (FINAL 3-TIER COMPATIBLE VERSION)
import pandas as pd
import numpy as np
import os
import sys
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ==========================================================
# --- Path Setup & Imports ---
# ==========================================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.append(project_root)

# ✅ 1. 导入新的特征集
from config import DEFENDER_SET, set_seed
from models.mlp_architecture import MLP_Classifier

# ==========================================================
# --- 1. Configuration ---
# ==========================================================
# --- Inputs (Teachers) ---
TRAIN_SET_PATH = os.path.join(project_root, 'data', 'splits', 'training_set.csv')
SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')
XGBOOST_HUNTER_PATH = os.path.join(project_root, 'models', 'xgboost_hunter.pkl')
MLP_HUNTER_PATH = os.path.join(project_root, 'models', 'mlp_hunter.pt')

# --- Output (Student) ---
DISTILLED_PROXY_HUNTER_PATH = os.path.join(project_root, 'models', 'proxy_hunter_distilled.pt')

# --- Training Parameters ---
# ✅ 2. 特征维度由DEFENDER_SET决定
FEATURE_DIM = len(DEFENDER_SET)
EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
XGBOOST_WEIGHT = 0.5  # 保持MLP和XGBoost同等重要


# ==========================================================
# --- 2. Main Distillation Function ---
# ==========================================================
def main():
    set_seed(2025)
    print("=" * 60);
    print("🚀 模型蒸馏: 训练集成代理猎手 (基于DEFENDER_SET)...");
    print("=" * 60)

    # --- 1. Load Assets ---
    try:
        df_train = pd.read_csv(TRAIN_SET_PATH)
        scaler = joblib.load(SCALER_PATH)
        xgboost_teacher = joblib.load(XGBOOST_HUNTER_PATH)
        mlp_teacher = MLP_Classifier(feature_dim=FEATURE_DIM).to(device)
        mlp_teacher.load_state_dict(torch.load(MLP_HUNTER_PATH, map_location=device))
        mlp_teacher.eval()
    except FileNotFoundError as e:
        print(f"Error: Could not find a core file - {e}");
        return

    # --- 2. Prepare Data and Soft Labels ---
    print("\n[步骤1] 正在从教师团生成软标签...")
    # ✅ 3. 使用DEFENDER_SET进行数据准备
    feature_names = scaler.feature_names_in_  # 确保与scaler训练时一致
    X_train_scaled = scaler.transform(df_train[feature_names].values)

    y_soft_xgb = xgboost_teacher.predict_proba(X_train_scaled)[:, 1]
    with torch.no_grad():
        y_soft_mlp = mlp_teacher.predict(
            torch.tensor(X_train_scaled, dtype=torch.float32).to(device)).cpu().numpy().flatten()

    y_train_soft_ensemble = (XGBOOST_WEIGHT * y_soft_xgb) + ((1 - XGBOOST_WEIGHT) * y_soft_mlp)

    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                  torch.tensor(y_train_soft_ensemble, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 3. Train Student Model ---
    student_model = MLP_Classifier(feature_dim=FEATURE_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=LEARNING_RATE)

    print("\n[步骤2] 开始蒸馏训练...")
    for epoch in range(EPOCHS):
        student_model.train()
        total_loss = 0
        for x_batch, y_soft_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            x_batch, y_soft_batch = x_batch.to(device), y_soft_batch.to(device)
            pred_logits = student_model(x_batch)
            loss = criterion(torch.sigmoid(pred_logits), y_soft_batch)
            optimizer.zero_grad();
            loss.backward();
            optimizer.step()
            total_loss += loss.item()
        print(f"  -> Epoch {epoch + 1} 完成 | 平均蒸馏损失: {total_loss / len(train_loader):.6f}")

    # --- 4. Save Distilled Model ---
    torch.save(student_model.state_dict(), DISTILLED_PROXY_HUNTER_PATH)
    print(f"\n✅ 蒸馏代理猎手已保存到: {DISTILLED_PROXY_HUNTER_PATH}")


if __name__ == "__main__":
    main()