# models/train_mlp_hunter.py (FINAL 3-TIER COMPATIBLE VERSION)
import pandas as pd
import numpy as np
import os
import sys
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==========================================================
# --- Path Setup & Imports ---
# ==========================================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# ✅ 1. 导入新的特征集
from config import DEFENDER_SET, set_seed
from models.mlp_architecture import MLP_Classifier

# ==========================================================
# --- 1. Configuration ---
# ==========================================================
# Paths...
TRAIN_SET_PATH = os.path.join(project_root, 'data', 'splits', 'training_set.csv')
TEST_SET_PATH = os.path.join(project_root, 'data', 'splits', 'holdout_test_set.csv')
SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')
MLP_HUNTER_MODEL_PATH = os.path.join(project_root, 'models', 'mlp_hunter.pt')

# Training Parameters...
# ✅ 2. 特征维度由DEFENDER_SET决定
FEATURE_DIM = len(DEFENDER_SET)
EPOCHS = 80
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 2025

BEST_PARAMS = {'pos_weight': 5, 'learning_rate': 0.001}


# ==========================================================
# --- 2. Main Training Function ---
# ==========================================================
def main():
    set_seed(RANDOM_SEED)
    print("=" * 60);
    print("🚀 开始训练 ResNet-MLP Hunter (基于DEFENDER_SET)...");
    print("=" * 60)

    # --- Load Assets & Prepare Data ---
    df_train_full = pd.read_csv(TRAIN_SET_PATH)
    df_test = pd.read_csv(TEST_SET_PATH)
    scaler = joblib.load(SCALER_PATH)

    # ✅ 3. 使用DEFENDER_SET进行数据准备
    # 使用scaler中存储的列名，确保与scaler训练时一致
    feature_names = scaler.feature_names_in_
    X_test_scaled = scaler.transform(df_test[feature_names].values)
    y_test = df_test['label'].values

    X_train_full_scaled = scaler.transform(df_train_full[feature_names].values)
    y_train_full = df_train_full['label'].values

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full_scaled, y_train_full, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, stratify=y_train_full
    )
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_tensor_x = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_tensor_y = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    # --- Initialize & Train ---
    pos_weight_tensor = torch.tensor([BEST_PARAMS['pos_weight']], dtype=torch.float32).to(device)
    model = MLP_Classifier(feature_dim=FEATURE_DIM).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BEST_PARAMS['learning_rate'])

    print("\n[步骤] 正在训练模型...")
    best_val_loss = float('inf')
    for epoch in tqdm(range(EPOCHS), desc="Training"):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss = criterion(model(x_batch), y_batch)
            optimizer.zero_grad();
            loss.backward();
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_tensor_x), val_tensor_y).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MLP_HUNTER_MODEL_PATH)

    print(f"\n✅ 训练完成，最佳验证损失: {best_val_loss:.6f}")

    # --- Final Evaluation ---
    print("\n--- 最终'ResNet-MLP Hunter'在【留出测试集】上的真实性能报告 ---")
    final_model = MLP_Classifier(feature_dim=FEATURE_DIM).to(device)
    final_model.load_state_dict(torch.load(MLP_HUNTER_MODEL_PATH, map_location=device))
    final_model.eval()
    with torch.no_grad():
        y_pred = (final_model.predict(
            torch.tensor(X_test_scaled, dtype=torch.float32).to(device)) > 0.5).int().cpu().numpy()
    print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Bot (1)'], digits=4))


if __name__ == "__main__":
    main()