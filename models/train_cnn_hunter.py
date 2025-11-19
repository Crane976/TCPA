# In a new file: models/train_cnn_hunter.py
import pandas as pd
import numpy as np
import os
import sys
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==========================================================
# --- Path Setup & Imports ---
# ==========================================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import DEFENDER_SET, set_seed
# ✅ 1. 导入新的CNN模型和我们之前定义的FocalLoss
from models.cnn_architecture import CNN_Classifier
from models.mlp_architecture import FocalLoss

# ==========================================================
# --- 1. Configuration ---
# ==========================================================
TRAIN_SET_PATH = os.path.join(project_root, 'data', 'splits', 'training_set.csv')
TEST_SET_PATH = os.path.join(project_root, 'data', 'splits', 'holdout_test_set.csv')
SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')
# ✅ 2. 为新模型指定新的保存路径
CNN_HUNTER_MODEL_PATH = os.path.join(project_root, 'models', 'cnn_hunter.pt')

FEATURE_DIM = len(DEFENDER_SET)
EPOCHS = 100
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 2025
BEST_PARAMS = {'learning_rate': 0.0005}

# ==========================================================
# --- 2. Main Training Function ---
# ==========================================================
def main():
    set_seed(RANDOM_SEED)
    print("=" * 60)
    print("🚀 开始训练 1D-CNN Hunter (Focal Loss + 阈值优化)...")
    print("=" * 60)
    print(f"使用设备: {device}")

    # --- 数据加载和准备 (与MLP版本完全相同) ---
    print("\n[步骤] 正在加载数据和Scaler...")
    df_train_full = pd.read_csv(TRAIN_SET_PATH)
    df_test = pd.read_csv(TEST_SET_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = scaler.feature_names_in_
    X_test_scaled = scaler.transform(df_test[feature_names].values)
    y_test = df_test['label'].values
    X_train_full_scaled = scaler.transform(df_train_full[feature_names].values)
    y_train_full = df_train_full['label'].values
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full_scaled, y_train_full, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, stratify=y_train_full
    )
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_tensor_x = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_tensor_y = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
    print("✅ 数据准备完毕。")

    # --- 模型初始化、损失函数、优化器 ---
    benign_ratio = (y_train_full == 0).sum() / len(y_train_full)
    # ✅ 3. 初始化CNN模型
    model = CNN_Classifier(feature_dim=FEATURE_DIM).to(device)
    criterion = FocalLoss(alpha=benign_ratio, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BEST_PARAMS['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=False)

    print("\n[步骤1] 正在训练模型...")
    best_val_loss = float('inf')
    for epoch in tqdm(range(EPOCHS), desc="Training"):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # CNN的训练过程与MLP完全一样
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(val_tensor_x)
            val_loss = criterion(val_logits, val_tensor_y).item()
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CNN_HUNTER_MODEL_PATH)

    print(f"\n✅ 训练完成，最佳验证损失: {best_val_loss:.6f}")

    # --- 在验证集上寻找最佳决策阈值 ---
    print("\n[步骤2] 正在验证集上寻找最佳决策阈值...")
    final_model = CNN_Classifier(feature_dim=FEATURE_DIM).to(device)
    final_model.load_state_dict(torch.load(CNN_HUNTER_MODEL_PATH, map_location=device))
    final_model.eval()
    with torch.no_grad():
        val_probs = final_model.predict(val_tensor_x).cpu().numpy()
    best_threshold, best_f1 = 0.5, 0
    for threshold in np.arange(0.01, 1.0, 0.01):
        y_val_pred = (val_probs > threshold).astype(int)
        current_f1 = f1_score(y_val, y_val_pred, pos_label=1)
        if current_f1 > best_f1:
            best_f1, best_threshold = current_f1, threshold
    print(f"✅ 最佳阈值查找完毕: {best_threshold:.2f} (在该阈值下验证集F1分数为: {best_f1:.4f})")

    # --- 使用最佳阈值在测试集上进行最终评估 ---
    print("\n--- 最终'1D-CNN Hunter'在【留出测试集】上的真实性能报告 ---")
    with torch.no_grad():
        test_tensor_x = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        test_probs = final_model.predict(test_tensor_x).cpu().numpy()
        y_pred = (test_probs > best_threshold).astype(int)
    print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Bot (1)'], digits=4))

if __name__ == "__main__":
    main()