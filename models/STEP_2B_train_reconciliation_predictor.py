# models/STEP_2B_train_reconciliation_predictor.py (UPGRADED TO LSTM)

import pandas as pd
import os
import sys
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.append(project_root)

from config import DEFENDER_SET, ATTACKER_ACTION_SET, set_seed
from models.lstm_predictor import LSTMPredictor  # ✅ 导入我们全新的LSTM预测器模型

# --- 配置区 ---
TRAIN_SET_PATH = os.path.join(project_root, 'data', 'splits', 'training_set.csv')
SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')
# ✅ 输出: LSTM预测器模型的路径，使用 .pt 后缀
PREDICTOR_MODEL_PATH = os.path.join(project_root, 'models', 'lstm_reconciliation_predictor.pt')

# --- 模型参数 ---
INPUT_DIM = len(ATTACKER_ACTION_SET)
OUTPUT_DIM = len(list(set(DEFENDER_SET) - set(ATTACKER_ACTION_SET)))

# --- 训练参数 ---
EPOCHS = 150  # 深度学习模型通常需要更多轮次
BATCH_SIZE = 64
LEARNING_RATE = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    set_seed(2025)
    print("=" * 60);
    print("🚀 协调框架(升级版) - STEP 2B: 训练LSTM关联特征预测器...");
    print("=" * 60)

    derived_features_set = sorted(list(set(DEFENDER_SET) - set(ATTACKER_ACTION_SET)))
    print(f"   >>> 目标: 学习用 {INPUT_DIM}维 '行动集' 去预测 {OUTPUT_DIM}维 '衍生集' <<<")

    print("\n[步骤1] 加载数据和Scaler...")
    df_train_full = pd.read_csv(TRAIN_SET_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("\n[步骤2] 准备真实Bot流量用于训练...")
    df_bot_train = df_train_full[df_train_full['label'] == 1].copy()

    # ✅ 核心修改: 神经网络需要归一化的数据！
    bot_scaled_full = scaler.transform(df_bot_train[DEFENDER_SET])
    df_bot_scaled = pd.DataFrame(bot_scaled_full, columns=DEFENDER_SET)

    # 输入X: 9维可控特征 (scaled)
    X_train = df_bot_scaled[ATTACKER_ACTION_SET].values
    # 输出Y: 14维衍生特征 (scaled)
    Y_train = df_bot_scaled[derived_features_set].values

    # 为LSTM准备数据: [样本数, 序列长度=1, 特征维度=9]
    X_train_seq = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)

    dataset = TensorDataset(X_train_seq, Y_train_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"✅ 数据准备完毕, 使用 {len(dataset)} 条Bot样本。")

    print("\n[步骤3] 开始训练LSTM回归预测器...")
    model = LSTMPredictor(INPUT_DIM, OUTPUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()  # 均方误差损失，适用于回归任务

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  -> Epoch {epoch + 1:3d}/{EPOCHS}, Train Loss: {total_loss / len(loader):.6f}")

    torch.save(model.state_dict(), PREDICTOR_MODEL_PATH)
    print(f"\n✅ LSTM关联特征预测器已成功保存到: {PREDICTOR_MODEL_PATH}")


if __name__ == "__main__":
    main()