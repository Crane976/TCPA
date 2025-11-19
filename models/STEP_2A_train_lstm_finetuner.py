# models/STEP_2A_train_lstm_finetuner.py
import pandas as pd
import numpy as np
import os
import sys
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.append(project_root)

from config import DEFENDER_SET, ATTACKER_KNOWLEDGE_SET, ATTACKER_ACTION_SET, set_seed
from models.lstm_finetuner import LSTMFinetuner  # 导入新模型

# --- 配置区 ---
TRAIN_SET_PATH = os.path.join(project_root, 'data', 'splits', 'training_set.csv')
SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')
LSTM_FINETUNER_MODEL_PATH = os.path.join(project_root, 'models', 'lstm_finetuner.pt')  # 新模型路径

# --- 模型参数 ---
INPUT_DIM = len(ATTACKER_KNOWLEDGE_SET)  # 13维
OUTPUT_DIM = len(ATTACKER_ACTION_SET)  # 9维
HIDDEN_DIM = 64

# --- 训练参数 ---
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    set_seed(2025)
    print("=" * 60);
    print("🚀 三级框架 - STEP 2A: 训练LSTM精调器 (战术层)...");
    print("=" * 60)
    print(f"   >>> 目标: 学习从13维'认知集'精确映射到9维'行动集' <<<")

    print("\n[步骤1] 加载数据和Scaler...")
    df_train_full = pd.read_csv(TRAIN_SET_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("\n[步骤2] 准备真实Bot流量用于训练...")
    df_bot_train = df_train_full[df_train_full['label'] == 1].copy()

    # 先对所有Bot流量的23维特征进行缩放
    bot_scaled_full = scaler.transform(df_bot_train[DEFENDER_SET])
    df_bot_scaled = pd.DataFrame(bot_scaled_full, columns=DEFENDER_SET)

    # 输入X: 13维认知集 (scaled)
    X_train = df_bot_scaled[ATTACKER_KNOWLEDGE_SET].values
    # 输出Y: 9维行动集 (scaled)
    Y_train = df_bot_scaled[ATTACKER_ACTION_SET].values

    # 为LSTM准备数据: [样本数, 序列长度=1, 特征维度]
    X_train_seq = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)

    dataset = TensorDataset(X_train_seq, Y_train_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"✅ 数据准备完毕, 使用 {len(dataset)} 条Bot样本。")

    print("\n[步骤3] 开始训练LSTM精调器...")
    model = LSTMFinetuner(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

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

    torch.save(model.state_dict(), LSTM_FINETUNER_MODEL_PATH)
    print(f"\n✅ LSTM精调器已保存到: {LSTM_FINETUNER_MODEL_PATH}")


if __name__ == "__main__":
    main()