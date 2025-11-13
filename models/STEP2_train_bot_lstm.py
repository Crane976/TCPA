# models/STEP2_train_bot_lstm.py (FINAL 3-TIER ASYMMETRIC STRATEGY)
import pandas as pd
import numpy as np
import os
import sys
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

# ==========================================================
# --- Path Setup & Imports ---
# ==========================================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# ✅✅✅ 1. 导入最终的三层特征体系 ✅✅✅
from config import DEFENDER_SET, ATTACKER_KNOWLEDGE_SET, ATTACKER_ACTION_SET, set_seed
from models.style_transfer_cae import ConditionalAutoencoder
from models.bot_pattern_lstm import BotPatternLSTM
from models.mlp_architecture import MLP_Classifier

# ==========================================================
# --- 1. Configuration ---
# ==========================================================
# --- Paths ---
TRAIN_SET_PATH = os.path.join(project_root, 'data', 'splits', 'training_set.csv')
SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')
CAE_MODEL_PATH = os.path.join(project_root, 'models', 'style_transfer_cae.pt')
PROXY_HUNTER_MODEL_PATH = os.path.join(project_root, 'models', 'proxy_hunter_distilled.pt')
LSTM_MODEL_PATH = os.path.join(project_root, 'models', 'bot_pattern_lstm_final.pt')

# --- Model Parameters (适配三层体系) ---
# Proxy hunter sees the widest view
PROXY_FEATURE_DIM = len(DEFENDER_SET)
# CAE operates on our knowledge boundary
CAE_FEATURE_DIM = len(ATTACKER_KNOWLEDGE_SET)
LATENT_DIM_CAE = 5
NUM_CLASSES_CAE = 2
# LSTM operates on our action boundary
INPUT_DIM_LSTM = LATENT_DIM_CAE
OUTPUT_DIM_LSTM = len(ATTACKER_ACTION_SET)
HIDDEN_DIM_LSTM = 64
COND_DIM_LSTM = NUM_CLASSES_CAE

# --- Training Parameters ---
WINDOW_SIZE = 3
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.0005
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 最终解决方案: 双重“紧箍咒” ---
LAMBDA_ADV = 1.0  # 攻击动机
LAMBDA_L2 = 0.01  # 扰动大小惩罚
LABEL_SMOOTHING = 0.9  # 标签平滑


# ==========================================================
# --- 2. Custom Dataset ---
# ==========================================================
class AdversarialDataset(torch.utils.data.Dataset):
    def __init__(self, X_seq, Y_seq_delta, C_seq, source_features_for_proxy):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.Y_seq_delta = torch.tensor(Y_seq_delta, dtype=torch.float32)
        self.C_seq = torch.tensor(C_seq, dtype=torch.float32)
        # ✅ 2. 数据集现在携带防御者视野的完整特征，专供代理猎手使用
        self.source_features_for_proxy = torch.tensor(source_features_for_proxy, dtype=torch.float32)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return (self.X_seq[idx], self.Y_seq_delta[idx], self.C_seq[idx], self.source_features_for_proxy[idx])


# ==========================================================
# --- 3. Main Training Function ---
# ==========================================================
def main():
    set_seed(2025)
    print("=" * 60);
    print("🚀 对抗训练框架 (最终版 - 三层非对称策略)...");
    print("=" * 60)
    print(
        f"   >>> 防御者视野: {PROXY_FEATURE_DIM}维, 攻击者认知: {CAE_FEATURE_DIM}维, 攻击者行动: {OUTPUT_DIM_LSTM}维 <<<")
    print(f"   >>> 策略: 扰动学习 + 双重正则化 <<<")
    print(f"使用设备: {device}")

    # --- 1. Load Assets ---
    try:
        df_train_full = pd.read_csv(TRAIN_SET_PATH)
        scaler = joblib.load(SCALER_PATH)
        # ✅ 3. 初始化模型时使用正确的维度
        cae_model = ConditionalAutoencoder(CAE_FEATURE_DIM, LATENT_DIM_CAE, NUM_CLASSES_CAE).to(device)
        cae_model.load_state_dict(torch.load(CAE_MODEL_PATH, map_location=device, weights_only=True))
        proxy_hunter = MLP_Classifier(PROXY_FEATURE_DIM).to(device)
        proxy_hunter.load_state_dict(torch.load(PROXY_HUNTER_MODEL_PATH, map_location=device, weights_only=True))
    except FileNotFoundError as e:
        print(f"错误: 找不到核心文件 - {e}");
        return

    cae_model.eval();
    proxy_hunter.eval()
    for param in cae_model.parameters(): param.requires_grad = False
    for param in proxy_hunter.parameters(): param.requires_grad = False

    # --- 2. Data Preparation ---
    print("\n[步骤1] 正在准备三层非对称训练数据...")
    df_benign_train = df_train_full[df_train_full['label'] == 0].copy()
    df_bot_train = df_train_full[df_train_full['label'] == 1].copy()

    # KNN Pairing
    benign_anchors = df_benign_train[['Flow Duration']].values
    bot_anchors = df_bot_train[['Flow Duration']].values
    knn_model = NearestNeighbors(n_neighbors=1).fit(bot_anchors)
    _, indices = knn_model.kneighbors(benign_anchors)
    df_bot_paired = df_bot_train.iloc[indices.ravel()].reset_index(drop=True)
    df_benign_train = df_benign_train.reset_index(drop=True)

    # ✅ 4. 按三层体系准备数据
    # 先用scaler转换所有防御者能看到的特征
    X_benign_def_scaled = scaler.transform(df_benign_train[DEFENDER_SET].values)
    X_bot_paired_def_scaled = scaler.transform(df_bot_paired[DEFENDER_SET].values)
    X_bot_def_scaled = scaler.transform(df_bot_train[DEFENDER_SET].values)

    df_benign_scaled = pd.DataFrame(X_benign_def_scaled, columns=DEFENDER_SET)
    df_bot_paired_scaled = pd.DataFrame(X_bot_paired_def_scaled, columns=DEFENDER_SET)
    df_bot_scaled = pd.DataFrame(X_bot_def_scaled, columns=DEFENDER_SET)

    # Calculate Deltas (只在攻击者行动集上计算)
    source_action_features = df_benign_scaled[ATTACKER_ACTION_SET].values
    target_action_features = df_bot_paired_scaled[ATTACKER_ACTION_SET].values
    Y_delta_transform = target_action_features - source_action_features
    Y_delta_recon = np.zeros_like(df_bot_scaled[ATTACKER_ACTION_SET].values)

    # Encode to Latent Space (只使用攻击者认知集)
    with torch.no_grad():
        X_benign_knowledge_tensor = torch.tensor(df_benign_scaled[ATTACKER_KNOWLEDGE_SET].values,
                                                 dtype=torch.float32).to(device)
        benign_labels = torch.zeros(len(X_benign_knowledge_tensor), NUM_CLASSES_CAE, device=device);
        benign_labels[:, 0] = 1
        Z_latent_benign = cae_model.encode(X_benign_knowledge_tensor, benign_labels)

        X_bot_knowledge_tensor = torch.tensor(df_bot_scaled[ATTACKER_KNOWLEDGE_SET].values, dtype=torch.float32).to(
            device)
        bot_labels = torch.zeros(len(X_bot_knowledge_tensor), NUM_CLASSES_CAE, device=device);
        bot_labels[:, 1] = 1
        Z_latent_bot = cae_model.encode(X_bot_knowledge_tensor, bot_labels)

    # Create Sequences
    def create_sequences(latent, delta, source_features_for_proxy, window):
        X_seq, Y_seq, F_seq_proxy = [], [], []
        if len(latent) < window: return [np.array([])] * 3
        for i in range(len(latent) - window + 1):
            X_seq.append(latent[i: i + window])
            Y_seq.append(delta[i + window - 1])
            F_seq_proxy.append(source_features_for_proxy[i + window - 1])
        return np.array(X_seq), np.array(Y_seq), np.array(F_seq_proxy)

    X_seq_t, Y_seq_t, F_proxy_t = create_sequences(Z_latent_benign.cpu().numpy(), Y_delta_transform,
                                                   X_benign_def_scaled, WINDOW_SIZE)
    X_seq_r, Y_seq_r, F_proxy_r = create_sequences(Z_latent_bot.cpu().numpy(), Y_delta_recon, X_bot_def_scaled,
                                                   WINDOW_SIZE)

    X_seq = np.concatenate([X_seq_t, X_seq_r]);
    Y_seq = np.concatenate([Y_seq_t, Y_seq_r])
    F_proxy_seq = np.concatenate([F_proxy_t, F_proxy_r]);
    C_seq = np.zeros((len(X_seq), NUM_CLASSES_CAE));
    C_seq[:, 1] = 1

    # --- 3. Create DataLoader ---
    (X_train, X_val, Y_train, Y_val, C_train, C_val,
     F_proxy_train, F_proxy_val) = train_test_split(
        X_seq, Y_seq, C_seq, F_proxy_seq, test_size=0.1, random_state=2025)

    train_dataset = AdversarialDataset(X_train, Y_train, C_train, F_proxy_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"✅ 数据准备完毕, 训练样本数: {len(train_dataset)}")

    # --- 4. Initialize Models and Loss Functions ---
    lstm_model = BotPatternLSTM(INPUT_DIM_LSTM, HIDDEN_DIM_LSTM, OUTPUT_DIM_LSTM, COND_DIM_LSTM).to(device)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
    criterion_mse = nn.MSELoss()
    criterion_adv = nn.BCEWithLogitsLoss()
    action_indices_in_defender_set = [DEFENDER_SET.index(f) for f in ATTACKER_ACTION_SET]

    # --- 5. Adversarial Training Loop with Regularization ---
    print("\n[步骤2] 开始对抗训练...")
    for epoch in range(EPOCHS):
        lstm_model.train()
        total_recon_loss, total_adv_loss, total_l2_loss = 0, 0, 0

        for x_batch, y_delta_batch, c_batch, f_proxy_batch in train_loader:
            x_batch, y_delta_batch, c_batch, f_proxy_batch = (
                x.to(device) for x in [x_batch, y_delta_batch, c_batch, f_proxy_batch])

            predicted_delta = lstm_model(x_batch, c_batch)

            # ✅ 5. 在防御者视野的完整特征上，精确地应用扰动
            adversarial_features = f_proxy_batch.clone()
            adversarial_features[:, action_indices_in_defender_set] += predicted_delta
            adversarial_features = torch.clamp(adversarial_features, 0, 1)

            # --- Calculate Losses ---
            loss_recon = criterion_mse(predicted_delta, y_delta_batch)

            adversarial_logits = proxy_hunter(adversarial_features)
            smooth_target_labels = torch.full_like(adversarial_logits, LABEL_SMOOTHING)
            loss_adv = criterion_adv(adversarial_logits, smooth_target_labels)

            loss_l2 = torch.mean(torch.pow(predicted_delta, 2))

            total_loss = loss_recon + LAMBDA_ADV * loss_adv + LAMBDA_L2 * loss_l2

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_recon_loss += loss_recon.item();
            total_adv_loss += loss_adv.item();
            total_l2_loss += loss_l2.item()

        avg_recon = total_recon_loss / len(train_loader)
        avg_adv = total_adv_loss / len(train_loader)
        avg_l2 = total_l2_loss / len(train_loader)
        print(f"  -> Epoch {epoch + 1:2d}/{EPOCHS}, Recon: {avg_recon:.6f}, Adv: {avg_adv:.6f}, L2: {avg_l2:.6f}")

    # --- Save Final Model ---
    torch.save(lstm_model.state_dict(), LSTM_MODEL_PATH)
    print(f"\n✅ 最终版'正则化扰动'LSTM引擎已保存在: {LSTM_MODEL_PATH}")


if __name__ == "__main__":
    main()