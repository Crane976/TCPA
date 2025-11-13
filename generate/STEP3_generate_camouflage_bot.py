# generate/STEP3_generate_camouflage_bot.py (FINAL 3-TIER ASYMMETRIC STRATEGY)
import pandas as pd
import numpy as np
import os
import sys
import joblib
import torch
from tqdm import tqdm

# ==========================================================
# --- 路径修正与模块导入 ---
# ==========================================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.style_transfer_cae import ConditionalAutoencoder
from models.bot_pattern_lstm import BotPatternLSTM
# ✅✅✅ 1. 导入最终的三层特征体系 ✅✅✅
from config import DEFENDER_SET, ATTACKER_KNOWLEDGE_SET, ATTACKER_ACTION_SET

# ==========================================================
# --- 1. 配置区 ---
# ==========================================================
# --- 输入 ---
TRAIN_SET_PATH = os.path.join(project_root, 'data', 'splits', 'training_set.csv')
SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')
CAE_MODEL_PATH = os.path.join(project_root, 'models', 'style_transfer_cae.pt')
LSTM_MODEL_PATH = os.path.join(project_root, 'models', 'bot_pattern_lstm_final.pt')  # 确保指向最终训练的模型

# --- 输出 ---
GENERATED_DIR = os.path.join(project_root, 'data', 'generated')
os.makedirs(GENERATED_DIR, exist_ok=True)
OUTPUT_CAMOUFLAGE_PATH = os.path.join(GENERATED_DIR, 'final_camouflage_bot.csv')

# --- 模型参数 (与 STEP2 保持一致) ---
CAE_FEATURE_DIM = len(ATTACKER_KNOWLEDGE_SET)
LATENT_DIM_CAE = 5
NUM_CLASSES_CAE = 2
INPUT_DIM_LSTM = LATENT_DIM_CAE
OUTPUT_DIM_LSTM = len(ATTACKER_ACTION_SET)
HIDDEN_DIM_LSTM = 64
COND_DIM_LSTM = NUM_CLASSES_CAE

# --- 生成参数 ---
NUM_TO_GENERATE = 40000
WINDOW_SIZE = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PERTURBATION_STRENGTH_ALPHA = 1.0


# ==========================================================
# --- 2. 辅助函数 ---
# ==========================================================
def create_latent_sequences_for_generation(data, window_size):
    sequences = []
    if len(data) >= window_size:
        for i in range(len(data) - window_size + 1):
            sequences.append(data[i:i + window_size])
    return np.array(sequences)


# ==========================================================
# --- 3. 主生成函数 ---
# ==========================================================
def main():
    print("=" * 60);
    print("🚀 扰动学习框架 (最终版 - 三层非对称策略) - STEP 3: 生成流量...");
    print("=" * 60)
    print(f"   >>> 当前扰动放大系数 (Alpha): {PERTURBATION_STRENGTH_ALPHA} <<<")
    print(f"使用设备: {device}")

    # --- 加载资产 ---
    try:
        df_train_full = pd.read_csv(TRAIN_SET_PATH)
        scaler = joblib.load(SCALER_PATH)
        # ✅ 2. 初始化模型时使用正确的维度
        cae_model = ConditionalAutoencoder(CAE_FEATURE_DIM, LATENT_DIM_CAE, NUM_CLASSES_CAE).to(device)
        cae_model.load_state_dict(torch.load(CAE_MODEL_PATH, map_location=device, weights_only=True))
        cae_model.eval()
        lstm_model = BotPatternLSTM(INPUT_DIM_LSTM, HIDDEN_DIM_LSTM, OUTPUT_DIM_LSTM, COND_DIM_LSTM).to(device)
        lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device, weights_only=True))
        lstm_model.eval()
    except FileNotFoundError as e:
        print(f"错误: 找不到核心文件 - {e}");
        return

    # --- 准备“母体”流量 ---
    print("\n正在准备良性流量作为生成母体...")
    df_benign_source = df_train_full[df_train_full['label'] == 0].copy().head(NUM_TO_GENERATE)
    if len(df_benign_source) < NUM_TO_GENERATE:
        print(f"警告: 训练集中良性流量不足 {NUM_TO_GENERATE}, 将使用 {len(df_benign_source)} 条。")
    if len(df_benign_source) < WINDOW_SIZE:
        print(f"错误: 可用良性流量不足 {WINDOW_SIZE} 条!");
        return

    # ✅ 3. 按三层体系准备数据
    # 先用scaler转换所有防御者能看到的特征
    X_benign_def_scaled = scaler.transform(df_benign_source[DEFENDER_SET].values)
    df_benign_scaled = pd.DataFrame(X_benign_def_scaled, columns=DEFENDER_SET)

    # --- 执行增量注入流程 ---
    print("\n开始执行 增量(Delta) 注入流程...")
    # 步骤1: 编码良性流量到潜在空间 (只使用攻击者认知集)
    print("  - 步骤1: 将'攻击者认知集'编码为潜在表示(z)...")
    with torch.no_grad():
        X_benign_knowledge_tensor = torch.tensor(df_benign_scaled[ATTACKER_KNOWLEDGE_SET].values,
                                                 dtype=torch.float32).to(device)
        benign_labels = torch.zeros(len(X_benign_knowledge_tensor), NUM_CLASSES_CAE, device=device);
        benign_labels[:, 0] = 1
        Z_benign_latent = cae_model.encode(X_benign_knowledge_tensor, benign_labels)

    # 步骤2: 用LSTM预测“增量” (输出维度为攻击者行动集)
    print("  - 步骤2: 用LSTM预测'攻击者行动集'上的特征增量(Delta)...")
    latent_sequences = create_latent_sequences_for_generation(Z_benign_latent.cpu().numpy(), WINDOW_SIZE)
    latent_sequences_tensor = torch.FloatTensor(latent_sequences).to(device)
    condition_tensor = torch.zeros(len(latent_sequences_tensor), NUM_CLASSES_CAE, device=device);
    condition_tensor[:, 1] = 1
    with torch.no_grad():
        predicted_deltas = lstm_model(latent_sequences_tensor, condition_tensor).cpu().numpy()

    # ✅ 4. 精确应用增量
    print(f"  - 步骤3: 正在应用预测的增量 (放大 {PERTURBATION_STRENGTH_ALPHA} 倍)...")
    num_generated = len(predicted_deltas)
    # 我们的“画布”是完整的防御者视野
    adversarial_features_scaled = np.copy(X_benign_def_scaled)
    # 找到行动集在大画布上的精确位置
    action_indices_in_defender_set = [DEFENDER_SET.index(f) for f in ATTACKER_ACTION_SET]

    for i in tqdm(range(num_generated), desc="应用扰动"):
        target_sample_index = i + WINDOW_SIZE - 1
        if target_sample_index >= len(adversarial_features_scaled): break
        adversarial_features_scaled[target_sample_index, action_indices_in_defender_set] += (
                predicted_deltas[i] * PERTURBATION_STRENGTH_ALPHA)

    # --- 后续处理 ---
    adversarial_features_scaled = np.clip(adversarial_features_scaled, 0, 1)
    # 我们只保留那些被成功扰动的样本
    final_generated_features = adversarial_features_scaled[WINDOW_SIZE - 1: WINDOW_SIZE - 1 + num_generated]

    print("\n正在反定标并将最终伪装流量保存到CSV...")
    # 反定标时， scaler期望得到DEFENDER_SET维度的输入
    final_features_original_scale = scaler.inverse_transform(final_generated_features)
    df_camouflage = pd.DataFrame(final_features_original_scale, columns=DEFENDER_SET)
    df_camouflage.to_csv(OUTPUT_CAMOUFLAGE_PATH, index=False)

    print(f"\n✅ {len(df_camouflage)} 条伪装Bot流量生成完毕！文件已保存到: {OUTPUT_CAMOUFLAGE_PATH}")


if __name__ == "__main__":
    main()