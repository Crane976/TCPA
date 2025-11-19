# generate/STEP3_generate_with_3tier.py (FINAL PURE DEEP LEARNING VERSION)
import pandas as pd
import numpy as np
import os
import sys
import joblib
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.append(project_root)

# ✅ 1. 导入所有需要的模型
from models.style_transfer_cae import ConditionalAutoencoder
from models.lstm_finetuner import LSTMFinetuner
from models.lstm_predictor import LSTMPredictor  # 导入新的LSTM预测器
from config import DEFENDER_SET, ATTACKER_KNOWLEDGE_SET, ATTACKER_ACTION_SET, set_seed

# --- 配置区 ---
# 输入
BENIGN_SOURCE_PATH = os.path.join(project_root, 'data', 'filtered', 'benign_traffic.csv')
SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')
TRAIN_SET_PATH = os.path.join(project_root, 'data', 'splits', 'training_set.csv')
CAE_MODEL_PATH = os.path.join(project_root, 'models', 'style_transfer_cae.pt')
LSTM_FINETUNER_MODEL_PATH = os.path.join(project_root, 'models', 'lstm_finetuner.pt')
# ✅ 2. 修改: 加载新的LSTM预测器模型
PREDICTOR_MODEL_PATH = os.path.join(project_root, 'models', 'lstm_reconciliation_predictor.pt')

# 输出
# ✅ 3. 修改: 使用新的输出文件名以作区分
OUTPUT_CSV_PATH = os.path.join(project_root, 'data', 'generated', 'final_camouflage_bot_3tier_lstm.csv')

# 模型参数
FEATURE_DIM_CAE = len(ATTACKER_KNOWLEDGE_SET)
LATENT_DIM_CAE = 5
NUM_CLASSES_CAE = 2
INPUT_DIM_LSTM_FINETUNER = len(ATTACKER_KNOWLEDGE_SET)
OUTPUT_DIM_LSTM_FINETUNER = len(ATTACKER_ACTION_SET)
# ✅ 4. 新增: LSTM预测器的参数
INPUT_DIM_PREDICTOR = len(ATTACKER_ACTION_SET)
OUTPUT_DIM_PREDICTOR = len(list(set(DEFENDER_SET) - set(ATTACKER_ACTION_SET)))

# 生成参数
NUM_TO_GENERATE = 40000
TRANSFER_ALPHA = 2.0
LSTM_FUSION_LAMBDA = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    set_seed(2025)
    print("=" * 60);
    print("🚀 纯深度学习三级框架 - STEP 3: 生成最终伪装流量...");
    print("=" * 60)
    print(f"   (LSTM融合系数 Lambda: {LSTM_FUSION_LAMBDA})")

    print("\n[步骤1] 加载全部三个DL模型及数据...")
    scaler = joblib.load(SCALER_PATH)

    # ✅ 5. 修改: 加载LSTMPredictor模型，而不是XGBoost
    predictor = LSTMPredictor(INPUT_DIM_PREDICTOR, OUTPUT_DIM_PREDICTOR).to(device)
    predictor.load_state_dict(torch.load(PREDICTOR_MODEL_PATH, map_location=device))
    predictor.eval()

    cae_model = ConditionalAutoencoder(FEATURE_DIM_CAE, LATENT_DIM_CAE, NUM_CLASSES_CAE).to(device)
    cae_model.load_state_dict(torch.load(CAE_MODEL_PATH, map_location=device))
    cae_model.eval()

    lstm_finetuner = LSTMFinetuner(INPUT_DIM_LSTM_FINETUNER, OUTPUT_DIM_LSTM_FINETUNER).to(device)
    lstm_finetuner.load_state_dict(torch.load(LSTM_FINETUNER_MODEL_PATH, map_location=device))
    lstm_finetuner.eval()

    df_benign_source = pd.read_csv(BENIGN_SOURCE_PATH).head(NUM_TO_GENERATE)
    df_train_full = pd.read_csv(TRAIN_SET_PATH)
    print("✅ 所有资产加载完毕。")

    # --- 步骤2 和 步骤3 的逻辑与之前完全相同，无需修改 ---
    # ... (从您的原代码中直接复制即可)
    print("\n[步骤2] TIER 1 (战略层): 使用CAE进行风格迁移...")
    with torch.no_grad():
        df_benign_train = df_train_full[df_train_full['label'] == 0];
        benign_scaled = scaler.transform(df_benign_train[DEFENDER_SET]);
        df_benign_scaled = pd.DataFrame(benign_scaled, columns=DEFENDER_SET)
        benign_knowledge = df_benign_scaled[ATTACKER_KNOWLEDGE_SET].values;
        c_benign = torch.tensor([1.0, 0.0]).expand(len(benign_knowledge), -1).to(device)
        z_benign_mean = torch.mean(
            cae_model.encode(torch.tensor(benign_knowledge, dtype=torch.float32).to(device), c_benign), dim=0)
        df_bot_train = df_train_full[df_train_full['label'] == 1];
        bot_scaled = scaler.transform(df_bot_train[DEFENDER_SET]);
        df_bot_scaled = pd.DataFrame(bot_scaled, columns=DEFENDER_SET)
        bot_knowledge = df_bot_scaled[ATTACKER_KNOWLEDGE_SET].values;
        c_bot = torch.tensor([0.0, 1.0]).expand(len(bot_knowledge), -1).to(device)
        z_bot_mean = torch.mean(cae_model.encode(torch.tensor(bot_knowledge, dtype=torch.float32).to(device), c_bot),
                                dim=0)
        transfer_vector = z_bot_mean - z_benign_mean
    source_scaled = scaler.transform(df_benign_source[DEFENDER_SET]);
    df_source_scaled = pd.DataFrame(source_scaled, columns=DEFENDER_SET)
    X_source_knowledge = df_source_scaled[ATTACKER_KNOWLEDGE_SET].values;
    X_source_tensor = torch.tensor(X_source_knowledge, dtype=torch.float32).to(device)
    c_benign_source = torch.tensor([1.0, 0.0]).expand(len(X_source_tensor), -1).to(device);
    c_bot_target = torch.tensor([0.0, 1.0]).expand(len(X_source_tensor), -1).to(device)
    with torch.no_grad():
        z_fake_bot = cae_model.encode(X_source_tensor, c_benign_source) + TRANSFER_ALPHA * transfer_vector
        generated_knowledge_features_scaled = cae_model.decode(z_fake_bot, c_bot_target)
    print("✅ 13维'粗加工'核心特征生成完毕。")

    print("\n[步骤3] TIER 2 (战术层): 使用LSTM进行特征精调...")
    with torch.no_grad():
        input_for_lstm = generated_knowledge_features_scaled.unsqueeze(1)
        refined_action_features_scaled = lstm_finetuner(input_for_lstm)
        df_knowledge_scaled = pd.DataFrame(generated_knowledge_features_scaled.cpu().numpy(),
                                           columns=ATTACKER_KNOWLEDGE_SET)
        original_action_features_scaled = torch.tensor(df_knowledge_scaled[ATTACKER_ACTION_SET].values,
                                                       dtype=torch.float32).to(device)
        fused_action_features_scaled = (
                                                   1 - LSTM_FUSION_LAMBDA) * original_action_features_scaled + LSTM_FUSION_LAMBDA * refined_action_features_scaled
        fused_action_features_scaled = fused_action_features_scaled.cpu().numpy()
        fused_action_features_scaled = np.clip(fused_action_features_scaled, 0, 1)
    print("✅ 9维'融合后'的行动特征生成完毕。")

    # --- 步骤4 和 步骤5 的逻辑被彻底重构 ---

    print("\n[步骤4] TIER 3 (执行层): 使用LSTM预测衍生特征...")
    with torch.no_grad():
        # ✅ 6. 为LSTM预测器准备输入: [N, 1, 9], 数据必须是scaled
        input_for_predictor = torch.tensor(fused_action_features_scaled, dtype=torch.float32).unsqueeze(1).to(device)

        # LSTM输出预测的14维衍生特征 (scaled)
        predicted_derived_features_scaled = predictor(input_for_predictor).cpu().numpy()
    print("✅ 14维衍生特征预测完毕。")

    print("\n[步骤5] 拼接、逆向缩放并保存最终流量...")
    # ✅ 7. 拼接完整的23维scaled特征
    df_fused_action = pd.DataFrame(fused_action_features_scaled, columns=ATTACKER_ACTION_SET)
    df_predicted_derived = pd.DataFrame(predicted_derived_features_scaled,
                                        columns=sorted(list(set(DEFENDER_SET) - set(ATTACKER_ACTION_SET))))

    df_final_scaled = pd.concat([df_fused_action, df_predicted_derived], axis=1)
    # 确保列的顺序与DEFENDER_SET完全一致，这步至关重要！
    df_final_scaled = df_final_scaled[DEFENDER_SET]

    # ✅ 8. 对完整的23维scaled特征进行一次性逆向缩放
    final_features_unscaled = scaler.inverse_transform(df_final_scaled.values)
    df_final_unscaled = pd.DataFrame(final_features_unscaled, columns=DEFENDER_SET)

    df_final_unscaled.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n✅ {len(df_final_unscaled)} 条'纯深度学习框架'伪装Bot流量已保存到: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()