# generate/STEP3_Variant_B_no_constraint.py
# Ablation Study Variant B: w/o Hard Constraints (No Physical Consistency)
# Adaptive for both CIC-IDS2017 and CSE-CIC-IDS2018

import pandas as pd
import numpy as np
import os
import sys
import joblib
import torch
from sklearn.cluster import KMeans  # 保留聚类

# ==========================================================
# --- 路径修正与模块导入 ---
# ==========================================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.append(project_root)

import config
from models.style_transfer_cae import ConditionalAutoencoder
from models.lstm_finetuner import LSTMFinetuner
from models.lstm_predictor import LSTMPredictor
from config import DEFENDER_SET, ATTACKER_KNOWLEDGE_SET, ATTACKER_ACTION_SET, COMPLEX_SET, set_seed

# ==========================================================
# --- 配置区 (基于 Config) ---
# ==========================================================
CLEAN_DATA_PATH = os.path.join(config.SPLITS_DIR, 'training_set.csv')
TEST_DATA_PATH = os.path.join(config.SPLITS_DIR, 'holdout_test_set.csv')

SCALER_PATH = config.SCALER_PATH
MODEL_DIR = config.MODEL_SAVE_DIR
CAE_MODEL_PATH = os.path.join(MODEL_DIR, 'style_transfer_cae.pt')
LSTM_FINETUNER_MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_finetuner.pt')
PREDICTOR_MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_reconciliation_predictor.pt')

# 🔥 动态输出文件名
OUTPUT_CSV_NAME = f'variant_B_no_constraint_{config.CURRENT_DATASET}.csv'
OUTPUT_CSV_PATH = os.path.join(project_root, 'data', 'generated', OUTPUT_CSV_NAME)

FEATURE_DIM_CAE = len(ATTACKER_KNOWLEDGE_SET)
LATENT_DIM_CAE = 5
NUM_CLASSES_CAE = 2
INPUT_DIM_LSTM_FINETUNER = len(ATTACKER_KNOWLEDGE_SET)
OUTPUT_DIM_LSTM_FINETUNER = len(ATTACKER_ACTION_SET)
INPUT_DIM_PREDICTOR = len(ATTACKER_ACTION_SET)
OUTPUT_DIM_PREDICTOR = len(COMPLEX_SET)

# --- 战术参数 ---
TACTICAL_SUPPRESSION_RATIO = 100
TACTICAL_WINDOW_CAP_2018 = 1000
MIMIC_INTENSITY = 0.98
NUM_BOT_CLUSTERS = 5
WATERMARK_KEY = 97
WATERMARK_FEATURE = 'Flow Duration'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# --- 辅助函数 ---
# ==========================================================
def calculate_adaptive_quantity():
    """自适应计算生成数量"""
    print(f"\n🔍 [Variant B] 正在分析测试集规模: {config.CURRENT_DATASET} ...")
    if not os.path.exists(TEST_DATA_PATH):
        return 40000

    df_test = pd.read_csv(TEST_DATA_PATH)
    label_col = 'Label' if 'Label' in df_test.columns else 'label'

    if df_test[label_col].dtype == object:
        df_test[label_col] = df_test[label_col].apply(lambda x: 0 if str(x).lower() == 'benign' else 1)

    real_bot_count = len(df_test[df_test[label_col] == 1])

    if config.CURRENT_DATASET == 'CIC-IDS2017':
        target_num = real_bot_count * TACTICAL_SUPPRESSION_RATIO
    elif config.CURRENT_DATASET == 'CSE-CIC-IDS2018':
        target_num = min(real_bot_count, TACTICAL_WINDOW_CAP_2018) * TACTICAL_SUPPRESSION_RATIO
    else:
        target_num = 40000

    print(f"   -> 目标 Bot 数: {real_bot_count}")
    print(f"   -> 计划生成数: {target_num}")
    return int(target_num)


def inject_watermark_variant_B(df, key, feature_name):
    """
    Variant B 专用的水印注入:
    只修改 Duration，**故意不** 更新 Bytes/s 和 Pkts/s。
    这样会人为制造出物理逻辑漏洞，模拟没有硬约束的情况。
    """
    print(f"\n🌊 [步骤7] 注入水印 (Variant B Mode)...")
    df_w = df.copy()
    values = df_w[feature_name].values.astype(int)
    residuals = values % key
    new_values = values - residuals
    mask_too_small = (new_values <= 0)
    new_values[mask_too_small] += key
    df_w[feature_name] = new_values

    print("   -> ⚠️ 注意: Variant B 不会同步更新关联特征 (Rate/Pkts)，故意保留不自洽性!")
    # 这里我们什么都不做，直接返回，这就是消融实验的精髓

    return df_w


# ==========================================================
# --- 主函数 ---
# ==========================================================
def main():
    set_seed(2025)
    print("=" * 60)
    print(f"🚀 [消融实验 Variant B] 无硬约束 (No Physical Constraints)")
    print(f"   Dataset: {config.CURRENT_DATASET}")
    print("=" * 60)

    # 0. 确定数量
    NUM_TO_GENERATE = calculate_adaptive_quantity()
    if NUM_TO_GENERATE <= 0: return

    # 1. 加载模型 (不变)
    print("\n[步骤1] 加载模型与数据...")
    scaler = joblib.load(SCALER_PATH)
    predictor = LSTMPredictor(INPUT_DIM_PREDICTOR, OUTPUT_DIM_PREDICTOR).to(device)
    try:
        predictor.load_state_dict(torch.load(PREDICTOR_MODEL_PATH, map_location=device))
    except:
        predictor.load_state_dict(torch.load(PREDICTOR_MODEL_PATH, map_location=device))
    predictor.eval()

    cae_model = ConditionalAutoencoder(FEATURE_DIM_CAE, LATENT_DIM_CAE, NUM_CLASSES_CAE).to(device)
    cae_model.load_state_dict(torch.load(CAE_MODEL_PATH, map_location=device))
    cae_model.eval()

    lstm_finetuner = LSTMFinetuner(INPUT_DIM_LSTM_FINETUNER, OUTPUT_DIM_LSTM_FINETUNER).to(device)
    lstm_finetuner.load_state_dict(torch.load(LSTM_FINETUNER_MODEL_PATH, map_location=device))
    lstm_finetuner.eval()

    df_clean_full = pd.read_csv(CLEAN_DATA_PATH)

    # 1.1 准备 Benign 母体
    df_benign_source = df_clean_full[df_clean_full['label'] == 0].sample(n=NUM_TO_GENERATE, replace=True,
                                                                         random_state=2025)

    # 1.2 准备 Bot 全量数据
    df_bot_all = df_clean_full[df_clean_full['label'] == 1]
    # 针对 2018 采样聚类
    if len(df_bot_all) > 20000:
        df_bot_clustering = df_bot_all.sample(n=20000, random_state=2025)
    else:
        df_bot_clustering = df_bot_all

    # 1.5 聚类 (保留)
    print("\n[步骤1.5] 执行聚类聚焦 (Ablation: No, 聚类保留)...")
    bot_scaled_full = scaler.transform(df_bot_clustering[DEFENDER_SET])
    kmeans = KMeans(n_clusters=NUM_BOT_CLUSTERS, random_state=2025, n_init=10)
    kmeans.fit(bot_scaled_full)
    centers_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
    df_bot_centers = pd.DataFrame(centers_unscaled, columns=DEFENDER_SET)
    tutor_indices = np.random.randint(0, NUM_BOT_CLUSTERS, size=NUM_TO_GENERATE)
    df_bot_tutors = df_bot_centers.iloc[tutor_indices].reset_index(drop=True)

    # 2. 风格植入 (不变)
    print("\n[步骤2] TIER 1: 执行点对点风格植入...")
    with torch.no_grad():
        source_scaled = scaler.transform(df_benign_source[DEFENDER_SET])
        k_indices = [DEFENDER_SET.index(c) for c in ATTACKER_KNOWLEDGE_SET]

        X_benign_full = torch.tensor(source_scaled, dtype=torch.float32).to(device)
        X_benign = X_benign_full[:, k_indices]
        c_benign = torch.tensor([1.0, 0.0], dtype=torch.float32).expand(len(X_benign), -1).to(device)
        z_benign = cae_model.encode(X_benign, c_benign)

        tutors_scaled = scaler.transform(df_bot_tutors[DEFENDER_SET])
        X_bot_full = torch.tensor(tutors_scaled, dtype=torch.float32).to(device)
        X_bot = X_bot_full[:, k_indices]
        c_bot_input = torch.tensor([0.0, 1.0], dtype=torch.float32).expand(len(X_bot), -1).to(device)
        z_bot = cae_model.encode(X_bot, c_bot_input)

        z_hybrid = (1 - MIMIC_INTENSITY) * z_benign + MIMIC_INTENSITY * z_bot
        c_bot_target = torch.tensor([0.0, 1.0], dtype=torch.float32).expand(len(z_hybrid), -1).to(device)
        generated_knowledge_features_scaled = cae_model.decode(z_hybrid, c_bot_target)

    # 3. LSTM (不变)
    print("\n[步骤3] TIER 2: LSTM 微调...")
    with torch.no_grad():
        input_for_lstm = generated_knowledge_features_scaled.unsqueeze(1)
        refined_action = lstm_finetuner(input_for_lstm)
        fused_action = np.clip(refined_action.cpu().numpy(), 0, 1)

    # 4. 预测 (不变)
    print("\n[步骤4] TIER 3: 衍生特征预测...")
    with torch.no_grad():
        input_predictor = torch.tensor(fused_action, dtype=torch.float32).unsqueeze(1).to(device)
        predicted_complex = predictor(input_predictor).cpu().numpy()
        predicted_complex = np.clip(predicted_complex, 0, 1)

    # 5. 逆向缩放 (不变)
    print("\n[步骤5] 逆向缩放...")
    X_gen_full = np.zeros((NUM_TO_GENERATE, len(DEFENDER_SET)))

    for i, col in enumerate(ATTACKER_ACTION_SET):
        col_idx = DEFENDER_SET.index(col)
        X_gen_full[:, col_idx] = fused_action[:, i]

    for i, col in enumerate(COMPLEX_SET):
        col_idx = DEFENDER_SET.index(col)
        X_gen_full[:, col_idx] = predicted_complex[:, i]

    X_gen_original = scaler.inverse_transform(X_gen_full)
    df_final = pd.DataFrame(X_gen_original, columns=DEFENDER_SET)

    # ------------------------------------------------------------------
    # ❌ 核心消融点: 移除硬约束 (No Hard Constraints)
    # ------------------------------------------------------------------
    print("\n[步骤6] ❌ 跳过物理硬约束校准 (Ablation: No Constraints)...")
    print("   -> 直接使用神经网络预测的原始值 (存在 Rate != Total/Duration 逻辑漏洞)")

    # 仅补全缺失列 (Calculable Set 在 Tier-123 中没有预测)
    # 那些没有被 LSTM 预测的特征 (比如 Flow Bytes/s)，我们必须给一个值，否则 Scaler 报错。
    # 为了体现"无约束"，我们给它们赋 0，或者随机数，或者保留 NaN (如果评估脚本能处理)
    # 最公平的做法：不计算。如果必须计算，就用错误的公式算 (比如只除以1，不除以Duration)
    # 这里我们选择补 0，模拟攻击者忘记处理这些依赖特征。
    for col in DEFENDER_SET:
        if col not in df_final.columns:
            df_final[col] = 0

    # 重新排列列顺序
    df_final = df_final[DEFENDER_SET]

    # 7. 水印 (修改版，不更新关联特征)
    df_final_watermarked = inject_watermark_variant_B(df_final, WATERMARK_KEY, WATERMARK_FEATURE)
    df_final_watermarked['Label'] = 1

    df_final_watermarked.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n✅ Variant B (No Constraint) 生成完毕: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()