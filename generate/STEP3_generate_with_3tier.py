# generate/STEP3_generate_with_3tier.py
# (FINAL VERSION: ADAPTIVE TSR 100:1 + CLUSTERED FOCUS + HARD CONSTRAINTS + WATERMARK)

import pandas as pd
import numpy as np
import os
import sys
import joblib
import torch
from sklearn.cluster import KMeans

# ==========================================================
# --- 路径修正与模块导入 ---
# ==========================================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.append(project_root)

# 🔥 导入 config 以获取当前数据集信息
import config
from models.style_transfer_cae import ConditionalAutoencoder
from models.lstm_finetuner import LSTMFinetuner
from models.lstm_predictor import LSTMPredictor
from config import DEFENDER_SET, ATTACKER_KNOWLEDGE_SET, ATTACKER_ACTION_SET, COMPLEX_SET, set_seed

# ==========================================================
# --- 配置区 (路径与模型) ---
# ==========================================================
# 训练集路径 (用于提取良性载体和Bot风格)
CLEAN_DATA_PATH = os.path.join(project_root, 'data', 'splits', 'training_set.csv')
# 🔥 测试集路径 (新增：用于侦察真实Bot数量，计算压制比)
TEST_DATA_PATH = os.path.join(project_root, 'data', 'splits', 'holdout_test_set.csv')

SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')
CAE_MODEL_PATH = os.path.join(project_root, 'models', 'style_transfer_cae.pt')
LSTM_FINETUNER_MODEL_PATH = os.path.join(project_root, 'models', 'lstm_finetuner.pt')
PREDICTOR_MODEL_PATH = os.path.join(project_root, 'models', 'lstm_reconciliation_predictor.pt')

# 🔥 动态输出路径：根据数据集名称自动命名，防止覆盖
output_filename = f'final_camouflage_{config.CURRENT_DATASET}_TSR100.csv'
OUTPUT_CSV_PATH = os.path.join(project_root, 'data', 'generated', output_filename)

FEATURE_DIM_CAE = len(ATTACKER_KNOWLEDGE_SET)
LATENT_DIM_CAE = 5
NUM_CLASSES_CAE = 2
INPUT_DIM_LSTM_FINETUNER = len(ATTACKER_KNOWLEDGE_SET)
OUTPUT_DIM_LSTM_FINETUNER = len(ATTACKER_ACTION_SET)
INPUT_DIM_PREDICTOR = len(ATTACKER_ACTION_SET)
OUTPUT_DIM_PREDICTOR = len(COMPLEX_SET)

# --- 战术参数 ---
# 注意：NUM_TO_GENERATE 不再硬编码，而是由 calculate_adaptive_quantity() 计算
TACTICAL_SUPPRESSION_RATIO = 100  # 核心战术指标 100:1
TACTICAL_WINDOW_CAP_2018 = 1000  # 2018数据集的战术窗口上限 (只压制前1000个Bot)

# 模仿强度 (0.98)
MIMIC_INTENSITY = 0.98

# Bot 聚类簇数
NUM_BOT_CLUSTERS = 5

# --- 水印参数 (溯源核心) ---
WATERMARK_KEY = 97
WATERMARK_FEATURE = 'Flow Duration'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# --- 自适应数量计算函数 (修复版) ---
# ==========================================================
def calculate_adaptive_quantity():
    """
    根据当前数据集和测试集中的真实Bot数量，计算符合 100:1 压制比的生成数量。
    包含标签类型自动识别和数量兜底逻辑。
    """
    print(f"\n🔍 [战术侦察] 正在分析测试集: {config.CURRENT_DATASET} ...")

    if not os.path.exists(TEST_DATA_PATH):
        print(f"   -> ❌ 警告: 未找到测试集文件: {TEST_DATA_PATH}")
        print("   -> ⚠️ 启用默认兜底数量: 40000")
        return 40000

    # 读取测试集 Label
    try:
        # 只读取 Label 列以加速
        df_test = pd.read_csv(TEST_DATA_PATH)
        # 兼容性处理：检查列名是 'Label' 还是 'label'
        label_col = 'Label' if 'Label' in df_test.columns else 'label'

        # 打印一下当前的标签分布，方便调试
        unique_labels = df_test[label_col].unique()
        print(f"   -> DEBUG: 测试集包含的标签类型: {unique_labels}")

        # --- 核心修复: 多重匹配逻辑 ---
        # 1. 尝试匹配数字 1
        real_bot_count = len(df_test[df_test[label_col] == 1])

        # 2. 如果没找到，尝试匹配字符串 'Bot' (或 config 中定义的 malicious label)
        if real_bot_count == 0:
            target_str = getattr(config, 'MALICIOUS_LABEL', 'Bot')  # 默认为 'Bot'
            real_bot_count = len(df_test[df_test[label_col] == target_str])

        print(f"   -> 侦测到测试集中真实 Bot 数量: {real_bot_count}")

        # --- 兜底逻辑 ---
        if real_bot_count == 0:
            print("   -> ⚠️ 警告: 未能检测到任何 Bot 样本 (可能是标签不匹配或测试集全为良性)。")
            print("   -> ⚠️ 启用强制兜底模式: 默认生成 40,000 条，以防止程序崩溃。")
            return 40000

        # --- 正常计算逻辑 ---
        target_num = 0

        if config.CURRENT_DATASET == 'CIC-IDS2017':
            # 2017: 全量压制
            target_num = real_bot_count * TACTICAL_SUPPRESSION_RATIO
            print(f"   -> 战术模式: 全量饱和打击 (Full Scale)")

        elif config.CURRENT_DATASET == 'CSE-CIC-IDS2018':
            # 2018: 战术窗口采样
            tactical_targets = min(real_bot_count, TACTICAL_WINDOW_CAP_2018)
            target_num = tactical_targets * TACTICAL_SUPPRESSION_RATIO
            print(f"   -> 战术模式: 战术窗口压制 (Tactical Window Cap: {TACTICAL_WINDOW_CAP_2018} Targets)")

        else:
            # 默认
            target_num = 40000
            print("   -> 战术模式: 默认设置")

        print(f"   -> ⚠️ 最终确定生成数量 (NUM_TO_GENERATE): {target_num}")
        return int(target_num)

    except Exception as e:
        print(f"   -> ❌ 侦察阶段发生错误: {e}")
        print("   -> ⚠️ 启用异常兜底模式: 默认生成 40,000 条")
        return 40000


# ==========================================================
# --- 水印注入函数 (保持原样) ---
# ==========================================================
def inject_watermark(df, key, feature_name):
    """
    在指定特征中注入模运算水印 (LSB Steganography)
    逻辑: 修改数值，使其 % key == 0
    """
    print(f"\n🌊 [步骤7] 正在注入溯源水印 (Key={key}, Feature={feature_name})...")

    # 复制一份以免影响原数据指针
    df_w = df.copy()

    # 获取原始值并转为整数 (微秒级时间戳本身就是整数)
    values = df_w[feature_name].values.astype(int)

    # 计算余数 (Residuals)
    residuals = values % key

    # 修改值: 减去余数，使其能被 key 整除
    new_values = values - residuals

    # 修正边界情况: Duration 不能为 0 或负数
    # 如果减去余数后 <= 0，则加一个 Key，保证它是正数且依然能被 Key 整除
    mask_too_small = (new_values <= 0)
    new_values[mask_too_small] += key

    df_w[feature_name] = new_values

    # 验证注入率
    success_rate = np.mean(df_w[feature_name] % key == 0)
    print(f"   -> 水印注入完成。理论验证通过率: {success_rate * 100:.2f}%")

    # ⚠️ 关键步骤: 重新计算速率特征以保持硬约束自洽
    # 因为 Flow Duration 变了，Bytes/s 和 Pkts/s 必须同步变
    print("   -> 正在同步更新关联特征 (Bytes/s, Pkts/s) 以维持数学自洽...")

    duration_sec = df_w['Flow Duration'] / 1e6  # 微秒转秒

    if 'Total Length of Fwd Packets' in df_w.columns:
        total_bytes = df_w['Total Length of Fwd Packets'] + df_w['Total Length of Bwd Packets']
        df_w['Flow Bytes/s'] = total_bytes / (duration_sec + 1e-9)

    if 'Total Fwd Packets' in df_w.columns:
        total_pkts = df_w['Total Fwd Packets'] + df_w['Total Backward Packets']
        df_w['Flow Packets/s'] = total_pkts / (duration_sec + 1e-9)

    return df_w


# ==========================================================
# --- 主函数 ---
# ==========================================================
def main():
    set_seed(2025)

    # 🔥 步骤0: 自适应计算生成数量
    NUM_TO_GENERATE = calculate_adaptive_quantity()

    # 再次检查，防止生成数为0
    if NUM_TO_GENERATE <= 0:
        print("❌ 错误: 生成数量为 0，强制退出以避免报错。")
        return

    print("=" * 60)
    print(f"🚀 (Decoy + ClusterFocus + Traceability) STEP 3: 生成 ({config.CURRENT_DATASET})...")
    print("=" * 60)
    print(f"   生成数量: {NUM_TO_GENERATE} (Based on 100:1 TSR)")
    print(f"   模仿强度: {MIMIC_INTENSITY}")
    print(f"   Bot聚类数: {NUM_BOT_CLUSTERS}")
    print(f"   溯源密钥: {WATERMARK_KEY}")

    # --- 1. 加载模型及数据 ---
    print("\n[步骤1] 加载模型及清洗后的数据...")
    scaler = joblib.load(SCALER_PATH)

    predictor = LSTMPredictor(INPUT_DIM_PREDICTOR, OUTPUT_DIM_PREDICTOR).to(device)
    # 增加 weights_only=False 以兼容旧版 PyTorch 保存的模型，防止 FutureWarning 刷屏
    # 如果你的 PyTorch 版本较新且模型是新的，可以尝试去掉，但为了稳妥这里先不管警告
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
    # 使用计算出的 NUM_TO_GENERATE
    # 🔥 关键修复: 如果 NUM_TO_GENERATE > 0 才能采样，否则会报错
    print(f"   -> 正在从背景流量中采样 {NUM_TO_GENERATE} 条作为载体...")
    df_benign_source = df_clean_full[df_clean_full['label'] == 0].sample(n=NUM_TO_GENERATE, replace=True,
                                                                         random_state=2025)

    # 1.2 准备 Bot 全量数据 (用于聚类)
    df_bot_all = df_clean_full[df_clean_full['label'] == 1]

    # 针对 IDS2018 数据量过大的优化：如果Bot太多，聚类时采样一下以提速 (不影响后续逻辑)
    if len(df_bot_all) > 20000:
        print(f"   -> (优化) Bot样本过多 ({len(df_bot_all)})，采样 20,000 个用于提取聚类风格...")
        df_bot_all_for_cluster = df_bot_all.sample(n=20000, random_state=2025)
    else:
        df_bot_all_for_cluster = df_bot_all

    print(f"✅ 准备完毕: {len(df_benign_source)} Benign 母体, {len(df_bot_all_for_cluster)} 真实 Bot 样本(用于聚类)。")

    # --- 1.5 Bot 风格聚类 (寻找最强特征) ---
    print(f"\n[步骤1.5] 对真实 Bot 进行聚类 (K={NUM_BOT_CLUSTERS}) 以提取纯粹风格...")

    # 缩放 Bot 数据
    bot_scaled_full = scaler.transform(df_bot_all_for_cluster[DEFENDER_SET])

    # 执行 KMeans
    kmeans = KMeans(n_clusters=NUM_BOT_CLUSTERS, random_state=2025, n_init=10)
    kmeans.fit(bot_scaled_full)

    # 获取聚类中心 (Scaled状态)
    centers_scaled = kmeans.cluster_centers_

    # 将中心逆向缩放回原始空间，构建 DataFrame
    centers_unscaled = scaler.inverse_transform(centers_scaled)
    df_bot_centers = pd.DataFrame(centers_unscaled, columns=DEFENDER_SET)

    print(f"   -> 成功提取 {NUM_BOT_CLUSTERS} 个 Bot 风格中心。")

    # 随机分配导师：让 NUM_TO_GENERATE 个母体随机选择这 5 个中心之一进行模仿
    tutor_indices = np.random.randint(0, NUM_BOT_CLUSTERS, size=NUM_TO_GENERATE)
    df_bot_tutors = df_bot_centers.iloc[tutor_indices].reset_index(drop=True)

    print(f"   -> 导师分配完毕: 所有生成样本将强制模仿这 {NUM_BOT_CLUSTERS} 个中心。")

    # --- 2. 强力风格植入 (TIER 1) ---
    print("\n[步骤2] TIER 1: 执行点对点风格植入...")
    with torch.no_grad():
        # 2.1 Benign Z (Source)
        source_scaled = scaler.transform(df_benign_source[DEFENDER_SET])
        df_source_scaled = pd.DataFrame(source_scaled, columns=DEFENDER_SET)
        X_benign = torch.tensor(df_source_scaled[ATTACKER_KNOWLEDGE_SET].values, dtype=torch.float32).to(device)
        c_benign = torch.tensor([1.0, 0.0], dtype=torch.float32).expand(len(X_benign), -1).to(device)
        z_benign = cae_model.encode(X_benign, c_benign)

        # 2.2 Bot Z (Centers as Tutors)
        tutors_scaled = scaler.transform(df_bot_tutors[DEFENDER_SET])
        df_tutors_scaled = pd.DataFrame(tutors_scaled, columns=DEFENDER_SET)
        X_bot = torch.tensor(df_tutors_scaled[ATTACKER_KNOWLEDGE_SET].values, dtype=torch.float32).to(device)
        c_bot_input = torch.tensor([0.0, 1.0], dtype=torch.float32).expand(len(X_bot), -1).to(device)
        z_bot = cae_model.encode(X_bot, c_bot_input)

        # 2.3 混合 (MIMIC_INTENSITY = 0.98)
        # 极度偏向 Bot，Benign 只提供极微小的扰动
        z_hybrid = (1 - MIMIC_INTENSITY) * z_benign + MIMIC_INTENSITY * z_bot

        # 2.4 解码
        c_bot_target = torch.tensor([0.0, 1.0], dtype=torch.float32).expand(len(z_hybrid), -1).to(device)
        generated_knowledge_features_scaled = cae_model.decode(z_hybrid, c_bot_target)

    # --- 3. LSTM 精调 (TIER 2) ---
    print("\n[步骤3] TIER 2: LSTM 战术微调...")
    with torch.no_grad():
        input_for_lstm = generated_knowledge_features_scaled.unsqueeze(1)
        refined_action = lstm_finetuner(input_for_lstm)

        df_knowledge_scaled = pd.DataFrame(generated_knowledge_features_scaled.cpu().numpy(),
                                           columns=ATTACKER_KNOWLEDGE_SET)
        original_action = torch.tensor(df_knowledge_scaled[ATTACKER_ACTION_SET].values, dtype=torch.float32).to(device)

        # 融合: LSTM 的权重保持 0.7
        fused_action = 0.3 * original_action + 0.7 * refined_action
        fused_action = np.clip(fused_action.cpu().numpy(), 0, 1)

    # --- 4. 衍生特征预测 (TIER 3) ---
    print("\n[步骤4] TIER 3: 衍生特征预测...")
    with torch.no_grad():
        input_predictor = torch.tensor(fused_action, dtype=torch.float32).unsqueeze(1).to(device)
        predicted_complex = predictor(input_predictor).cpu().numpy()
        predicted_complex = np.clip(predicted_complex, 0, 1)

    # --- 5. 逆向缩放 ---
    print("\n[步骤5] 逆向缩放...")
    df_temp_action = pd.DataFrame(0, index=range(NUM_TO_GENERATE), columns=DEFENDER_SET)
    df_temp_action[ATTACKER_ACTION_SET] = fused_action
    action_unscaled = pd.DataFrame(scaler.inverse_transform(df_temp_action), columns=DEFENDER_SET)[ATTACKER_ACTION_SET]

    df_temp_complex = pd.DataFrame(0, index=range(NUM_TO_GENERATE), columns=DEFENDER_SET)
    df_temp_complex[COMPLEX_SET] = predicted_complex
    complex_unscaled = pd.DataFrame(scaler.inverse_transform(df_temp_complex), columns=DEFENDER_SET)[COMPLEX_SET]

    df_final = pd.concat([action_unscaled, complex_unscaled], axis=1)

    # --- 6. 硬约束校准 ---
    print("\n[步骤6] 应用硬约束 (初次校准)...")
    # 基础计算
    df_final['Total Fwd Packets'] = df_final['Total Fwd Packets'].clip(lower=1)
    df_final['Total Backward Packets'] = df_final['Total Backward Packets'].clip(lower=0)
    df_final['Average Packet Size'] = df_final['Average Packet Size'].clip(lower=0)

    df_final['Total Length of Fwd Packets'] = df_final['Total Fwd Packets'] * df_final['Average Packet Size']
    df_final['Total Length of Bwd Packets'] = df_final['Total Backward Packets'] * df_final['Average Packet Size']

    total_pkts = df_final['Total Fwd Packets'] + df_final['Total Backward Packets']
    total_len = df_final['Total Length of Fwd Packets'] + df_final['Total Length of Bwd Packets']
    df_final['Packet Length Mean'] = total_len / (total_pkts + 1e-9)

    df_final['Flow Duration'] = df_final['Flow Duration'].clip(lower=1)
    duration_sec = df_final['Flow Duration'] / 1e6
    df_final['Flow Bytes/s'] = total_len / (duration_sec + 1e-9)
    df_final['Flow Packets/s'] = total_pkts / (duration_sec + 1e-9)
    df_final['Down/Up Ratio'] = df_final['Total Backward Packets'] / (df_final['Total Fwd Packets'] + 1e-9)

    # 极值修正
    cols_root = ['Fwd Packet Length', 'Bwd Packet Length', 'Flow IAT', 'Fwd IAT', 'Bwd IAT']
    for root in cols_root:
        if f'{root} Min' in df_final.columns and f'{root} Max' in df_final.columns:
            df_final[f'{root} Min'] = df_final[f'{root} Min'].clip(lower=0)
            df_final[f'{root} Max'] = np.maximum(df_final[f'{root} Max'], df_final[f'{root} Min'])
            if f'{root} Mean' in df_final.columns:
                df_final[f'{root} Mean'] = np.clip(df_final[f'{root} Mean'], df_final[f'{root} Min'],
                                                   df_final[f'{root} Max'])

    # 补全列
    for col in DEFENDER_SET:
        if col not in df_final.columns:
            df_final[col] = 0
    df_final = df_final[DEFENDER_SET]

    # --- 7. 注入溯源水印 (关键步骤) ---
    df_final_watermarked = inject_watermark(df_final, WATERMARK_KEY, WATERMARK_FEATURE)

    # --- 保存 ---
    df_final_watermarked.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n✅ {len(df_final_watermarked)} 条'聚类聚焦+可溯源'诱饵流量已保存到: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()