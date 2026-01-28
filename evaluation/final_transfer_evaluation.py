# evaluation/final_transfer_evaluation.py (FINAL: LABEL NORMALIZATION FIX)
import pandas as pd
import numpy as np
import os
import sys
import joblib
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.append(project_root)

# 🔥 引入 config 模块
import config
from config import DEFENDER_SET, set_seed
from models.mlp_architecture import MLP_Classifier
from models.cnn_architecture import CNN_Classifier

# ==============================================================================
# 🎯 最佳阈值配置
# ==============================================================================
# 针对 IDS2017 的微调阈值
MODEL_THRESHOLDS_2017 = {
    "KNN Hunter": 0.90,
    "1D-CNN Hunter": 0.90,
    "XGBoost Hunter": 0.85,
    "MLP Hunter": 0.76
}

# 针对 IDS2018 的微调阈值
MODEL_THRESHOLDS_2018 = {
    "KNN Hunter": 0.30,
    "1D-CNN Hunter": 0.57,
    "XGBoost Hunter": 0.60,
    "MLP Hunter": 0.57
}

# 动态选择
MODEL_THRESHOLDS = MODEL_THRESHOLDS_2018 if config.CURRENT_DATASET == 'CSE-CIC-IDS2018' else MODEL_THRESHOLDS_2017


# ------------------------------------------------------------------------------
# 2. 核心评估函数 (保持不变)
# ------------------------------------------------------------------------------
def evaluate_hunter(hunter_name, hunter_model, X_cam_scaled, X_benign_test, X_bot_test, y_bot_test, device,
                    threshold=0.5):
    print("\n" + "=" * 50)
    print(f"--- 正在评估对抗: {hunter_name} ---")
    print(f"    👉 使用最佳决策阈值: {threshold:.2f}")

    # --- 统一预测接口 ---
    if isinstance(hunter_model, nn.Module):
        hunter_model.eval()
        with torch.no_grad():
            t_cam = torch.tensor(X_cam_scaled, dtype=torch.float32).to(device)
            t_benign = torch.tensor(X_benign_test, dtype=torch.float32).to(device)
            t_bot = torch.tensor(X_bot_test, dtype=torch.float32).to(device)

            preds_cam = (hunter_model.predict(t_cam) > threshold).int().cpu().numpy().flatten()
            preds_benign = (hunter_model.predict(t_benign) > threshold).int().cpu().numpy().flatten()
            preds_bot = (hunter_model.predict(t_bot) > threshold).int().cpu().numpy().flatten()

    else:
        def batch_predict_with_threshold(model, data, thr, batch_size=5000):
            n_samples = len(data)
            preds = []
            for i in range(0, n_samples, batch_size):
                batch = data[i:i + batch_size]
                probs = model.predict_proba(batch)[:, 1]
                batch_preds = (probs >= thr).astype(int)
                preds.extend(batch_preds)
            return np.array(preds)

        preds_cam = batch_predict_with_threshold(hunter_model, X_cam_scaled, threshold)
        preds_benign = batch_predict_with_threshold(hunter_model, X_benign_test, threshold)
        preds_bot = batch_predict_with_threshold(hunter_model, X_bot_test, threshold)

    # --- 计算指标 ---
    decoy_success_count = np.sum(preds_cam == 1)
    decoy_rate = decoy_success_count / len(X_cam_scaled) * 100

    base_tp = np.sum(preds_bot == 1)
    base_fn = len(y_bot_test) - base_tp
    recall = base_tp / (base_tp + base_fn) * 100

    base_fp = np.sum(preds_benign == 1)

    base_alerts = base_fp + base_tp
    mix_alerts = base_alerts + decoy_success_count

    dsr = (decoy_success_count / mix_alerts) * 100 if mix_alerts > 0 else 0
    base_precision = (base_tp / base_alerts) * 100 if base_alerts > 0 else 0
    hunter_precision_decayed = (base_tp / mix_alerts) * 100 if mix_alerts > 0 else 0

    print(f"  - 诱饵生成成功数 (Decoy Success): {decoy_success_count} / {len(X_cam_scaled)} ({decoy_rate:.2f}%)")
    print(f"  - 真实Bot捕获率 (Recall): {recall:.2f}%")
    print(f"  - 原始误报数 (Benign FP): {base_fp}")
    print("---------------------------------------------")
    print(f"  🎯 警报污染率 (DSR): {dsr:.2f}%")
    print(f"  📉 精确率从 {base_precision:.2f}% 衰减为: {hunter_precision_decayed:.2f}%")

    return {
        "Hunter": hunter_name,
        "Threshold": threshold,
        "Decoy Rate (%)": decoy_rate,
        "Recall (%)": recall,
        "Base Precision (%)": base_precision,
        "Decayed Precision (%)": hunter_precision_decayed,
        "DSR (Pollution) (%)": dsr
    }


# ------------------------------------------------------------------------------
# 3. 主流程
# ------------------------------------------------------------------------------
def main():
    set_seed(2025)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print(f"🚀 最终迁移攻击评估 ({config.CURRENT_DATASET})")
    print("=" * 60)

    # 1. 动态路径配置
    decoy_filename = f'baseline_ProGen_CIC-IDS2017.csv'
    CAMOUFLAGE_BOT_PATH = os.path.join(project_root, 'data', 'generated', decoy_filename)
    TEST_SET_PATH = os.path.join(project_root, 'data', 'splits', 'holdout_test_set.csv')
    SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')

    MODEL_PATHS = {
        "1D-CNN Hunter": os.path.join(project_root, 'models', 'cnn_hunter.pt'),
        "XGBoost Hunter": os.path.join(project_root, 'models', 'xgboost_hunter.pkl'),
        "KNN Hunter": os.path.join(project_root, 'models', 'knn_hunter.pkl'),
        "MLP Hunter": os.path.join(project_root, 'models', 'mlp_hunter.pt'),
    }

    # 2. 加载数据
    print(f"\n[步骤1] 正在加载数据...\n  -> 诱饵: {CAMOUFLAGE_BOT_PATH}\n  -> 测试集: {TEST_SET_PATH}")
    try:
        df_cam = pd.read_csv(CAMOUFLAGE_BOT_PATH)
        df_test = pd.read_csv(TEST_SET_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError as e:
        print(f"❌ 错误: 找不到文件 - {e}")
        return

    # 3. 数据预处理
    df_cam.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cam.dropna(subset=DEFENDER_SET, inplace=True)
    df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test.dropna(subset=DEFENDER_SET, inplace=True)

    # 兼容 Label 列名
    label_col = 'Label' if 'Label' in df_test.columns else 'label'

    # 🔥🔥🔥 核心修复: 标签标准化 (String -> Int) 🔥🔥🔥
    # 如果检测到标签是字符串(Object)，强制转换为 0/1
    if df_test[label_col].dtype == object:
        print(f"   -> ⚠️ 检测到字符串标签 {df_test[label_col].unique()}，正在标准化为 [0, 1]...")
        # 逻辑：'Benign' (忽略大小写) -> 0, 其他 -> 1
        df_test[label_col] = df_test[label_col].apply(lambda x: 0 if str(x).lower() == 'benign' else 1)
        print(f"   -> 标签标准化完成。Benign(0): {len(df_test[df_test[label_col]==0])}, Bot(1): {len(df_test[df_test[label_col]==1])}")

    # 🔥 IDS2018 战术窗口采样逻辑
    if config.CURRENT_DATASET == 'CSE-CIC-IDS2018':
        print("\n⚠️ 检测到大规模数据集 (IDS2018)，执行评估阶段的战术采样...")

        # 此时 df_test[label_col] 已经是 0/1 了，可以安全筛选
        df_bot_full = df_test[df_test[label_col] == 1]
        if len(df_bot_full) > 1000:
            df_bot_sample = df_bot_full.sample(n=1000, random_state=2025)
        else:
            df_bot_sample = df_bot_full

        # 采样背景流量
        df_benign_full = df_test[df_test[label_col] == 0]
        # 这里的 100000 对应生成阶段的诱饵数量，保持 1:1 注入比或 100:1 压制比
        sample_size = min(len(df_benign_full), 100000)
        df_benign_sample = df_benign_full.sample(n=sample_size, random_state=2025)

        df_test_eval = pd.concat([df_bot_sample, df_benign_sample])
        print(f"   -> 采样后测试集: {len(df_bot_sample)} Bot + {len(df_benign_sample)} Benign")
    else:
        # IDS2017 全量评估
        df_test_eval = df_test
        print(f"   -> 全量测试集: {len(df_test_eval)} 样本")

    print(f"使用 {len(DEFENDER_SET)} 维特征进行评估...")

    # 4. 特征缩放
    X_cam_scaled = scaler.transform(df_cam[DEFENDER_SET])

    X_benign_scaled = scaler.transform(df_test_eval[df_test_eval[label_col] == 0][DEFENDER_SET])
    X_bot_scaled = scaler.transform(df_test_eval[df_test_eval[label_col] == 1][DEFENDER_SET])
    y_bot_numpy = df_test_eval[df_test_eval[label_col] == 1][label_col].values

    # 5. 加载模型并评估
    print("\n[步骤2] 开始评估...")
    results_list = []

    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"⚠️ 跳过 {name}: 模型文件不存在 ({path})")
            continue

        try:
            threshold = MODEL_THRESHOLDS.get(name, 0.5)

            if name == "MLP Hunter":
                model = MLP_Classifier(feature_dim=len(DEFENDER_SET)).to(device)
                model.load_state_dict(torch.load(path, map_location=device))
                result = evaluate_hunter(name, model, X_cam_scaled, X_benign_scaled, X_bot_scaled, y_bot_numpy, device,
                                         threshold)

            elif name == "1D-CNN Hunter":
                model = CNN_Classifier(feature_dim=len(DEFENDER_SET)).to(device)
                model.load_state_dict(torch.load(path, map_location=device))
                result = evaluate_hunter(name, model, X_cam_scaled, X_benign_scaled, X_bot_scaled, y_bot_numpy, device,
                                         threshold)

            else:
                # Sklearn/XGB
                model = joblib.load(path)
                result = evaluate_hunter(name, model, X_cam_scaled, X_benign_scaled, X_bot_scaled, y_bot_numpy, device,
                                         threshold)

            results_list.append(result)
        except Exception as e:
            print(f"⚠️ 无法加载或评估 {name}: {e}")

    # 6. 汇总报告
    print("\n\n" + "=" * 100)
    print(f"--- 最终评估汇总报告 ({config.CURRENT_DATASET}) ---")
    print("=" * 100)
    if results_list:
        results_df = pd.DataFrame(results_list).set_index("Hunter")
        print(results_df.to_string(float_format="%.2f"))

        save_path = os.path.join(project_root, 'data', f'evaluation_results_{config.CURRENT_DATASET}.csv')
        results_df.to_csv(save_path)
        print(f"\n✅ 结果已保存至: {save_path}")
    else:
        print("无结果。")


if __name__ == "__main__":
    main()