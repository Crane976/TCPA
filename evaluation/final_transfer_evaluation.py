# evaluation/final_transfer_evaluation.py (FINAL GRAND REVIEW VERSION)
import pandas as pd
import numpy as np
import os
import sys
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import f1_score  # 用于寻找阈值
import xgboost as xgb
from sklearn.model_selection import train_test_split

# ==========================================================
# --- Path Setup & Imports ---
# ==========================================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.append(project_root)

from config import DEFENDER_SET, set_seed
# ✅ 1. 导入所有五个模型的架构
from models.mlp_architecture import MLP_Classifier
from models.cnn_architecture import CNN_Classifier
from models.lstm_architecture import LSTM_Classifier
from models.transformer_architecture import Transformer_Classifier


# ==========================================================
# --- 1. Helper Function to Find Best Threshold ---
# ==========================================================
def find_best_threshold(model, X_val, y_val, device):
    """在验证集上为PyTorch模型寻找最佳决策阈值"""
    model.eval()
    with torch.no_grad():
        val_probs = model.predict(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu().numpy()

    best_threshold, best_f1 = 0.5, 0
    for threshold in np.arange(0.01, 1.0, 0.01):
        y_pred = (val_probs > threshold).astype(int)
        current_f1 = f1_score(y_val, y_pred, pos_label=1)
        if current_f1 > best_f1:
            best_f1, best_threshold = current_f1, threshold
    return best_threshold


# ==========================================================
# --- 2. Upgraded Evaluation Function ---
# ==========================================================
def evaluate_hunter(hunter_name, hunter_model, X_camouflage_scaled, X_benign_test, X_real_bot_test, y_real_bot_test,
                    device, threshold=0.5, batch_size=1024):  # ✅ 增加 batch_size 参数
    """
    评估单个猎手模型（已更新为支持分批次预测）。
    - threshold: 专为PyTorch模型设计的决策阈值
    - batch_size: 预测时使用的批次大小，防止CUDA错误
    """
    print("\n" + "=" * 50);
    print(f"--- 正在评估对抗: {hunter_name} ---");
    if not isinstance(hunter_model, xgb.XGBClassifier):
        print(f"    (使用最佳阈值: {threshold:.2f})")
    print("=" * 50)

    # 根据模型类型进行预测
    if isinstance(hunter_model, nn.Module):
        hunter_model.eval()
        all_preds = []

        # --- ✅ 分批次预测 ---
        def batch_predict(X_data):
            preds = []
            data_tensor = torch.tensor(X_data, dtype=torch.float32)
            dataset = torch.utils.data.TensorDataset(data_tensor)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            with torch.no_grad():
                for batch in loader:
                    batch_data = batch[0].to(device)
                    # 确保模型有 predict 方法，或者直接调用 forward
                    if hasattr(hunter_model, 'predict'):
                        probs = hunter_model.predict(batch_data)
                    else:
                        probs = hunter_model(batch_data)

                    pred_labels = (probs > threshold).int().cpu().numpy().flatten()
                    preds.extend(pred_labels)
            return np.array(preds)

        preds_cam = batch_predict(X_camouflage_scaled)
        preds_benign = batch_predict(X_benign_test)
        preds_bot = batch_predict(X_real_bot_test)

    else:  # 适用于XGBoost
        preds_cam = hunter_model.predict(X_camouflage_scaled)
        preds_benign = hunter_model.predict(X_benign_test)
        preds_bot = hunter_model.predict(X_real_bot_test)

    # 计算各项指标 (这部分逻辑不变)
    deceived_count = np.sum(preds_cam)
    deception_rate = deceived_count / len(X_camouflage_scaled) * 100

    base_fp = np.sum(preds_benign)
    base_tp = np.sum(preds_bot)
    base_fn = len(y_real_bot_test) - base_tp

    recall = base_tp / (base_tp + base_fn) * 100 if (base_tp + base_fn) > 0 else 0

    base_alerts = base_fp + base_tp
    mix_alerts = base_alerts + deceived_count

    dsr = (deceived_count / mix_alerts) * 100 if mix_alerts > 0 else 0
    base_precision = (base_tp / base_alerts) * 100 if base_alerts > 0 else 0
    hunter_precision_decayed = (base_tp / mix_alerts) * 100 if mix_alerts > 0 else 0

    print(f"  - 成功欺骗的伪装Bot数量: {deceived_count} / {len(X_camouflage_scaled)} ({deception_rate:.2f}%)")
    print(f"  - 真实Bot捕获率 (Recall): {recall:.2f}%")
    print(f"  - 误报数 (Benign -> Bot): {base_fp}")
    print("---------------------------------------------")
    print(f"  🎯 最终欺骗成功率 (DSR): {dsr:.2f}%")
    print(f"  📉 精确率从 {base_precision:.2f}% 衰减为: {hunter_precision_decayed:.2f}%")

    return {
        "Hunter": hunter_name,
        "Deception Rate (%)": deception_rate,
        "Recall (%)": recall,
        "Base Precision (%)": base_precision,
        "Decayed Precision (%)": hunter_precision_decayed,
        "DSR (%)": dsr
    }


# ==========================================================
# --- 3. Main Evaluation Orchestrator ---
# ==========================================================
def main():
    set_seed(2025)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 统一的路径配置 ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CAMOUFLAGE_BOT_PATH = os.path.join(project_root, 'data', 'generated',
                                       'final_camouflage_bot_3tier_lstm.csv')
    TRAIN_SET_PATH = os.path.join(project_root, 'data', 'splits', 'training_set.csv')
    TEST_SET_PATH = os.path.join(project_root, 'data', 'splits', 'holdout_test_set.csv')
    SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')

    MODEL_PATHS = {
        "XGBoost Hunter": os.path.join(project_root, 'models', 'xgboost_hunter.pkl'),
        "MLP Hunter": os.path.join(project_root, 'models', 'mlp_hunter.pt'),
        "1D-CNN Hunter": os.path.join(project_root, 'models', 'cnn_hunter.pt'),
        "LSTM Hunter": os.path.join(project_root, 'models', 'lstm_hunter.pt'),
        "Transformer Hunter": os.path.join(project_root, 'models', 'transformer_hunter.pt'),
    }

    print("=" * 50);
    print("🚀 最终迁移攻击评估 (大阅兵)...");
    print("=" * 50)

    # --- 1. 加载数据 ---
    print("\n[步骤1] 正在加载欺骗流量、测试集和Scaler...")
    try:
        df_cam = pd.read_csv(CAMOUFLAGE_BOT_PATH)
        df_train = pd.read_csv(TRAIN_SET_PATH)  # 需要训练集来划分出验证集
        df_test = pd.read_csv(TEST_SET_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError as e:
        print(f"错误: 找不到核心评估文件 - {e}");
        return

    feature_names = scaler.feature_names_in_
    X_cam_scaled = scaler.transform(df_cam[feature_names].values)

    # 准备测试数据
    df_benign_test = df_test[df_test['label'] == 0]
    df_bot_test = df_test[df_test['label'] == 1]
    X_benign_scaled = scaler.transform(df_benign_test[feature_names].values)
    X_bot_scaled = scaler.transform(df_bot_test[feature_names].values)
    y_bot_numpy = df_bot_test['label'].values

    # 准备验证数据 (用于寻找阈值)
    X_train_scaled = scaler.transform(df_train[feature_names].values)
    y_train = df_train['label'].values
    _, X_val, _, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=2025, stratify=y_train)
    print("✅ 数据加载和准备完毕。")

    # --- 2. 加载所有模型 ---
    print("\n[步骤2] 正在加载所有猎手模型...")
    hunters = {}
    try:
        # 加载XGBoost
        import xgboost as xgb
        hunters["XGBoost Hunter"] = joblib.load(MODEL_PATHS["XGBoost Hunter"])

        # 加载PyTorch模型
        model_defs = {
            "MLP Hunter": MLP_Classifier,
            "1D-CNN Hunter": CNN_Classifier,
            "LSTM Hunter": LSTM_Classifier,
            "Transformer Hunter": Transformer_Classifier
        }
        for name, model_class in model_defs.items():
            model = model_class(feature_dim=len(DEFENDER_SET)).to(device)
            model.load_state_dict(torch.load(MODEL_PATHS[name], map_location=device))
            model.eval()
            hunters[name] = model
        print("✅ 所有模型加载完毕。")
    except (FileNotFoundError, KeyError) as e:
        print(f"错误: 找不到模型文件或路径配置错误 - {e}");
        return

    # --- 3. 评估每个猎手并收集结果 ---
    print("\n[步骤3] 开始逐一评估猎手...")
    results_list = []
    for name, model in hunters.items():
        threshold = 0.5
        if isinstance(model, nn.Module):
            # 为每个NN模型动态寻找最佳阈值
            threshold = find_best_threshold(model, X_val, y_val, device)

        result = evaluate_hunter(name, model, X_cam_scaled, X_benign_scaled, X_bot_scaled, y_bot_numpy, device,
                                 threshold)
        results_list.append(result)

    # --- 4. 汇总并展示最终结果 ---
    print("\n\n" + "=" * 70)
    print("--- 最终迁移攻击评估汇总报告 ---")
    print("=" * 70)

    results_df = pd.DataFrame(results_list)
    results_df = results_df.set_index("Hunter")
    print(results_df.to_string(float_format="%.2f"))

    print("\n" + "=" * 70);
    print("--- 评估完成 ---");
    print("=" * 70)


if __name__ == "__main__":
    main()