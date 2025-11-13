# evaluation/final_transfer_evaluation.py (FINAL 3-TIER COMPATIBLE VERSION)
import pandas as pd
import numpy as np
import os
import sys
import joblib
import torch
import torch.nn as nn

# ==========================================================
# --- Path Setup & Imports ---
# ==========================================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.append(project_root)

# ✅ 1. 导入新的特征集
from config import DEFENDER_SET, set_seed
from models.mlp_architecture import MLP_Classifier


# ==========================================================
# --- 1. Helper Evaluation Function ---
# ==========================================================
def evaluate_hunter(hunter_name, hunter_model, X_camouflage_scaled, X_benign_test, X_real_bot_test, y_real_bot_test,
                    device):
    # ... (这个函数内部逻辑是通用的，不需要修改) ...
    print("\n" + "=" * 50);
    print(f"--- 正在评估对抗: {hunter_name} ---");
    print("=" * 50)
    if isinstance(hunter_model, nn.Module):
        with torch.no_grad():
            preds_cam = (hunter_model.predict(
                torch.tensor(X_camouflage_scaled, dtype=torch.float32).to(device)) > 0.5).int().cpu().numpy().flatten()
            preds_benign = (hunter_model.predict(
                torch.tensor(X_benign_test, dtype=torch.float32).to(device)) > 0.5).int().cpu().numpy().flatten()
            preds_bot = (hunter_model.predict(
                torch.tensor(X_real_bot_test, dtype=torch.float32).to(device)) > 0.5).int().cpu().numpy().flatten()
    elif "IsolationForest" in str(type(hunter_model)):
        preds_cam = np.array([1 if x == -1 else 0 for x in hunter_model.predict(X_camouflage_scaled)])
        preds_benign = np.array([1 if x == -1 else 0 for x in hunter_model.predict(X_benign_test)])
        preds_bot = np.array([1 if x == -1 else 0 for x in hunter_model.predict(X_real_bot_test)])
    else:
        preds_cam = hunter_model.predict(X_camouflage_scaled)
        preds_benign = hunter_model.predict(X_benign_test)
        preds_bot = hunter_model.predict(X_real_bot_test)
    deceived_count = np.sum(preds_cam);
    base_fp = np.sum(preds_benign);
    base_tp = np.sum(preds_bot)
    base_alerts = base_fp + base_tp;
    mix_alerts = base_alerts + deceived_count
    dsr = (deceived_count / mix_alerts) * 100 if mix_alerts > 0 else 0
    base_precision = (base_tp / base_alerts) * 100 if base_alerts > 0 else 0
    hunter_precision_decayed = (base_tp / mix_alerts) * 100 if mix_alerts > 0 else 0
    print(
        f"  - 成功欺骗的伪装Bot数量: {deceived_count} / {len(X_camouflage_scaled)} ({deceived_count / len(X_camouflage_scaled) * 100:.2f}%)")
    print(f"  - 真实Bot捕获率 (Recall): {base_tp / len(y_real_bot_test) * 100:.2f}%")
    print(f"  - 误报数 (Benign -> Bot): {base_fp}")
    print("---------------------------------------------")
    print(f"  🎯 最终欺骗成功率 (DSR): {dsr:.2f}%")
    print(f"  📉 精确率从 {base_precision:.2f}% 衰减为: {hunter_precision_decayed:.2f}%")


# ==========================================================
# --- 2. Main Evaluation Orchestrator ---
# ==========================================================
def main():
    set_seed(2025)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Paths ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CAMOUFLAGE_BOT_PATH = os.path.join(project_root, 'data', 'generated', 'final_camouflage_bot.csv')
    TEST_SET_PATH = os.path.join(project_root, 'data', 'splits', 'holdout_test_set.csv')
    SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')
    XGB_HUNTER_PATH = os.path.join(project_root, 'models', 'xgboost_hunter.pkl')
    MLP_HUNTER_PATH = os.path.join(project_root, 'models', 'mlp_hunter.pt')

    print("=" * 50);
    print("🚀 最终迁移攻击评估 (大阅兵)...");
    print("=" * 50)

    # --- 1. Load Data ---
    try:
        df_cam = pd.read_csv(CAMOUFLAGE_BOT_PATH)
        df_test = pd.read_csv(TEST_SET_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError as e:
        print(f"错误: 找不到核心评估文件 - {e}");
        return

    # ✅ 2. 使用DEFENDER_SET进行数据准备
    feature_names = scaler.feature_names_in_  # 确保与scaler训练时一致
    X_cam_scaled = scaler.transform(df_cam[feature_names].values)

    df_benign_test = df_test[df_test['label'] == 0]
    df_bot_test = df_test[df_test['label'] == 1]
    X_benign_scaled = scaler.transform(df_benign_test[feature_names].values)
    X_bot_scaled = scaler.transform(df_bot_test[feature_names].values)
    y_bot_numpy = df_bot_test['label'].values

    # --- 2. Load Models ---
    try:
        xgb_hunter = joblib.load(XGB_HUNTER_PATH)
        mlp_hunter = MLP_Classifier(feature_dim=len(DEFENDER_SET)).to(device)
        mlp_hunter.load_state_dict(torch.load(MLP_HUNTER_PATH, map_location=device))
        mlp_hunter.eval()
    except FileNotFoundError as e:
        print(f"错误: 找不到猎手模型文件 - {e}");
        return

    # --- 3. Evaluate each hunter ---
    hunters = {"XGBoost Hunter": xgb_hunter, "MLP Hunter": mlp_hunter}
    for name, model in hunters.items():
        evaluate_hunter(name, model, X_cam_scaled, X_benign_scaled, X_bot_scaled, y_bot_numpy, device)

    print("\n" + "=" * 50);
    print("--- 评估完成 ---");
    print("=" * 50)


if __name__ == "__main__":
    main()