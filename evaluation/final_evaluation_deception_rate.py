# evaluation/final_evaluation_deception_rate.py (FINAL 3-TIER COMPATIBLE VERSION)
import pandas as pd
import numpy as np
import os
import sys
import joblib

# ==========================================================
# --- 路径修正与模块导入 ---
# ==========================================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.append(project_root)

# ✅ 1. 导入新的特征集
from config import DEFENDER_SET

# ==========================================================
# --- 1. 配置区 ---
# ==========================================================
TEST_SET_PATH = os.path.join(project_root, 'data', 'splits', 'holdout_test_set.csv')
CAMOUFLAGE_BOT_PATH = os.path.join(project_root, 'data', 'generated', 'final_camouflage_bot.csv')
SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')
HUNTER_MODEL_PATH = os.path.join(project_root, 'models', 'xgboost_hunter.pkl')  # 主要评估XGBoost


# ==========================================================
# --- 2. 主评估函数 ---
# ==========================================================
def main():
    print("=" * 60);
    print("🚀 最终决战深度评估 (ACMF, DSR, ...)");
    print("=" * 60)

    # --- 1. 加载所有资产 ---
    try:
        hunter_model = joblib.load(HUNTER_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        df_test_full = pd.read_csv(TEST_SET_PATH)
        df_camouflage_bot = pd.read_csv(CAMOUFLAGE_BOT_PATH)
    except FileNotFoundError as e:
        print(f"错误: 找不到核心文件 - {e}");
        return

    # ✅ 2. 使用DEFENDER_SET进行数据准备
    feature_names = scaler.feature_names_in_  # 确保与scaler训练时一致

    # --- 2. 基准环境 (无欺骗) ---
    df_benign_test = df_test_full[df_test_full['label'] == 0]
    df_real_bot_test = df_test_full[df_test_full['label'] == 1]

    X_benign_scaled = scaler.transform(df_benign_test[feature_names])
    X_real_bot_scaled = scaler.transform(df_real_bot_test[feature_names])

    preds_benign_base = hunter_model.predict(X_benign_scaled)
    preds_real_bot_base = hunter_model.predict(X_real_bot_scaled)

    base_fp = np.sum(preds_benign_base);
    base_tp = np.sum(preds_real_bot_base)
    base_alerts = base_fp + base_tp
    base_precision = (base_tp / base_alerts) * 100 if base_alerts > 0 else 0
    base_cost_per_target = base_alerts / base_tp if base_tp > 0 else float('inf')

    # --- 3. 欺骗环境 ---
    X_camouflage_scaled = scaler.transform(df_camouflage_bot[feature_names])
    deceived_count = np.sum(hunter_model.predict(X_camouflage_scaled))

    # --- 4. 计算指标 ---
    mix_alerts = base_alerts + deceived_count
    dsr = (deceived_count / mix_alerts) * 100 if mix_alerts > 0 else 0
    hunter_precision_decayed = (base_tp / mix_alerts) * 100 if mix_alerts > 0 else 0
    mix_cost_per_target = mix_alerts / mix_tp if base_tp > 0 else float('inf')
    acmf = mix_cost_per_target / base_cost_per_target if base_cost_per_target not in [0, float('inf')] else float('inf')

    # --- 5. 打印战报 ---
    print("\n" + "=" * 45);
    print("--- 最终评估战报 ---");
    print("=" * 45)
    print(
        f"战场环境: {len(df_benign_test)} 良性 vs {len(df_real_bot_test)} 真实Bot | 注入 {len(df_camouflage_bot)} 伪装Bot")
    print("---------------------------------------------")
    print(f"基准性能: 误报(FP)={base_fp}, 捕获(TP)={base_tp}, 召回率={(base_tp / len(df_real_bot_test)) * 100:.2f}%")
    print("---------------------------------------------")
    print(
        f"战果: 成功欺骗 {deceived_count} / {len(df_camouflage_bot)} ({deceived_count / len(df_camouflage_bot) * 100:.2f}%)")
    print("=" * 45);
    print("         🚀 核心评估指标 🚀");
    print("=" * 45)
    print(f"🎯 欺骗成功率 (DSR): {dsr:.2f}% (告警中有{dsr:.2f}%是诱饵)")
    print(f"📉 精确率衰减: 从 {base_precision:.2f}% 降至 {hunter_precision_decayed:.2f}%")
    print(f"📈 攻击成本提升 (ACMF): {acmf:.2f} 倍")
    print("=" * 45)


if __name__ == "__main__":
    main()