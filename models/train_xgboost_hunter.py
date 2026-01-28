# models/train_xgboost_hunter.py (UNIVERSAL GPU VERSION)
# 适配: CIC-IDS2017 & CSE-CIC-IDS2018
# 亮点: 动态采样 + 开启GPU加速
import pandas as pd
import numpy as np
import os
import sys
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

# --- 路径设置 ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.append(project_root)

from config import DEFENDER_SET, set_seed

# --- 配置区 ---
TRAIN_SET_PATH = os.path.join(project_root, 'data', 'splits', 'training_set.csv')
TEST_SET_PATH = os.path.join(project_root, 'data', 'splits', 'holdout_test_set.csv')
SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')
MODELS_DIR = os.path.join(project_root, 'models')
HUNTER_MODEL_PATH = os.path.join(MODELS_DIR, 'xgboost_hunter.pkl')


def main():
    set_seed(2025)
    print("=" * 60)
    print("🚀 训练 XGBoost Hunter (通用适配版 - GPU加速)...")
    print("=" * 60)

    # --- 1. 加载与清洗 ---
    print("正在加载数据...")
    if not os.path.exists(TRAIN_SET_PATH):
        print(f"❌ 错误: 找不到训练集 {TRAIN_SET_PATH}")
        return

    df_train = pd.read_csv(TRAIN_SET_PATH)
    df_test = pd.read_csv(TEST_SET_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = scaler.feature_names_in_

    # 清洗 Inf/NaN
    for df in [df_train, df_test]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=[col for col in DEFENDER_SET if col in df.columns], inplace=True)

    # --- 2. 划分与动态采样 ---
    print("\n[步骤1] 构建训练子集 (Smart Sampling)...")
    X_full = df_train[feature_names]
    y_full = df_train['label']

    # 划分验证集 (保持真实分布，用于阈值寻优)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_full, y_full, test_size=0.2, random_state=2025, stratify=y_full
    )

    df_pool = pd.concat([X_train_split, y_train_split], axis=1)
    df_bot = df_pool[df_pool['label'] == 1]
    df_benign = df_pool[df_pool['label'] == 0]

    n_bot = len(df_bot)
    n_benign_total = len(df_benign)

    # 🔥 [策略升级] 动态采样比例
    # 对于树模型，不需要严格 1:1。给它更多良性样本(如 1:5)能减少误报。
    # 逻辑: 试图取 Bot 的 5 倍良性，但绝不超过良性总数。
    TARGET_RATIO = 5
    n_benign_sample = min(n_benign_total, n_bot * TARGET_RATIO)

    df_benign_sampled = df_benign.sample(n=n_benign_sample, random_state=2025)
    df_train_balanced = pd.concat([df_bot, df_benign_sampled])

    print(f"   -> Bot样本: {n_bot}")
    print(f"   -> Benign样本: {n_benign_sample} (Ratio 1:{n_benign_sample / n_bot:.1f})")
    print(f"   -> 总训练量: {len(df_train_balanced)}")

    # --- 3. 缩放 ---
    print("\n[步骤2] 数据标准化...")
    X_train_final = scaler.transform(df_train_balanced[feature_names])
    y_train_final = df_train_balanced['label']

    X_val_scaled = scaler.transform(X_val_split)  # 验证集保持真实比例
    X_test_scaled = scaler.transform(df_test[feature_names])
    y_test = df_test['label'].values

    # --- 4. 参数搜索 (GPU 加速) ---
    print("\n[步骤3] 正在搜索最佳参数 (RandomizedSearch)...")

    # 🔥 [关键] 开启 GPU 加速
    # tree_method='hist', device='cuda' 是 XGBoost 新版启用 GPU 的标准写法
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',  # 使用直方图算法 (最快)
        device='cuda',  # 使用 GPU (RTX 4060)
        use_label_encoder=False,
        random_state=2025
    )

    param_dist = {
        'n_estimators': [200, 400, 600],
        'max_depth': [6, 10, 14],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        # 'scale_pos_weight': [1, 3] # 如果不做采样，可以用这个参数平衡，但我们做了物理采样
    }

    # 注意: n_jobs 在 GPU 模式下通常设为 1 或 -1 均可，主要靠 GPU 算
    search = RandomizedSearchCV(
        xgb_clf, param_dist, n_iter=8, scoring='f1', cv=3, verbose=1, n_jobs=1, random_state=2025
    )

    try:
        search.fit(X_train_final, y_train_final)
    except Exception as e:
        print(f"⚠️ GPU训练失败 (可能是显存不足或版本问题): {e}")
        print("   -> 切换回 CPU 模式继续训练...")
        xgb_clf.set_params(device='cpu')
        search.fit(X_train_final, y_train_final)

    best_model = search.best_estimator_
    print(f"   -> 最佳参数: {search.best_params_}")

    # --- 5. 阈值寻优 ---
    print("\n[步骤4] 寻找最佳决策阈值...")
    # 使用 GPU 预测加速
    val_probs = best_model.predict_proba(X_val_scaled)[:, 1]

    best_thr, best_f1 = 0.5, 0
    thresholds = np.arange(0.1, 0.96, 0.05)

    for thr in thresholds:
        y_pred = (val_probs >= thr).astype(int)
        f1 = f1_score(y_val_split, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    print(f"✅ 最佳阈值: {best_thr:.2f} (Val F1: {best_f1:.4f})")

    # --- 6. 保存与最终评估 ---
    joblib.dump(best_model, HUNTER_MODEL_PATH)
    print(f"💾 模型已保存至: {HUNTER_MODEL_PATH}")

    print(f"\n--- 'XGBoost Hunter' 最终评估 (测试集) ---")
    test_probs = best_model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = (test_probs >= best_thr).astype(int)

    print(classification_report(y_test, y_test_pred, target_names=['Benign (0)', 'Bot (1)'], digits=4))


if __name__ == "__main__":
    main()