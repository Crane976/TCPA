# models/train_xgboost_hunter.py (Final Corrected Version)
import pandas as pd
import numpy as np
import os
import sys
import joblib
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# --- 路径修正与模块导入 ---
# ==========================================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)


# ✅ 1. 导入新的特征集
from config import DEFENDER_SET, set_seed


# ==========================================================
# --- 中文显示配置 ---
# ==========================================================
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("已设置字体为 SimHei。")
except Exception:
    print("警告: 未找到SimHei字体，中文可能无法显示。")

# ==========================================================
# --- 1. 配置区 ---
# ==========================================================
# ✅ 核心输入: 使用严格分离的数据集
TRAIN_SET_PATH = os.path.join(project_root, 'data', 'splits', 'training_set.csv')
TEST_SET_PATH = os.path.join(project_root, 'data', 'splits', 'holdout_test_set.csv')
SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')

# --- 输出 ---
MODELS_DIR = os.path.join(project_root, 'models')
FIGURES_DIR = os.path.join(project_root, 'figures')
HUNTER_MODEL_PATH = os.path.join(MODELS_DIR, 'xgboost_hunter.pkl')


# ==========================================================
# --- 2. 主训练函数 ---
# ==========================================================
def main():
    set_seed(2025)  # ✅ 在main函数开头调用
    print("==========================================================")
    print("🚀 开始训练'均衡型猎手' (在严格分离的数据集上)...")
    print("==========================================================")

    # --- 1. 加载所有资产 ---
    print("正在加载训练集、留出测试集和全局Scaler...")
    try:
        df_train = pd.read_csv(TRAIN_SET_PATH)
        df_test = pd.read_csv(TEST_SET_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError as e:
        print(f"错误: 找不到核心文件 - {e}");
        return

    # --- 2. 准备特征和标签 ---
    # 从DataFrame中分离特征 (X) 和标签 (y)
    X_train_raw = df_train[DEFENDER_SET]
    y_train = df_train['label']
    X_test_raw = df_test[DEFENDER_SET]
    y_test = df_test['label']

    # ✅ 核心操作: 使用加载的Scaler分别转换训练和测试数据
    print("正在使用Scaler转换数据...")
    X_train_scaled = scaler.transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    print(f"训练集大小: {X_train_scaled.shape}, 测试集大小: {X_test_scaled.shape}")

    # --- 3. 使用GridSearchCV寻找最佳超参数 ---
    print("\n[步骤1] 正在通过GridSearchCV寻找最佳超参数...")
    hunter_model_base = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='logloss', use_label_encoder=False,
        n_estimators=100, n_jobs=-1, random_state=2025
    )
    param_grid = {
        'scale_pos_weight': [5, 10, 15, 20],
        'max_depth': [5, 6, 7],
        'learning_rate': [0.05, 0.1]
    }
    grid_search = GridSearchCV(estimator=hunter_model_base, param_grid=param_grid, scoring='f1', cv=3, verbose=2)

    # ✅ 在正确的、归一化后的训练数据上执行搜索
    grid_search.fit(X_train_scaled, y_train)

    print(f"\n搜索完成！ -> 最佳参数组合: {grid_search.best_params_}")

    # --- 4. 使用最佳参数训练最终模型 ---
    print("\n[步骤2] 正在使用找到的最佳参数训练最终的'猎手'模型...")
    best_params = grid_search.best_params_
    hunter_model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='logloss', use_label_encoder=False,
        n_estimators=200, n_jobs=-1, random_state=42, **best_params
    )
    hunter_model.fit(X_train_scaled, y_train)
    joblib.dump(hunter_model, HUNTER_MODEL_PATH)
    print(f"✅ 最终'猎手'模型已保存到: {HUNTER_MODEL_PATH}")

    # --- 5. 在从未见过的留出测试集上进行最终评估 ---
    print("\n--- '猎手'在【留出测试集】上的真实性能报告 ---")
    y_pred = hunter_model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Bot (1)'], digits=4))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Bot'], yticklabels=['Benign', 'Bot'])
    plt.title("'猎手'模型在留出测试集上的混淆矩阵")
    plt.xlabel('预测标签');
    plt.ylabel('真实标签')
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "hunter_holdout_test_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    print(f"✅ 混淆矩阵已保存到: {cm_path}")
    plt.show()


if __name__ == "__main__":
    main()