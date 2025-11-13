import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- Path Setup ---
# 确保可以从项目根目录导入config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入您的特征集配置
try:
    from config import UNIFIED_FEATURE_SET
except ImportError:
    print("错误: 无法从 'config.py' 导入 UNIFIED_FEATURE_SET。")
    print("请确保该文件存在并且路径正确。")
    sys.exit(1)

# --- 配置区 ---
# 请确保这个路径指向您训练好的XGBoost模型
XGBOOST_MODEL_PATH = os.path.join(project_root, 'models', 'xgboost_hunter.pkl')

def analyze_feature_importance():
    """
    加载训练好的XGBoost模型并分析其特征重要性。
    """
    print("==========================================================")
    print("🚀 开始分析 XGBoost Hunter 的特征重要性...")
    print("==========================================================")

    # --- 1. 加载模型 ---
    try:
        xgb_hunter = joblib.load(XGBOOST_MODEL_PATH)
        print(f"✅ 成功加载模型: {XGBOOST_MODEL_PATH}")
    except FileNotFoundError:
        print(f"❌ 错误: 找不到XGBoost模型文件 at '{XGBOOST_MODEL_PATH}'")
        return

    # --- 2. 提取特征重要性 ---
    # .feature_importances_ 属性存储了每个特征的重要性得分
    importances = xgb_hunter.feature_importances_

    # --- 3. 创建一个DataFrame方便排序和绘图 ---
    feature_importance_df = pd.DataFrame({
        'Feature': UNIFIED_FEATURE_SET,
        'Importance': importances
    })

    # --- 4. 按重要性降序排序 ---
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    print("\n--- Top 10 最重要的特征 ---")
    print(feature_importance_df.head(10).to_string(index=False))
    print("-----------------------------\n")

    # --- 5. 可视化 ---
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('XGBoost Hunter - Feature Importance', fontsize=16)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout() # 自动调整布局

    # 保存图像
    output_path = os.path.join(project_root, 'analysis', 'xgboost_feature_importance.png')
    plt.savefig(output_path)
    print(f"✅ 特征重要性图已保存到: {output_path}")

    plt.show()

if __name__ == "__main__":
    analyze_feature_importance()