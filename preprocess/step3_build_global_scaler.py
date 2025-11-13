# preprocess/step3_build_global_scaler.py (FINAL 3-TIER COMPATIBLE VERSION)
import pandas as pd
import numpy as np
import os
import joblib
import sys
from sklearn.preprocessing import MinMaxScaler

# ==========================================================
# --- 路径修正与模块导入 ---
# ==========================================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# ✅✅✅ 核心修改: 导入新的、最广阔的特征集 ✅✅✅
from config import DEFENDER_SET

# ==========================================================
# --- 1. 配置区 ---
# ==========================================================
# 输入: 只使用严格分离的训练集来训练Scaler
INPUT_PATH = os.path.join(project_root, 'data', 'splits', 'training_set.csv')

# 输出: 我们的全局“度量衡”
SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')


# ==========================================================
# --- 2. 主函数 ---
# ==========================================================
def main():
    print("==========================================================")
    print("🚀 STEP 3 (Final): 构建全局Scaler (基于最广阔的DEFENDER_SET)")
    print("==========================================================")

    try:
        print(f"正在从严格分离的训练集加载数据: {INPUT_PATH}...")
        df_train = pd.read_csv(INPUT_PATH, low_memory=False)
    except FileNotFoundError as e:
        print(f"错误: 找不到训练集文件 - {e}")
        return

    # --- 1. 数据验证与清理 ---
    # 我们只关心特征列，标签列(label)不参与Scaler的训练
    print(f"Scaler将基于 {len(DEFENDER_SET)} 个防御者视野内的特征进行训练。")

    # ✅✅✅ 核心修改: 使用DEFENDER_SET来选择特征 ✅✅✅
    df_features = df_train[DEFENDER_SET].copy()

    # 清理无穷大和NaN值
    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 检查是否有整列都是NaN的情况，这可能在数据子集中发生
    df_features.dropna(axis=1, how='all', inplace=True)
    df_features.dropna(axis=0, how='any', inplace=True)  # 丢弃任何含有NaN的行

    print(f"数据清理后，用于训练Scaler的样本总数: {len(df_features)}")
    if len(df_features) == 0:
        print("错误：数据清理后没有剩余样本，请检查您的 training_set.csv 和 DEFENDER_SET 中的特征。")
        return

    # --- 2. 训练全局Scaler ---
    print("\n正在训练全局Scaler...")
    scaler = MinMaxScaler()

    # 核心操作: 在训练集的DEFENDER_SET上 .fit()
    scaler.fit(df_features)

    # --- 3. 保存结果 ---
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ 全局Scaler已保存到: {SCALER_PATH}")
    print("\n此脚本任务完成。后续所有步骤都将加载此Scaler来转换数据。")


if __name__ == "__main__":
    main()