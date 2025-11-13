# analysis/build_feature_sets.py (Fixed Version 3 - Corrected Cleaning Order)
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# ==========================================================
# --- 1. 配置区 ---
# ==========================================================
BENIGN_IN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'filtered',
                         'benign_traffic.csv')
BOT_IN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'filtered',
                      'bot_traffic_target.csv')

# --- 核心筛选参数 ---
LOW_VARIANCE_THRESHOLD = 0.001
HIGH_MISSING_RATE_THRESHOLD = 0.1
TIME_ANCHOR_FEATURES = [
    'Flow Duration', 'Flow IAT Mean', 'Fwd IAT Mean', 'Bwd IAT Mean'
]
CORRELATION_THRESHOLD = 0.1
COLLINEARITY_THRESHOLD = 0.95
ALLOWED_PREFIXES = ('Flow', 'Fwd IAT', 'Bwd IAT', 'Idle', 'Active')


# ==========================================================
# --- 2. 主分析函数 ---
# ==========================================================
def main():
    print("==========================================================")
    print("🚀 开始构建三层特征体系 (候选集 -> 统一集 -> 指纹集)...")
    print("==========================================================")

    # --- 加载并合并数据 ---
    print("正在加载完整的filtered数据集...")
    df_benign = pd.read_csv(BENIGN_IN, low_memory=False)
    df_bot = pd.read_csv(BOT_IN, low_memory=False)
    df_full = pd.concat([df_benign, df_bot], ignore_index=True)
    df_full.columns = df_full.columns.str.strip()
    df_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"数据集加载完毕，共 {len(df_full)} 条样本，{len(df_full.columns)} 个原始特征。")

    # --- 步骤一: 构建“候选特征集” (数据质量筛选) ---
    print("\n--- 步骤一: 构建'候选特征集' (CANDIDATE_FEATURE_SET) ---")
    numeric_features = df_full.select_dtypes(include=np.number).columns.tolist()
    missing_rates = df_full[numeric_features].isnull().sum() / len(df_full)
    high_missing_features = missing_rates[missing_rates > HIGH_MISSING_RATE_THRESHOLD].index.tolist()
    candidate_features_step1 = [f for f in numeric_features if f not in high_missing_features]

    # ✅ 核心修正: 只为方差计算创建一个临时填充的DataFrame
    df_for_variance = df_full[candidate_features_step1].copy()
    df_for_variance.fillna(df_for_variance.median(), inplace=True)

    variances = df_for_variance.var()
    low_variance_features = variances[variances < LOW_VARIANCE_THRESHOLD].index.tolist()
    candidate_feature_set = [f for f in candidate_features_step1 if f not in low_variance_features]
    print(f"✅ '候选特征集' 构建完成，共 {len(candidate_feature_set)} 个特征。")

    # ==========================================================
    # --- 步骤二: 构建“统一特征集” (相关性与共线性筛选) ---
    # ==========================================================
    print("\n--- 步骤二: 构建'统一特征集' (UNIFIED_FEATURE_SET) ---")

    # ✅ 核心修正: 使用原始的、带有NaN的候选集数据进行下一步
    df_for_corr_raw = df_full[candidate_feature_set].copy()

    print("  - 正在对数据进行临时归一化以进行相关性分析...")
    temp_scaler = MinMaxScaler()
    df_for_corr_scaled = pd.DataFrame(temp_scaler.fit_transform(df_for_corr_raw), columns=candidate_feature_set)

    # ✅ 核心修正: 在计算相关性时，让Pandas自动处理成对的缺失值
    # Pandas的 .corr() 默认会使用 'pairwise' 方法，只计算每对特征共有的非缺失值
    corr_matrix = df_for_corr_scaled.corr(method='pearson')

    # 检查锚点是否存在于corr_matrix中
    valid_anchors = [anchor for anchor in TIME_ANCHOR_FEATURES if anchor in corr_matrix.columns]
    if not valid_anchors:
        print("错误: 没有任何锚点特征存在于最终的候选集中，无法计算相关性。");
        return

    anchor_correlations = corr_matrix.loc[valid_anchors]
    avg_corr = anchor_correlations.abs().mean(axis=0)  # axis=0, mean across rows for each column
    highly_correlated_features = avg_corr[avg_corr > CORRELATION_THRESHOLD].index.tolist()
    print(f"  - 找到 {len(highly_correlated_features)} 个与时间锚点高度相关的特征。")

    corr_matrix_subset = corr_matrix.loc[highly_correlated_features, highly_correlated_features]
    to_drop = set()
    for i in range(len(corr_matrix_subset.columns)):
        for j in range(i):
            if abs(corr_matrix_subset.iloc[i, j]) > COLLINEARITY_THRESHOLD:
                colname_i = corr_matrix_subset.columns[i]
                colname_j = corr_matrix_subset.columns[j]
                if colname_i > colname_j:
                    to_drop.add(colname_i)
                else:
                    to_drop.add(colname_j)

    unified_feature_set = sorted([f for f in highly_correlated_features if f not in to_drop])
    to_drop_list = sorted(list(to_drop))
    print(f"  - 排除 {len(to_drop_list)} 个高度共线性特征: {to_drop_list}")
    print(f"\n✅ '统一特征集' 构建完成，共 {len(unified_feature_set)} 个特征。")

    # --- 步骤三 & 四 ... (后续代码不变) ...
    print("\n--- 步骤三: 构建'时间指纹全景' (TIME_FINGERPRINT_SET) ---")
    time_fingerprint_set = []
    excluded_by_prefix = []
    for feature in unified_feature_set:
        if feature.startswith(ALLOWED_PREFIXES):
            time_fingerprint_set.append(feature)
        else:
            excluded_by_prefix.append(feature)
    for anchor in valid_anchors:
        if anchor in unified_feature_set and anchor not in time_fingerprint_set:
            time_fingerprint_set.append(anchor)
    time_fingerprint_set = sorted(list(set(time_fingerprint_set)))
    print(f"  - 从'统一集'中根据前缀排除了 {len(excluded_by_prefix)} 个特征: {excluded_by_prefix}")
    print(f"\n✅ '时间指紋全景' 构建完成，共 {len(time_fingerprint_set)} 个特征。")
    print("\n==========================================================")
    print("               >>> 最终特征集定义 <<<")
    print("==========================================================")
    print("\n# --- 请将此部分复制并更新到您的 config.py 文件中 --- #\n")
    print("# 最终的、科学构建的统一特征集")
    print("UNIFIED_FEATURE_SET = [")
    if unified_feature_set:
        for i in range(0, len(unified_feature_set), 4):
            print(f"    {str(unified_feature_set[i:i + 4]).strip('[]')},")
    print("]\n")
    print("# 最终的、可在问题空间操作的时间指纹全景")
    print("TIME_FINGERPRINT_SET = [")
    if time_fingerprint_set:
        for i in range(0, len(time_fingerprint_set), 4):
            print(f"    {str(time_fingerprint_set[i:i + 4]).strip('[]')},")
    print("]\n")
    print("# --- config.py 更新内容结束 --- #")
    print("==========================================================")


if __name__ == "__main__":
    main()