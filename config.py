# config.py (THE FINAL ROBUST HYBRID VERSION - LOGICALLY CONSISTENT & LOG-SCALED)
import pandas as pd
import torch
import numpy as np
import random
import os
import sys
from sklearn.preprocessing import MinMaxScaler  # ✅ 新增导入

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.append(project_root)

# =================================================================
# --- 🌍 全局数据集配置开关 (Global Dataset Switch) ---
# =================================================================
CURRENT_DATASET = 'CSE-CIC-IDS2018'

# =================================================================
# --- 📁 路径配置 (Path Configuration) ---
# =================================================================
# 1. 获取当前文件(config.py)所在的目录，即项目根目录 D:\DTCA
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 2. 基础数据目录 D:\DTCA\data
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# 3. 模型保存目录 D:\DTCA\models
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'models')

# 4. Scaler 路径 D:\DTCA\models\global_scaler.pkl
SCALER_PATH = os.path.join(MODEL_SAVE_DIR, 'global_scaler.pkl')

# 5. 数据集切分目录 (存放 training_set.csv 和 holdout_test_set.csv)
SPLITS_DIR = os.path.join(BASE_DATA_DIR, 'splits')

# --- 根据数据集选择子目录 ---
if CURRENT_DATASET == 'CIC-IDS2017':
    RAW_CSV_NAME = 'Friday-WorkingHours-Morning.pcap_ISCX.csv'
    MALICIOUS_LABEL = 'Bot'
    OUTPUT_SUBDIR = 'cic_ids_2017'

elif CURRENT_DATASET == 'CSE-CIC-IDS2018':
    RAW_CSV_NAME = 'Friday-02-03-2018_TrafficForML_CICFlowMeter.csv'
    MALICIOUS_LABEL = 'Bot'
    OUTPUT_SUBDIR = 'cse_cic_ids_2018'
else:
    raise ValueError(f"未知的数据集: {CURRENT_DATASET}")

# --- 自动生成完整路径 ---
RAW_DATA_PATH = os.path.join(BASE_DATA_DIR, RAW_CSV_NAME)
PROCESSED_DIR = os.path.join(BASE_DATA_DIR, OUTPUT_SUBDIR, 'filtered')

# 打印调试信息，确保路径正确
print(f"🔄 当前工作数据集: {CURRENT_DATASET}")
print(f"📂 原始文件路径: {RAW_DATA_PATH}")
print(f"📂 输出目录: {PROCESSED_DIR}")
print(f"📂 项目根目录: {PROJECT_ROOT}")
print(f"📂 模型目录: {MODEL_SAVE_DIR}")
print(f"📂 Scaler路径: {SCALER_PATH}")
print(f"🎯 目标恶意标签: {MALICIOUS_LABEL}")

# config.py 中的 COLUMN_MAPPING 部分

if CURRENT_DATASET == 'CIC-IDS2017':
    COLUMN_MAPPING = {}

elif CURRENT_DATASET == 'CSE-CIC-IDS2018':
    RAW_CSV_NAME = 'Friday-02-03-2018_TrafficForML_CICFlowMeter.csv'
    MALICIOUS_LABEL = 'Bot'
    OUTPUT_SUBDIR = 'cse_cic_ids_2018'

    # 🔥 [完整版] 特征列名映射: 2018 (缩写) -> 2017 (全称/代码标准)
    # 基于你提供的原始CSV列名对比生成
    COLUMN_MAPPING = {
        # --- 目标端口 & 基础信息 ---
        'Dst Port': 'Destination Port',
        # 'Protocol': 'Protocol', # 2017列表中未提供，保留原名即可，反正DEFENDER_SET不用
        # 'Timestamp': 'Timestamp', # 同上

        # --- 包数量与长度 (CRITICAL) ---
        'Tot Fwd Pkts': 'Total Fwd Packets',
        'Tot Bwd Pkts': 'Total Backward Packets',
        'TotLen Fwd Pkts': 'Total Length of Fwd Packets',
        'TotLen Bwd Pkts': 'Total Length of Bwd Packets',

        # --- 包长统计 (CRITICAL) ---
        'Fwd Pkt Len Max': 'Fwd Packet Length Max',
        'Fwd Pkt Len Min': 'Fwd Packet Length Min',
        'Fwd Pkt Len Mean': 'Fwd Packet Length Mean',
        'Fwd Pkt Len Std': 'Fwd Packet Length Std',
        'Bwd Pkt Len Max': 'Bwd Packet Length Max',
        'Bwd Pkt Len Min': 'Bwd Packet Length Min',
        'Bwd Pkt Len Mean': 'Bwd Packet Length Mean',
        'Bwd Pkt Len Std': 'Bwd Packet Length Std',

        # --- 流速率 (CRITICAL) ---
        'Flow Byts/s': 'Flow Bytes/s',  # 注意 2018 拼写是 Byts
        'Flow Pkts/s': 'Flow Packets/s',

        # --- 流时间间隔 IAT (CRITICAL) ---
        # Flow IAT Mean/Std/Max/Min 名字一样，不用映射
        'Fwd IAT Tot': 'Fwd IAT Total',
        # Fwd IAT Mean/Std/Max/Min 名字一样
        'Bwd IAT Tot': 'Bwd IAT Total',
        # Bwd IAT Mean/Std/Max/Min 名字一样

        # --- 标志位 Flags ---
        'Fwd PSH Flags': 'Fwd PSH Flags',  # 一样
        'Bwd PSH Flags': 'Bwd PSH Flags',  # 一样
        'Fwd URG Flags': 'Fwd URG Flags',  # 一样
        'Bwd URG Flags': 'Bwd URG Flags',  # 一样
        'FIN Flag Cnt': 'FIN Flag Count',
        'SYN Flag Cnt': 'SYN Flag Count',
        'RST Flag Cnt': 'RST Flag Count',
        'PSH Flag Cnt': 'PSH Flag Count',
        'ACK Flag Cnt': 'ACK Flag Count',
        'URG Flag Cnt': 'URG Flag Count',
        'ECE Flag Cnt': 'ECE Flag Count',
        # CWE Flag Count 名字一样

        # --- 头部长度 ---
        'Fwd Header Len': 'Fwd Header Length',
        'Bwd Header Len': 'Bwd Header Length',

        # --- 速率与包长综合 ---
        'Fwd Pkts/s': 'Fwd Packets/s',
        'Bwd Pkts/s': 'Bwd Packets/s',
        'Pkt Len Min': 'Min Packet Length',  # 注意词序变化
        'Pkt Len Max': 'Max Packet Length',  # 注意词序变化
        'Pkt Len Mean': 'Packet Length Mean',
        'Pkt Len Std': 'Packet Length Std',
        'Pkt Len Var': 'Packet Length Variance',
        'Pkt Size Avg': 'Average Packet Size',  # 注意词序变化

        # --- 片段与子流 ---
        'Fwd Seg Size Avg': 'Avg Fwd Segment Size',  # 注意词序变化
        'Bwd Seg Size Avg': 'Avg Bwd Segment Size',  # 注意词序变化
        'Fwd Byts/b Avg': 'Fwd Avg Bytes/Bulk',
        'Fwd Pkts/b Avg': 'Fwd Avg Packets/Bulk',
        'Fwd Blk Rate Avg': 'Fwd Avg Bulk Rate',
        'Bwd Byts/b Avg': 'Bwd Avg Bytes/Bulk',
        'Bwd Pkts/b Avg': 'Bwd Avg Packets/Bulk',
        'Bwd Blk Rate Avg': 'Bwd Avg Bulk Rate',
        'Subflow Fwd Pkts': 'Subflow Fwd Packets',
        'Subflow Fwd Byts': 'Subflow Fwd Bytes',
        'Subflow Bwd Pkts': 'Subflow Bwd Packets',
        'Subflow Bwd Byts': 'Subflow Bwd Bytes',

        # --- 窗口与其它杂项 ---
        'Init Fwd Win Byts': 'Init_Win_bytes_forward',  # 2017用下划线，2018用缩写
        'Init Bwd Win Byts': 'Init_Win_bytes_backward',
        'Fwd Act Data Pkts': 'act_data_pkt_fwd',
        'Fwd Seg Size Min': 'min_seg_size_forward',

        # --- Active / Idle (名字完全一样，不需要映射) ---
        # Active Mean, Std, Max, Min
        # Idle Mean, Std, Max, Min
    }

def set_seed(seed_value=2025):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    print(f"✅ 全局随机种子已固定为: {seed_value}")


# =================================================================
# --- 自定义 Log-MinMax Scaler (解决长尾分布问题) ---
# =================================================================
# config.py 中的 LogMinMaxScaler 类 (修复版)

class LogMinMaxScaler:
    """
    自定义缩放器：先进行 Log1p 变换，再进行 MinMax 缩放。
    解决网络流量特征（如 Duration, Bytes）跨度过大导致的长尾分布问题。
    """

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        # 记录列名 (如果是DataFrame)
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns

        # ✅ 核心修复: 强制将数据截断为非负数 (处理脏数据中的负值)
        # 将 DataFrame 或 Numpy 数组中的负数全部置为 0
        X_safe = np.maximum(X, 0)

        # 1. Log变换: log(1 + x)
        X_log = np.log1p(X_safe)

        # 再次清洗: 万一 log 产生了 inf (虽然 max(0) 后不太可能，但为了稳健)
        if isinstance(X_log, pd.DataFrame):
            X_log.replace([np.inf, -np.inf], 0, inplace=True)
            X_log.fillna(0, inplace=True)
        else:
            X_log = np.nan_to_num(X_log, posinf=0, neginf=0)

        # 2. MinMax fit
        self.scaler.fit(X_log)
        return self

    def transform(self, X):
        # ✅ 核心修复: 同样在 transform 时强制非负
        X_safe = np.maximum(X, 0)

        # 1. Log变换
        X_log = np.log1p(X_safe)

        # 清洗潜在的 inf
        if isinstance(X_log, pd.DataFrame):
            X_log.replace([np.inf, -np.inf], 0, inplace=True)
            X_log.fillna(0, inplace=True)
        else:
            X_log = np.nan_to_num(X_log, posinf=0, neginf=0)

        # 2. MinMax transform
        return self.scaler.transform(X_log)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        # 1. MinMax inverse
        X_log = self.scaler.inverse_transform(X_scaled)
        # 2. Log inverse: exp(x) - 1
        X_original = np.expm1(X_log)
        # 3. 强制非负
        return np.maximum(X_original, 0)


print("✅ LogMinMaxScaler 类已加载 (Config内嵌版)")

# =================================================================
# --- 最终特征体系：逻辑自洽版 (Hard Constraints Ready) ---
# =================================================================

# ✅ 1. 行动集 (ATTACKER_ACTION_SET) - 核心预测目标
# 这些是模型(LSTM/CAE)直接修改或生成的变量。必须是相互独立的。
ATTACKER_ACTION_SET = sorted([
    # --- 空间域 (独立变量) ---
    'Total Fwd Packets',
    'Total Backward Packets',  # 注意：Bwd Packets 也是独立的，应该预测
    'Average Packet Size',  # 预测平均包大小，而不是总长度（更易学习）

    # --- 时间域 (独立变量) ---
    'Flow Duration',
    'Flow IAT Mean', 'Flow IAT Std',
    'Fwd IAT Mean', 'Fwd IAT Std',
    'Bwd IAT Mean', 'Bwd IAT Std',
    'Active Mean', 'Idle Mean',
])

# ✅ 2. 可计算集 (CALCULABLE_SET)
# 这些变量将通过数学公式强制计算得出，绝不让神经网络预测！
# 这样可以保证 100% 的数学逻辑自洽，攻击者无法抓到把柄。
CALCULABLE_SET = sorted([
    'Total Length of Fwd Packets',  # = Total Fwd Pkts * Avg Pkt Size (近似)
    'Total Length of Bwd Packets',  # = Total Bwd Pkts * Avg Pkt Size (近似)
    'Flow Bytes/s',
    'Flow Packets/s',
    'Packet Length Mean',
    'Down/Up Ratio',
    # 如果原本有 'Total Length'，在这里算
])

# ✅ 3. 复杂关联集 (COMPLEX_SET)
# 这些是难以通过简单公式计算的统计特征（如极值、方差）。
# 依然交给 LSTM Predictor (TIER 3) 去预测。
COMPLEX_SET = sorted([
    # 包长统计细节
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Std',
    'Packet Length Std', 'Packet Length Variance',

    # 时间极值
    'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Max', 'Bwd IAT Min'
])

# ✅ 4. 防御者集 (DEFENDER_SET)
DEFENDER_SET = sorted(list(set(ATTACKER_ACTION_SET) | set(CALCULABLE_SET) | set(COMPLEX_SET)))

# ✅ 5. 认知集 (ATTACKER_KNOWLEDGE_SET)
# CAE 输入。可以包含 CALCULABLE 的特征，因为输入时是看真实数据的。
ATTACKER_KNOWLEDGE_SET = sorted(list(set(ATTACKER_ACTION_SET) | {
    'Flow Bytes/s', 'Flow Packets/s',
    'Packet Length Mean',
    'Flow IAT Max', 'Fwd Packet Length Max'
}))

print("特征体系加载完毕:")
print(f"  - ACTION_SET: {len(ATTACKER_ACTION_SET)} 维 (空间+时间)")
print(f"  - CALCULABLE_SET: {len(CALCULABLE_SET)} 维")
print(f"  - COMPLEX_SET: {len(COMPLEX_SET)} 维 (待预测)")
print(f"  - DEFENDER_SET: {len(DEFENDER_SET)} 维 (总目标)")
print(f"  - KNOWLEDGE_SET: {len(ATTACKER_KNOWLEDGE_SET)} 维 (CAE输入)")

# --- 交叉验证 ---
assert set(ATTACKER_ACTION_SET).issubset(set(ATTACKER_KNOWLEDGE_SET)), "行动集必须是认知集的子集!"
assert set(ATTACKER_KNOWLEDGE_SET).issubset(set(DEFENDER_SET)), "认知集必须是防御者集的子集!"
print("✅ 特征集逻辑自洽性通过验证。")