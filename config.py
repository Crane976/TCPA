# config.py (THE ULTIMATE GROUND-TRUTH VERSION 2 - LOGICALLY PERFECT)
import torch
import numpy as np
import random
import os

def set_seed(seed_value=2025):
    # ... (函数内容不变) ...
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
# --- 最终的、三层非对称信息特征体系 (逻辑修正版) ---
# =================================================================

# ✅ 第一层: 防御者集 (DEFENDER_SET) - 上帝视角 (23维)
# 一个宽泛、完备的纯时间统计特征集合，代表信息最全的强大对手。
DEFENDER_SET = sorted([
    'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
])

# ✅ 第二层: 攻击者认知集 (ATTACKER_KNOWLEDGE_SET) - 情报边界 (13维)
# 防御者视野的一个真子集，代表我们通过情报分析所能掌握的核心特征。
# 这是CAE的输入空间。我们排除了更难预测的“Min”、“Max”和宏观的“Total”类特征。
ATTACKER_KNOWLEDGE_SET = sorted([
    'Flow Duration',
    'Flow IAT Mean', 'Flow IAT Std',
    'Fwd IAT Mean', 'Fwd IAT Std',
    'Bwd IAT Mean', 'Bwd IAT Std',
    'Active Mean', 'Active Std',
    'Idle Mean', 'Idle Std',
    'Fwd IAT Total', 'Bwd IAT Total', # 保持Total，因为它是宏观行为的关键
])

# ✅ 第三层: 攻击者行动集 (ATTACKER_ACTION_SET) - 物理约束 (9维)
# 攻击者认知集的一个严格真子集，代表我们最有把握在问题空间中安全操控的特征。
# 这是LSTM的扰动目标。我们只关注最核心的均值和标准差。
ATTACKER_ACTION_SET = sorted([
    'Flow Duration',
    'Flow IAT Mean', 'Flow IAT Std',
    'Fwd IAT Mean', 'Fwd IAT Std',
    'Bwd IAT Mean', 'Bwd IAT Std',
    'Active Mean',
    'Idle Mean',
])


print("最终三层非对称特征体系 (逻辑修正版) 加载完毕:")
print(f"  - 防御者视野 (DEFENDER_SET): {len(DEFENDER_SET)} 个特征")
print(f"  - 攻击者认知 (ATTACKER_KNOWLEDGE_SET): {len(ATTACKER_KNOWLEDGE_SET)} 个特征")
print(f"  - 攻击者行动 (ATTACKER_ACTION_SET): {len(ATTACKER_ACTION_SET)} 个特征")

# --- 交叉验证，确保逻辑自洽 ---
assert set(ATTACKER_ACTION_SET).issubset(set(ATTACKER_KNOWLEDGE_SET)), "行动集必须是认知集的子集!"
assert set(ATTACKER_KNOWLEDGE_SET).issubset(set(DEFENDER_SET)), "认知集必须是防御者集的子集!"
print("✅ 特征集逻辑自洽性通过验证。")