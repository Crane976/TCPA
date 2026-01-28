# evaluation/verify_watermark.py (FINAL: ADAPTIVE DATASET SUPPORT)
import pandas as pd
import numpy as np
import os
import sys

# 路径设置
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.append(project_root)

# 🔥 引入 config
import config

# 配置
# 动态匹配 STEP3 的输出文件名
GENERATED_FILENAME = f'final_camouflage_{config.CURRENT_DATASET}_TSR100.csv'
GENERATED_PATH = os.path.join(project_root, 'data', 'generated', GENERATED_FILENAME)

# 动态匹配测试集路径
TEST_SET_PATH = os.path.join(project_root, 'data', 'splits', 'holdout_test_set.csv')

WATERMARK_KEY = 97
WATERMARK_FEATURE = 'Flow Duration'


def verify(df, name):
    print(f"\n--- 验证数据集: {name} ---")
    print(f"   样本数量: {len(df)}")

    if WATERMARK_FEATURE not in df.columns:
        print(f"❌ 错误: 特征 '{WATERMARK_FEATURE}' 缺失，无法验证")
        return 0

    # 确保转换为整数 (微秒)
    try:
        values = df[WATERMARK_FEATURE].values.astype(int)
    except ValueError:
        print("❌ 错误: Flow Duration包含非数值字符，尝试清洗...")
        df[WATERMARK_FEATURE] = pd.to_numeric(df[WATERMARK_FEATURE], errors='coerce').fillna(0)
        values = df[WATERMARK_FEATURE].values.astype(int)

    # 提取逻辑: 余数为 0 即为我方流量
    # Verification: Check if value % K == 0
    matches = (values % WATERMARK_KEY == 0)

    # 计算比例
    accuracy = np.mean(matches)

    # 打印详细统计
    matched_count = np.sum(matches)
    print(f"   检出数量: {matched_count} / {len(df)}")
    print(f"   检出率 (Rate): {accuracy * 100:.4f}%")
    return accuracy


def main():
    print("=" * 60)
    print(f"🔐 水印溯源验证 (Key={WATERMARK_KEY}) - {config.CURRENT_DATASET}")
    print("=" * 60)

    # 1. 验证伪装流量 (Self-Identification Rate)
    # 预期: 接近 100%
    if os.path.exists(GENERATED_PATH):
        df_gen = pd.read_csv(GENERATED_PATH)
        acc_gen = verify(df_gen, "伪装诱饵流量 (Self)")
    else:
        print(f"❌ 未找到生成的诱饵文件: {GENERATED_PATH}")
        acc_gen = 0

    # 2. 验证真实流量 (False Positive Rate)
    # 预期: 接近 1/Key (约 1.03%)
    if os.path.exists(TEST_SET_PATH):
        df_test = pd.read_csv(TEST_SET_PATH)
        # 我们只关心真实流量是否会被误判，可以全量测，也可以只测Benign
        # 这里全量测，因为Bot也是真实流量，也不该被误判
        acc_test = verify(df_test, "真实背景流量 (Real Traffic)")
    else:
        print(f"❌ 未找到测试集文件: {TEST_SET_PATH}")
        acc_test = 0

    # 3. 理论值
    theoretical_fpr = (1 / WATERMARK_KEY) * 100

    print("\n" + "=" * 60)
    print(f"📊 溯源性能总结:")
    print(f"   - 自身识别率 (TPR / Self-ID): \t{acc_gen * 100:.2f}%  (目标: 100.00%)")
    print(f"   - 误伤率     (FPR / Collision): \t{acc_test * 100:.2f}%  (理论值: ~{theoretical_fpr:.2f}%)")
    print("=" * 60)

    if acc_gen > 0.99:
        print("✅ 溯源系统运行正常: 诱饵完全可控。")
    else:
        print("⚠️ 警告: 诱饵识别率未达标，请检查生成脚本中的硬约束逻辑。")


if __name__ == "__main__":
    main()