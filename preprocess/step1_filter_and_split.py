# preprocess/step1_filter_and_split.py
import pandas as pd
import os
import sys

# --- 路径黑魔法：把上级目录加入 sys.path 以便导入 config ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 从 config 导入配置，而不是在本地写死
from config import RAW_DATA_PATH, PROCESSED_DIR, MALICIOUS_LABEL, CURRENT_DATASET

# --- 输出文件名 (根据 config 动态生成的路径) ---
os.makedirs(PROCESSED_DIR, exist_ok=True)
benign_output_path = os.path.join(PROCESSED_DIR, 'benign_traffic.csv')
bot_output_path = os.path.join(PROCESSED_DIR, 'bot_traffic_target.csv')


# --- 主函数 ---
def main():
    print(f"🚀 启动数据筛选脚本...")
    print(f"正在加载原始数据: {RAW_DATA_PATH}")

    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ 错误: 文件不存在! 请检查路径: {RAW_DATA_PATH}")
        return

    # 使用 low_memory=False 避免DtypeWarning
    try:
        df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    except Exception as e:
        print(f"❌ 读取CSV失败: {e}")
        return

    # ✅ 关键步骤：清理列名
    df.columns = df.columns.str.strip()
    print("✅ 列名已清洗 (去除了首尾空格)")

    # ---------------------------------------------------------
    # 🔥 [新增代码] 统一标签格式 (核心修复)
    # ---------------------------------------------------------
    # 1. 去除标签列内容的首尾空格 (防止 ' BENIGN' 这种情况)
    if df['Label'].dtype == 'object':
        df['Label'] = df['Label'].str.strip()

    # 2. 统一将 'BENIGN' (全大写, 2017版) 替换为 'Benign' (首字母大写, 2018版)
    # 这样后续代码只需要筛选 'Benign' 即可同时适配两个数据集
    df['Label'] = df['Label'].replace({'BENIGN': 'Benign'})

    print("✅ 标签格式已统一 (BENIGN -> Benign)")
    # ---------------------------------------------------------

    # 确保标签列存在
    if 'Label' not in df.columns:
        print(f"❌ 错误: 找不到 'Label' 列。现有列名: {df.columns.tolist()}")
        return

    print(f"\n📊 [{CURRENT_DATASET}] 原始标签分布:")
    print(df['Label'].value_counts())

    # --- 筛选良性流量 ---
    # 2018年的良性标签也是 'Benign' (注意大小写，CIC2018有时是 'Benign')
    benign_df = df[df['Label'] == 'Benign'].copy()

    # 2018年数据量巨大，为了调试方便，如果是2018，可以先采样一部分 (例如 50万条)
    # 如果你想跑全量，注释掉下面这两行
    #if len(benign_df) > 500000:
        #print(f"⚠️ 良性数据过多 ({len(benign_df)}条)，随机采样 500,000 条以加速实验...")
        #benign_df = benign_df.sample(n=500000, random_state=42)

    print(f"\n筛选出 {len(benign_df)} 条良性流量...")
    benign_df.to_csv(benign_output_path, index=False)
    print(f"✅ 已保存良性流量到: {benign_output_path}")

    # --- 筛选Bot流量 ---
    # 使用 config 中定义的 MALICIOUS_LABEL
    bot_df = df[df['Label'] == MALICIOUS_LABEL].copy()

    if len(bot_df) == 0:
        print(f"❌ 警告: 未找到标签为 '{MALICIOUS_LABEL}' 的流量!")
        print("请检查上方打印的 '原始标签分布'，确认2018数据集中Botnet的具体标签名。")
        # 针对 CIC2018 Friday 数据的备选方案：如果不是 'Bot'，可能是 'Bot-Zeus' 或 'Bot-Ares'
        # 你可以在这里手动改为包含 'Bot' 字符的
        # bot_df = df[df['Label'].str.contains('Bot', case=False)].copy()
    else:
        print(f"\n筛选出 {len(bot_df)} 条Bot流量 ({MALICIOUS_LABEL})...")
        bot_df.to_csv(bot_output_path, index=False)
        print(f"✅ 已保存Bot流量到: {bot_output_path}")


if __name__ == "__main__":
    main()