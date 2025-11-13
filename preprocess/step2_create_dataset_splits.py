# step2_create_dataset_splits.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys # ✅ 导入sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from config import set_seed # ✅ 从config导入函数

# --- 配置 ---
BENIGN_IN = r'D:\DTCA\data\filtered\benign_traffic.csv'
BOT_IN = r'D:\DTCA\data\filtered\bot_traffic_target.csv'
OUTPUT_DIR = r'D:\DTCA\data\splits'  # 输出到一个全新的文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    set_seed(2025)  # ✅ 在main函数开头调用
    print("Creating permanent train/test splits...")
    df_benign = pd.read_csv(BENIGN_IN)
    df_bot = pd.read_csv(BOT_IN)
    df_benign['label'] = 0
    df_bot['label'] = 1

    df_full = pd.concat([df_benign, df_bot], ignore_index=True)

    # 将80%作为训练集，20%作为永久的、不可触碰的留出测试集
    df_train, df_test = train_test_split(df_full, test_size=0.2, random_state=2025, stratify=df_full['label'])

    df_train.to_csv(os.path.join(OUTPUT_DIR, 'training_set.csv'), index=False)
    df_test.to_csv(os.path.join(OUTPUT_DIR, 'holdout_test_set.csv'), index=False)

    print(f"Training set created with {len(df_train)} samples.")
    print(f"Hold-out test set created with {len(df_test)} samples.")
    print(f"Splits saved to '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()