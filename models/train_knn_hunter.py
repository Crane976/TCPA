# models/train_knn_hunter.py (UNIVERSAL ROBUST VERSION)
# 适配: CIC-IDS2017 (小样本Bot) & CSE-CIC-IDS2018 (海量样本)
import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
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
HUNTER_MODEL_PATH = os.path.join(MODELS_DIR, 'knn_hunter.pkl')


def main():
    set_seed(2025)
    print("=" * 60)
    print("🚀 开始训练 KNN Hunter (通用适配版)...")
    print("=" * 60)

    # --- 1. 加载数据 ---
    print("正在加载数据...")
    if not os.path.exists(TRAIN_SET_PATH):
        print(f"❌ 错误: 找不到训练集 {TRAIN_SET_PATH}")
        return

    try:
        # Step 2 已经帮我们把列名标准化了，所以这里不用担心列名映射
        df_train_full = pd.read_csv(TRAIN_SET_PATH)
        df_test = pd.read_csv(TEST_SET_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_names = scaler.feature_names_in_
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # --- 2. 数据清洗 (去除 Inf/NaN) ---
    print("正在清洗数据 (去除 Inf/NaN)...")
    for df, name in [(df_train_full, "训练集"), (df_test, "测试集")]:
        len_before = len(df)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=[col for col in DEFENDER_SET if col in df.columns], inplace=True)
        print(f"   -> {name}: 清洗掉 {len_before - len(df)} 条脏数据")

    # --- 3. 构建训练子集 (核心逻辑修改) ---
    print("\n[步骤1] 构建训练子集 (智能采样)...")
    X_full = df_train_full[DEFENDER_SET]
    y_full = df_train_full['label']

    # 划分验证集 (保持真实比例，用于寻找阈值)
    # 注意: 这一步是为了模拟真实环境下的阈值选择，不做平衡处理
    X_train_pool, X_val_natural, y_train_pool, y_val_natural = train_test_split(
        X_full, y_full, test_size=0.2, random_state=2025, stratify=y_full
    )

    # 分离 Bot 和 Benign
    df_pool = pd.concat([X_train_pool, y_train_pool], axis=1)
    df_bot = df_pool[df_pool['label'] == 1]
    df_benign = df_pool[df_pool['label'] == 0]

    # 🔥 [策略A: 针对海量数据的性能保护]
    # KNN 推理极慢。如果 Bot 样本超过 2万 (如 IDS2018 有20多万)，
    # 必须下采样 Bot，否则验证和测试阶段会跑死机。
    # 2万个样本足够勾勒出 Bot 的决策边界了。
    MAX_BOT_SAMPLES = 20000

    if len(df_bot) > MAX_BOT_SAMPLES:
        print(f"⚠️ [性能优化] Bot样本过多 ({len(df_bot)})，下采样至 {MAX_BOT_SAMPLES} 以加速KNN推理。")
        df_bot = df_bot.sample(n=MAX_BOT_SAMPLES, random_state=2025)

    n_bot_final = len(df_bot)

    # 🔥 [策略B: 针对不平衡数据的比例控制]
    # 我们希望良性样本多一些 (例如 1:10)，以减少误报。
    TARGET_RATIO = 10
    target_benign_count = n_bot_final * TARGET_RATIO

    # 🔥 [策略C: 针对 IDS2018 的防崩溃修复]
    # 确保我们不索取超过实际拥有的良性样本数
    n_benign_available = len(df_benign)

    if target_benign_count > n_benign_available:
        print(f"⚠️ [防崩溃] 目标良性样本数 ({target_benign_count}) 超过库存 ({n_benign_available})。")
        print("   -> 将使用全部可用良性样本。")
        n_benign_final = n_benign_available
    else:
        n_benign_final = target_benign_count

    df_benign_sampled = df_benign.sample(n=n_benign_final, random_state=2025)

    # 合并
    df_train_balanced = pd.concat([df_bot, df_benign_sampled])

    print(f"✅ 最终训练集构建完成:")
    print(f"   -> Bot样本: {n_bot_final}")
    print(f"   -> Benign样本: {n_benign_final}")
    print(f"   -> 总计: {len(df_train_balanced)} (比例 1:{n_benign_final / n_bot_final:.1f})")

    # --- 4. 缩放 ---
    print("\n[步骤2] 数据标准化 (Log-MinMax)...")
    # 注意: scaler 已经是在全量训练集上 fit 过的，这里直接 transform
    X_train_final = scaler.transform(df_train_balanced[DEFENDER_SET])
    y_train_final = df_train_balanced['label']

    # 验证集和测试集也要转换
    X_val_natural_scaled = scaler.transform(X_val_natural)
    X_test_scaled = scaler.transform(df_test[DEFENDER_SET])
    y_test = df_test['label']

    # --- 5. 训练 ---
    print(f"\n[步骤3] 训练 KNN (K=31, Distance-weighted)...")
    # K=31 是为了在含噪环境中获得更平滑的决策边界
    knn_model = KNeighborsClassifier(n_neighbors=31, weights='distance', n_jobs=-1)

    with tqdm(total=1, desc="KNN Fitting") as pbar:
        knn_model.fit(X_train_final, y_train_final)
        pbar.update(1)

    # --- 6. 阈值寻优 (关键步骤) ---
    print("\n[步骤4] 在【真实分布验证集】上寻找最佳决策阈值...")
    # 注意: 这里的 X_val_natural 保持了真实世界的比例
    # KNN predict_proba 会比较慢，请耐心等待
    print(f"   -> 正在对 {len(X_val_natural_scaled)} 条验证数据进行推理 (这可能需要几分钟)...")
    val_probs = knn_model.predict_proba(X_val_natural_scaled)[:, 1]

    best_threshold = 0.5
    best_f1 = 0
    # 搜索范围
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]

    for thr in thresholds:
        y_val_pred = (val_probs >= thr).astype(int)
        f1 = f1_score(y_val_natural, y_val_pred)
        # print(f"      Thr={thr:.2f}, F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thr

    print(f"✅ 最佳阈值锁定: {best_threshold:.2f} (验证集 F1: {best_f1:.4f})")

    # --- 7. 保存与评估 ---
    joblib.dump(knn_model, HUNTER_MODEL_PATH)
    print(f"💾 模型已保存至: {HUNTER_MODEL_PATH}")

    print(f"\n--- 最终测试集评估 (阈值={best_threshold:.2f}) ---")
    print(f"   -> 正在对 {len(X_test_scaled)} 条测试数据进行推理...")
    test_probs = knn_model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (test_probs >= best_threshold).astype(int)

    print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Bot (1)'], digits=4))


if __name__ == "__main__":
    main()