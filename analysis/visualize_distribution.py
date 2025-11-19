# analysis/visualize_distribution.py (FIXED AGAIN)

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import os

# 将项目根目录添加到Python路径中，以便导入config
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from config import DEFENDER_SET

# --- 1. 配置参数 ---
TSNE_PERPLEXITY = 30
RANDOM_STATE = 42

# --- 2. 加载数据 ---
print("🚀 [步骤1] 正在加载数据...")
try:
    camouflage_bot_df = pd.read_csv(os.path.join(project_root, 'data', 'generated', 'final_camouflage_bot_3tier_lstm.csv'))
    benign_df = pd.read_csv(os.path.join(project_root, 'data', 'filtered', 'benign_traffic.csv'))
    real_bot_df = pd.read_csv(os.path.join(project_root, 'data', 'filtered', 'bot_traffic_target.csv'))
    print(f"  - 伪装Bot: {len(camouflage_bot_df)} 条")
    print(f"  - 原始良性: {len(benign_df)} 条")
    print(f"  - 真实Bot: {len(real_bot_df)} 条")
except FileNotFoundError as e:
    print(f"❌ 文件未找到: {e}")
    print("请确保您的项目结构和文件名与代码中的路径匹配。")
    sys.exit(1)

# --- 3. 动态确定采样数量并准备数据 ---
n_samples = min(len(camouflage_bot_df), len(benign_df), len(real_bot_df))
if n_samples == 0:
    print("❌ 错误：至少有一个数据集为空，无法进行可视化。")
    sys.exit(1)

# 增加一个检查，确保perplexity的值小于样本数
if TSNE_PERPLEXITY >= n_samples:
    print(f"⚠️ 警告: t-SNE 的 Perplexity ({TSNE_PERPLEXITY}) 不能大于等于样本数 ({n_samples}).")
    TSNE_PERPLEXITY = n_samples - 1
    print(f"   -> 已自动调整 Perplexity 为: {TSNE_PERPLEXITY}")


print(f"\n🚀 [步骤2] 检测到最小数据集有 {n_samples} 个样本，将以此数量进行均衡采样...")

# 从每个数据集中采样
sample_benign = benign_df.sample(n=n_samples, random_state=RANDOM_STATE)
sample_real_bot = real_bot_df.sample(n=n_samples, random_state=RANDOM_STATE)
sample_camouflage_bot = camouflage_bot_df.sample(n=n_samples, random_state=RANDOM_STATE)

# 提取DEFENDER_SET特征
X_benign = sample_benign[DEFENDER_SET]
X_real_bot = sample_real_bot[DEFENDER_SET]
X_camouflage = sample_camouflage_bot[DEFENDER_SET]

# 合并数据
X_combined = pd.concat([X_benign, X_real_bot, X_camouflage], axis=0)

# 创建标签
y_benign = np.full(X_benign.shape[0], 'Real Benign')
y_real_bot = np.full(X_real_bot.shape[0], 'Real Bot')
y_camouflage = np.full(X_camouflage.shape[0], 'Camouflage Bot')
y_combined = np.concatenate([y_benign, y_real_bot, y_camouflage])

print("  - 数据准备完成。")


# --- 4. 数据缩放 ---
print("\n🚀 [步骤3] 正在使用 global_scaler.pkl 进行数据缩放...")
try:
    scaler_path = os.path.join(project_root, 'models', 'global_scaler.pkl')
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X_combined[scaler.feature_names_in_])
    print("  - 缩放完成。")
except FileNotFoundError:
    print(f"❌ Scaler文件未找到: {scaler_path}")
    print("请先运行 preprocess/step3_build_global_scaler.py 来生成scaler。")
    sys.exit(1)
except Exception as e:
    print(f"❌ 数据缩放时发生错误: {e}")
    sys.exit(1)

# --- 5. 执行t-SNE降维 ---
print(f"\n🚀 [步骤4] 正在执行 t-SNE 降维 (Perplexity={TSNE_PERPLEXITY})... 这可能需要几分钟...")
# -------------------- V V V 这里是修改的地方 V V V --------------------
tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=RANDOM_STATE, max_iter=1000, init='pca', learning_rate='auto')
# -------------------- A A A 这里是修改的地方 A A A --------------------
X_tsne = tsne.fit_transform(X_scaled)
print("  - t-SNE 完成。")


# --- 6. 绘图 ---
print("\n🚀 [步骤5] 正在生成可视化图表...")
plot_df = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
plot_df['label'] = y_combined

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 10))

palette = {
    'Real Benign': 'green',
    'Real Bot': 'red',
    'Camouflage Bot': 'blue'
}

sns.scatterplot(
    x='TSNE1', y='TSNE2',
    hue='label',
    palette=palette,
    data=plot_df,
    legend='full',
    alpha=0.6,
    s=20,
    ax=ax
)

ax.set_title(f'Feature Space Distribution (t-SNE, Perplexity={TSNE_PERPLEXITY})', fontsize=16)
ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
ax.legend(title='Data Type', markerscale=2)

output_path = os.path.join(project_root, 'figures', 'tsne_distribution_analysis.png')
plt.savefig(output_path, dpi=300)

print(f"\n✅ 可视化图表已保存到: {output_path}")
plt.show()