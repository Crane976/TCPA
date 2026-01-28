import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import os

# 将项目根目录添加到Python路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 导入配置
from config import DEFENDER_SET, COLUMN_MAPPING

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] # 或者 'DejaVu Sans'

# --- 配置参数 ---
TSNE_PERPLEXITY = 30
RANDOM_STATE = 42
MAX_SAMPLES_PER_CLASS = 2000  # 稍微减少一点点，防止太乱，或者保持 2000 也可以
FIGURE_SIZE = (16, 9)  # 16:9 比例，更适合论文宽幅插图

# --- 数据集映射 ---
DATASETS = {
    'CIC-IDS2017': {
        'benign': 'data/cic_ids_2017/filtered/benign_traffic.csv',
        'bot': 'data/cic_ids_2017/filtered/bot_traffic_target.csv',
        'files': {
            'No Cluster': 'data/generated/variant_A_no_cluster_CIC-IDS2017.csv',
            'Final Model': 'data/generated/final_camouflage_CIC-IDS2017_TSR100.csv',
            'No Constraint': 'data/generated/variant_B_no_constraint_CIC-IDS2017.csv'
        }
    },
    'CSE-CIC-IDS2018': {
        'benign': 'data/cse_cic_ids_2018/filtered/benign_traffic.csv',
        'bot': 'data/cse_cic_ids_2018/filtered/bot_traffic_target.csv',
        'files': {
            'No Cluster': 'data/generated/variant_A_no_cluster_CSE-CIC-IDS2018.csv',
            'Final Model': 'data/generated/final_camouflage_CSE-CIC-IDS2018_TSR100.csv',
            'No Constraint': 'data/generated/variant_B_no_constraint_CSE-CIC-IDS2018.csv'
        }
    }
}


def load_and_sample(path, label, n_samples):
    """加载数据并采样，增加列名映射容错处理"""
    try:
        full_path = os.path.join(project_root, path)
        if not os.path.exists(full_path):
            return None

        df = pd.read_csv(full_path)

        # 列名映射逻辑
        missing_cols = [c for c in DEFENDER_SET if c not in df.columns]
        if len(missing_cols) > 0:
            rename_dict = {k: v for k, v in COLUMN_MAPPING.items() if k in df.columns}
            df.rename(columns=rename_dict, inplace=True)

        if len(df) > n_samples:
            df = df.sample(n=n_samples, random_state=RANDOM_STATE)

        try:
            return df[DEFENDER_SET]
        except KeyError:
            return None
    except Exception:
        return None


def main():
    # 加载 Scaler
    scaler_path = os.path.join(project_root, 'models', 'global_scaler.pkl')
    try:
        scaler = joblib.load(scaler_path)
        print("✅ Global Scaler 加载成功")
    except:
        print("❌ Scaler 加载失败")
        return

    # 创建画布
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE)
    # 调整布局间距，留出顶部放图例，留出左侧放行标
    plt.subplots_adjust(top=0.88, bottom=0.05, left=0.08, right=0.98, hspace=0.1, wspace=0.05)

    # 定义更高级的配色 (Seaborn Muted/Deep)
    # Real Benign: 灰色/浅绿 (作为背景)
    # Real Bot: 红色 (强调)
    # Camouflage: 蓝色 (强调)
    palette = {
        'Real Benign': '#a1d99b',  # 浅绿色 (Light Green) - 不抢眼
        'Real Bot': '#d62728',  # 鲜红色 (Red)
        'Camouflage Bot': '#1f77b4'  # 鲜蓝色 (Blue)
    }

    # 如果你更喜欢之前的深绿色，可以改回 '#2ca02c'

    rows = ['CIC-IDS2017', 'CSE-CIC-IDS2018']
    cols = ['No Cluster', 'Final Model', 'No Constraint']

    for row_idx, dataset_name in enumerate(rows):
        print(f"\n🚀 处理数据集: {dataset_name} ...")

        if dataset_name == 'CIC-IDS2017':
            path_benign = DATASETS['CIC-IDS2017']['benign']
            path_bot = DATASETS['CIC-IDS2017']['bot']
        else:
            path_benign = DATASETS['CSE-CIC-IDS2018']['benign']
            path_bot = DATASETS['CSE-CIC-IDS2018']['bot']

        df_benign = load_and_sample(path_benign, 'Real Benign', MAX_SAMPLES_PER_CLASS)
        df_bot = load_and_sample(path_bot, 'Real Bot', MAX_SAMPLES_PER_CLASS)

        if df_benign is None or df_bot is None:
            continue

        for col_idx, strategy_name in enumerate(cols):
            ax = axes[row_idx, col_idx]

            fname = DATASETS[dataset_name]['files'][strategy_name]
            df_camo = load_and_sample(fname, 'Camouflage Bot', MAX_SAMPLES_PER_CLASS)

            if df_camo is None:
                continue

            # 合并与缩放
            X_combined = pd.concat([df_benign, df_bot, df_camo], axis=0)
            y_combined = (['Real Benign'] * len(df_benign) +
                          ['Real Bot'] * len(df_bot) +
                          ['Camouflage Bot'] * len(df_camo))

            X_scaled = scaler.transform(X_combined[scaler.feature_names_in_])

            # t-SNE
            tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
            X_tsne = tsne.fit_transform(X_scaled)

            # 绘图 - 调整点的大小(s)和透明度(alpha)
            # 关键：调整绘制顺序，让 Bot 和 Camouflage 浮在 Benign 上面
            sns.scatterplot(
                x=X_tsne[:, 0], y=X_tsne[:, 1],
                hue=y_combined,
                palette=palette,
                hue_order=['Real Benign', 'Real Bot', 'Camouflage Bot'],  # 强制绘制顺序
                alpha=0.6,
                s=10,  # 点变小一点，看起来更精致
                ax=ax,
                legend=False,
                linewidth=0  # 去掉点的描边，在大数据量下更清晰
            )

            # --- 样式美化 ---
            # 1. 彻底去除坐标轴刻度和标签 (因为 t-SNE 坐标无物理意义)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")

            # 2. 设置列标题 (策略名) - 仅第一行显示
            if row_idx == 0:
                ax.set_title(strategy_name, fontsize=16, fontweight='bold', pad=15)

            # 3. 设置行标题 (数据集名) - 仅第一列显示，且放在左侧外边
            if col_idx == 0:
                # 使用 text 在坐标轴左侧绘制旋转文字
                ax.text(-0.05, 0.5, dataset_name,
                        transform=ax.transAxes,
                        fontsize=16, fontweight='bold',
                        va='center', ha='right', rotation=90)

    # --- 统一图例 ---
    # 使用自定义 Line2D 创建漂亮的图例点
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Real Benign (Background)',
               markerfacecolor=palette['Real Benign'], markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Real Bot (Target)',
               markerfacecolor=palette['Real Bot'], markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Camouflage Bot (Ours)',
               markerfacecolor=palette['Camouflage Bot'], markersize=12)
    ]

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=3, fontsize=14, frameon=False)  # frameon=False 去掉图例边框，更现代

    pdf_path = os.path.join(project_root, 'figures', 'Figure_2.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"✅ [期刊格式] PDF 文件已保存到: {pdf_path}")

    # 2. 同时也保存一份高分辨率 PNG 用于预览或备用 (提升 DPI 到 600 以保万全)
    png_path = os.path.join(project_root, 'figures', 'Figure_2.png')
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    print(f"✅ [高分预览] PNG 文件已保存到: {png_path}")
    plt.show()


if __name__ == "__main__":
    main()