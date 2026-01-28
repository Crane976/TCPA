# comparative_study/baseline_TSAF/generate_tsaf.py
# Baseline Implementation: TSAF (Time Series Adversarial Framework)
# Reference: Lu et al., Computers & Security 2025
# Logic: Iterative FGSM (I-FGSM) as described in TSAF Algorithm 1

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys
import joblib

# --- 路径适配 ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path: sys.path.append(project_root)

import config
from config import DEFENDER_SET, set_seed
from models.cnn_architecture import CNN_Classifier

# ==========================================================
# --- 配置区 (Match TSAF Algo 1 Inputs) ---
# ==========================================================
# TSAF Algorithm 1 Parameters:
# - n: number of perturbations (Implemented via Mask)
# - epsilon (learning rate/step size): 0.01 (ALPHA)
# - T (iterations): 20
# - dom_range: [0, 1] (Ensured by Scaler & Clip)

MAX_PERTURBATION = 0.1  # 对应 L-inf norm constraint
STEP_SIZE = 0.01  # 对应 Algorithm 1 中的 learning rate epsilon
ITERATIONS = 20  # 对应 Algorithm 1 中的 T

# TSAF 强调只修改时间特征
TIME_FEATURES_KEYWORDS = ['Duration', 'IAT', 'Active', 'Idle']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_time_feature_mask(feature_names):
    """
    [TSAF Function 20] get_mask
    生成掩码，只允许时间特征被修改
    """
    mask = []
    print("\n[TSAF Constraint] Locking non-time features (Spatial & Header)...")
    for feat in feature_names:
        # 严格筛选：必须包含时间关键字，且不能是速率（速率是计算结果）
        is_time = any(k in feat for k in TIME_FEATURES_KEYWORDS)
        if is_time and 'Bytes/s' not in feat and 'Packets/s' not in feat:
            mask.append(1.0)
        else:
            mask.append(0.0)

    mask_tensor = torch.tensor(mask, dtype=torch.float32).to(device)
    print(f"   -> Mask generated. {int(mask_tensor.sum().item())} time features represent the attack surface.")
    return mask_tensor


def iterative_fgsm_attack(model, data_x, target_y, mask, eps, alpha, T):
    """
    [TSAF Algorithm 1] TSAF attack for flow-based time series IDS
    本质是 Iterative-FGSM (I-FGSM)
    """
    # 4: Initialize delta randomly (Small random start to escape local minima)
    delta = torch.zeros_like(data_x).uniform_(-0.01, 0.01).to(device) * mask
    delta.requires_grad = True

    # 动态选择 Loss (适配二分类/多分类)
    with torch.no_grad():
        test_out = model(data_x[:1])
    use_bce = (test_out.shape[1] == 1)

    if use_bce:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    # 7: for step in [1, T] do
    for t in range(T):
        # 9: apply_mask_and_adv (Generate perturbed data)
        # x_adv = x + delta
        perturbed_data = data_x + delta
        perturbed_data = torch.clamp(perturbed_data, 0, 1)  # Ensure dom_range

        # 11: compute predictions
        outputs = model(perturbed_data)

        # 12: calculate loss (Targeted: Minimize loss to 'Bot' class)
        if use_bce:
            loss = loss_fn(outputs, target_y.float().view(-1, 1))
        else:
            loss = loss_fn(outputs, target_y)

        # 14: Compute gradients of loss w.r.t. delta
        model.zero_grad()
        loss.backward()
        grad = delta.grad.data

        # Apply Mask to Gradients (TSAF constraint: only time features update)
        grad = grad * mask

        # 15: Update delta using optimizer (Here we use Sign SGD -> FGSM logic)
        # Targeted Attack: Gradient Descent (Move towards target class 1)
        delta.data = delta.data - alpha * grad.sign()

        # Projection (Clip delta to stay within epsilon ball)
        delta.data = torch.clamp(delta.data, -eps, eps)

        # Re-apply mask to delta to be safe
        delta.data = delta.data * mask

        # Reset gradient for next step
        delta.grad.zero_()

    # 17: Apply mask and delta to generate perturbed feature data
    final_adv_x = data_x + delta.detach()
    final_adv_x = torch.clamp(final_adv_x, 0, 1)

    # 18: Return UAP (Here we return the perturbed samples directly)
    return final_adv_x


def main():
    set_seed(2025)
    print("=" * 60)
    print(f"🚀 [Baseline Reproduction] TSAF: Iterative FGSM Framework")
    print(f"   Dataset: {config.CURRENT_DATASET}")
    print("=" * 60)

    # 1. 动态路径 & 数量
    if config.CURRENT_DATASET == 'CIC-IDS2017':
        NUM_TO_GENERATE = 39300
    else:
        NUM_TO_GENERATE = 100000

    OUTPUT_PATH = os.path.join(project_root, 'data', 'generated', f'baseline_TSAF_{config.CURRENT_DATASET}.csv')

    # 2. 加载数据 (Benign 载体)
    print(f"Loading Benign samples (Source)... Target: {NUM_TO_GENERATE}")
    train_path = os.path.join(config.SPLITS_DIR, 'training_set.csv')

    if not os.path.exists(train_path):
        print(f"❌ 错误: 找不到训练集文件: {train_path}")
        return

    df_train = pd.read_csv(train_path)
    df_benign = df_train[df_train['label'] == 0].sample(n=NUM_TO_GENERATE, replace=True, random_state=2025)

    print(f"Loading Scaler from {config.SCALER_PATH}...")
    scaler = joblib.load(config.SCALER_PATH)

    X_benign = scaler.transform(df_benign[DEFENDER_SET])

    # 转 Tensor
    X_benign_tensor = torch.tensor(X_benign, dtype=torch.float32).to(device)
    # 目标标签: Bot (1)
    target_labels = torch.ones(NUM_TO_GENERATE, dtype=torch.long).to(device)

    # 3. 加载白盒替身模型 (1D-CNN)
    print("Loading Surrogate White-box Model (1D-CNN)...")
    cnn_path = os.path.join(config.MODEL_SAVE_DIR, 'cnn_hunter.pt')

    surrogate_model = CNN_Classifier(feature_dim=len(DEFENDER_SET)).to(device)
    # 忽略 pickle 警告
    try:
        surrogate_model.load_state_dict(torch.load(cnn_path, map_location=device))
    except TypeError:
        # Fallback if weights_only arg causes issue on old torch versions
        surrogate_model.load_state_dict(torch.load(cnn_path, map_location=device))

    surrogate_model.eval()

    # 4. 生成 (Iterative FGSM)
    print(f"Starting TSAF Generation (Iterative FGSM, T={ITERATIONS}, Step={STEP_SIZE})...")

    feature_mask = get_time_feature_mask(DEFENDER_SET)

    BATCH_SIZE = 512
    adv_samples_list = []

    import math
    num_batches = math.ceil(NUM_TO_GENERATE / BATCH_SIZE)

    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, NUM_TO_GENERATE)

        batch_x = X_benign_tensor[start_idx:end_idx]
        batch_y = target_labels[start_idx:end_idx]

        # 调用改名后的函数
        adv_batch = iterative_fgsm_attack(
            surrogate_model, batch_x, batch_y, feature_mask,
            eps=MAX_PERTURBATION, alpha=STEP_SIZE, T=ITERATIONS
        )

        adv_samples_list.append(adv_batch.cpu().numpy())

        if i % 20 == 0:
            print(f"   -> Batch {i}/{num_batches} done.")

    X_adv_np = np.concatenate(adv_samples_list, axis=0)

    # 5. 保存与后处理
    print("Inverse scaling...")
    X_adv_original = scaler.inverse_transform(X_adv_np)

    df_adv = pd.DataFrame(X_adv_original, columns=DEFENDER_SET)
    df_adv['Label'] = 1

    # ==========================================================
    # 🔥 [公平性修正] Post-processing for Fair Comparison
    # 强制修正空间特征的浮点误差，避免 Straw Man 攻击
    # ==========================================================
    print("Applying Integer Rounding to Spatial Features (Fairness Correction)...")

    integer_cols = [
        'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
        'Fwd Header Length', 'Bwd Header Length',
        'Subflow Fwd Packets', 'Subflow Fwd Bytes',
        'Subflow Bwd Packets', 'Subflow Bwd Bytes',
        'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
        'act_data_pkt_fwd', 'min_seg_size_forward',
        'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
        'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio'
    ]

    count_fixed = 0
    for col in integer_cols:
        if col in df_adv.columns:
            df_adv[col] = df_adv[col].clip(lower=0)
            df_adv[col] = df_adv[col].round().astype(int)
            count_fixed += 1

    print(f"   -> Fixed {count_fixed} spatial feature columns to Integers.")

    df_adv.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ TSAF Baseline Generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()