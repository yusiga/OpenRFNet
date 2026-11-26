import glob
import os
import h5py
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.signal import stft

# ======================
# 基本配置
# ======================
base_dir_rf = '../data_hover/rf_data_1ms'
base_dir_stft = '../data_hover/stft_data_1ms'

Fs = int(100e6)
DT = 0.001
modu_snr_size = int(Fs * DT)  # 100,000
# STFT 参数
window_size = 448
overlap_ratio = 0.5
window = 'hamming'
eps = 1e-20


# ======================
# Step 1: 计算环境信号阈值
# ======================
def get_noise_threshold(env_mat_path, percentile=95):
    print(f"--- 正在计算环境噪声阈值 ({percentile}%) ---")
    data = loadmat(env_mat_path)
    I, Q = data['RF0_I'][0], data['RF0_Q'][0]
    x = I + 1j * Q

    num_slices = len(x) // modu_snr_size
    x_slices = x[:num_slices * modu_snr_size].reshape(num_slices, modu_snr_size)
    # 计算每个切片的平均功率
    powers = np.mean(np.abs(x_slices) ** 2, axis=1)
    threshold = np.percentile(powers, percentile)

    print(f"环境切片数: {num_slices}, 选定阈值: {threshold:.3e}")
    return threshold


# ======================
# Step 2: 处理单个切片并保存
# ======================
def save_single_slice(slice_data, label, mat_basename, subset, idx):
    # 路径构建
    rf_folder = os.path.join(base_dir_rf, label, subset, mat_basename)
    stft_folder = os.path.join(base_dir_stft, label, subset, mat_basename)
    os.makedirs(rf_folder, exist_ok=True)
    os.makedirs(stft_folder, exist_ok=True)

    magnitude = np.sqrt(slice_data.real ** 2 + slice_data.imag ** 2)
    magnitude_db = 20 * np.log10(magnitude + eps)

    # db_min = np.min(magnitude_db)
    # db_max = np.max(magnitude_db)
    #
    # magnitude_norm = (magnitude_db - db_min) / (db_max - db_min)

    magnitude_db = magnitude_db[np.newaxis, :]

    with h5py.File(os.path.join(rf_folder, f'{subset}_{idx}_rf.h5'), 'w') as f:
        f.create_dataset('IQ Magnitude', data=magnitude_db)
        f.attrs['label'] = label

    # 保存 STFT
    _, _, Zxx = stft(slice_data, fs=Fs, nperseg=window_size,
                     noverlap=int(window_size * overlap_ratio),
                     window=window, return_onesided=False)
    magnitudes = 20 * np.log10(np.abs(Zxx) + eps)
    # min_magnitude = np.min(magnitudes)
    # max_magnitude = np.max(magnitudes)
    # normalized_magnitudes = (magnitudes - min_magnitude) / (max_magnitude - min_magnitude)
    Zxx_combined = np.expand_dims(magnitudes, axis=0).astype(np.float32)
    with h5py.File(os.path.join(stft_folder, f'{subset}_{idx}_stft.h5'), 'w') as f:
        f.create_dataset('STFT Magnitude', data=Zxx_combined)
        f.attrs['label'] = label


# ======================
# 主逻辑
# ======================
if __name__ == '__main__':
    # 1. 定标
    env_file = '../../single_hover/environment.mat'
    threshold = get_noise_threshold(env_file)

    # 2. 获取待处理文件 (排除环境文件)
    mat_files_paths = glob.glob('../..//single_hover/*.mat')

    for file_path in mat_files_paths:
        mat_basename = os.path.splitext(os.path.basename(file_path))[0]

        # 过滤条件 (根据你之前的逻辑)
        parts = mat_basename.split('_')
        if len(parts) == 1 or parts[-2] != 'u1':
            continue

        print(f"\nProcessing: {mat_basename}")
        label = '_'.join(parts[0:2])

        # 读取并切片
        data = loadmat(file_path)
        I, Q = data['RF0_I'][0], data['RF0_Q'][0]
        data_ch = I + 1j * Q

        total_raw_slices = len(data_ch) // modu_snr_size
        valid_slices = []

        # 能量检测过滤
        for i in range(total_raw_slices):
            sl = data_ch[i * modu_snr_size: (i + 1) * modu_snr_size]
            if np.mean(np.abs(sl) ** 2) >= threshold:
                valid_slices.append(sl)

        num_valid = len(valid_slices)
        print(f"原始切片: {total_raw_slices}, 有效切片: {num_valid} (保留率: {num_valid / total_raw_slices:.1%})")

        if num_valid < 5:  # 如果有效切片太少，不足以划分 3:1:1 则跳过
            continue

        # 3. 划分数据集 (3:1:1)
        indices = np.arange(num_valid)
        temp_indexes, test_indexes = train_test_split(indices, test_size=0.2, random_state=42)
        train_indexes, val_indexes = train_test_split(temp_indexes, test_size=0.25, random_state=42)
        # 4. 执行保存
        for subset, idx_list in zip(['train', 'val', 'test'], [train_indexes, val_indexes, test_indexes]):
            for i, val in enumerate(idx_list):
                save_single_slice(valid_slices[val], label, mat_basename, subset, i)
            print(f"--- {subset} set saved ({len(idx_list)} slices)")

    print("\n[DONE] 所有数据已清洗并按 3:1:1 划分完成。")
