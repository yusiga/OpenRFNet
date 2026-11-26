import h5py
import numpy as np

file_path = '/home/fankaiqing/DroneRFa_ztr/data/stft_30w_clean/T1001/train/T1001_D00_S0011/train_102_stft.h5'
with h5py.File(file_path, 'r') as file:
    stft = file['STFT Magnitude'][:]


file_path1 = '/home/fankaiqing/DroneRFa_old/data/stft_30w_clean/T1001/train/T1001_D00_S0011/train_102_stft.h5'
with h5py.File(file_path1, 'r') as file:
    stft1 = file['STFT Magnitude'][:]
if stft.shape != stft1.shape:
    print("\n❌ 警告：两个数组的形状不一致，无法进行逐元素比较！")
else:
    print("\n✅ 形状一致，正在进行数值比较...")

    # 2. 检查是否完全一致 (Exact Match)
    is_exact_equal = np.array_equal(stft, stft1)
    print(f"是否完全相等 (np.array_equal): {is_exact_equal}")

    # 3. 检查是否在浮点误差允许范围内一致 (Floating-point tolerance)
    # 默认容差：atol=1e-08, rtol=1e-05
    is_close = np.allclose(stft, stft1)
    print(f"是否近似相等 (np.allclose)  : {is_close}")
