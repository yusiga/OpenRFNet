from scipy.signal import welch, resample
import glob, os, numpy as np, scipy.io, h5py
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['agg.path.chunksize'] = 10000
import sys
from datetime import datetime

# ========== 日志 ==========
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"preprocess_{datetime.now():%Y%m%d_%H%M%S}.log")


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger(log_file)
sys.stderr = sys.stdout
print(f"[LOGGING] 所有输出已保存至: {log_file}")

# ======================
# 基本配置
# ======================
pic_dir = '../pic/clean_pic'
os.makedirs(pic_dir, exist_ok=True)

# 输出目录：每个 mat 文件拼接后的新 mat 存这里
base_dir = '../data/clean_dataset'
mat_files_paths = glob.glob('../..//single_hover/*.mat')

Fs = 100e6
DT = 0.001  # 1 ms


def debug_plot(x_out, p_frame_max, base_th, fs_gate, save_path, label_for_plot):
    t = np.arange(len(x_out)) / fs_gate * 1e3
    env = np.abs(x_out)
    plt.figure(figsize=(10, 5))
    plt.plot(t, env, color='gray', lw=0.8, label='|x_out| Envelope')
    plt.axhline(base_th, color='r', linestyle='--', lw=1.2, label=f"Noise floor={base_th:.3e}")
    plt.axhline(p_frame_max, color='r', linestyle='-', lw=1.2, label=f"power={p_frame_max:.3e}")
    plt.title(f"{label_for_plot} | Slice Energy Visualization")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[DEBUG] Saved pictest: {save_path}")

# ======================
# 背景噪声阈值
# ======================
def compute_noise_threshold_by_slices(noise_file_path, slice_ms=1.0, q=0.95):
    data = scipy.io.loadmat(noise_file_path)
    I, Q = data['RF0_I'][0], data['RF0_Q'][0]
    x = I + 1j * Q
    slice_len = int(slice_ms * 1e-3 * Fs)
    total = len(x) // slice_len
    all_pm = []
    for i in range(total):
        sl = x[i * slice_len:(i + 1) * slice_len]
        all_pm.append(np.mean(np.abs(sl) ** 2))
    all_pm = np.asarray(all_pm, dtype=np.float64)
    base_th = np.quantile(all_pm, q)
    print(f"[Noise threshold @{int(q * 100)}%] base_th={base_th:.3e}, "
          f"mean={all_pm.mean():.3e}, median={np.median(all_pm):.3e}")
    return base_th


# ======================
# 能量门控
# ======================
def gate_on_downsampled(x_slice, base_th, margin=1):
    pm = np.mean(np.abs(x_slice) ** 2)
    th_burst = base_th * margin
    slice_type = "active" if pm >= th_burst else "background"
    return pm, slice_type


# ======================
# 主循环
# ======================
if __name__ == '__main__':
    base_th = compute_noise_threshold_by_slices('../../single_hover/environment.mat')

    global_total_slices = 0
    global_keep_slices = 0

    for file_index, mat_file_path in enumerate(mat_files_paths):
        try:
            data = scipy.io.loadmat(mat_file_path)
            mat_basename = os.path.splitext(os.path.basename(mat_file_path))[0]
            print(f"\nProcessing {file_index + 1}/{len(mat_files_paths)}: {mat_basename}")
            parts = mat_basename.split('_')
            label = '_'.join(parts[0:2])
            I, Q = data['RF0_I'][0], data['RF0_Q'][0]
            x_full = I + 1j * Q
        except Exception as e:
            print(f"Error reading {mat_file_path}: {e}")
            continue

        slice_len = int(DT * Fs)
        total_slices = len(x_full) // slice_len
        global_total_slices += total_slices

        kept_slices = []
        file_keep_slices = 0

        for i in range(total_slices):
            x_slice = x_full[i * slice_len:(i + 1) * slice_len]
            pm, slice_type = gate_on_downsampled(x_slice, base_th)

            keep = ((label == 'environment' and slice_type == 'background') or
                    (label != 'environment' and slice_type == 'active'))

            # 每 100 个切片画一张图（仅用于可视化）
            if i % 100 == 0:
                iq_folder = os.path.join(pic_dir, label, mat_basename)
                os.makedirs(iq_folder, exist_ok=True)
                iq_path = os.path.join(iq_folder, f"{mat_basename}_slice{i}_IQ.png")
                debug_plot(x_slice, pm, base_th, Fs, iq_path, label)

            if not keep:
                continue

            kept_slices.append(x_slice)
            file_keep_slices += 1

        # 当前文件处理完成：将所有 keep 的切片拼接成一个新的 IQ 序列并保存为 mat
        if file_keep_slices == 0:
            print(f"[WARN] 文件 {mat_basename} 没有任何切片通过门控，跳过保存。")
            continue

        x_kept = np.concatenate(kept_slices)  # 1D 复向量
        I_new = x_kept.real.astype(np.float32)
        Q_new = x_kept.imag.astype(np.float32)

        # 保持和原始 mat 一致的形状（1 x N）
        I_new = I_new[np.newaxis, :]
        Q_new = Q_new[np.newaxis, :]

        label_base_dir = os.path.join(base_dir, label)
        os.makedirs(label_base_dir, exist_ok=True)
        out_path = os.path.join(label_base_dir, f"{mat_basename}.mat")

        scipy.io.savemat(out_path, {'RF0_I': I_new, 'RF0_Q': Q_new})
        keep_ratio = file_keep_slices / total_slices if total_slices > 0 else 0.0
        print(f"[SAVE] {mat_basename}: kept {file_keep_slices}/{total_slices} slices "
              f"({keep_ratio:.2%}) -> {out_path}")

        global_keep_slices += file_keep_slices

    # 全局保留率
    if global_total_slices > 0:
        global_ratio = global_keep_slices / global_total_slices
        print(f"\n[SUMMARY] 全部文件总切片数: {global_total_slices}, "
              f"总保留切片数: {global_keep_slices}, "
              f"总体保留率: {global_ratio:.2%}")
    else:
        print("\n[SUMMARY] 未处理到任何切片。")
