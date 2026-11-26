import os
import glob
import time
import torch
import numpy as np
import configparser
from TimesFM_lora import TimesFMForClassification
import h5py

config = configparser.ConfigParser()
config.read('config.ini')

# ====== 配置区 ======
NUM_REPEATS = 100          # 重复推理次数（可根据需要调大/调小）
WARMUP_ROUNDS = 10         # 预热次数（前几次不计入统计，消除冷启动影响）
BATCH_SIZE = 1             # 单样本推理（衡量单次推理时延）
# ====================


def load_single_sample():
    """
    从数据集中找到第一个可用的 h5 文件，读取并返回一个样本 tensor。
    返回 shape: (1, 1, L)  即 (batch=1, channel=1, length)
    """
    path_to_files = '../data_hover/rf_data_1ms_norm/*'

    # 遍历找到第一个 h5 文件
    for parent_path in glob.glob(path_to_files):
        test_parent_path = os.path.join(parent_path, 'train/*')
        for files_path in glob.glob(test_parent_path):
            h5_file_path = os.path.join(files_path, '*.h5')
            for file_path in glob.glob(h5_file_path):
                print(f"加载样本文件: {file_path}")
                with h5py.File(file_path, 'r') as f:
                    magnitude_norm = f['IQ Magnitude'][:]
                # shape: (1, 1, L)
                sample = torch.from_numpy(magnitude_norm).float().unsqueeze(0)
                print(f"样本形状: {sample.shape}")
                return sample

    raise FileNotFoundError("未找到任何 h5 数据文件，请检查 config.ini 中的 path_to_files 路径")


@torch.no_grad()
def measure_latency():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"推理设备: {device}")

    num_classes = config.getint('general', 'num_classes')

    # 1) 加载模型
    print("正在加载模型...")
    model = TimesFMForClassification(num_classes, device=str(device)).to(device)
    checkpoint = torch.load(
        config.get('general', 'path_to_model_stage2'),
        map_location=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("模型加载完毕。")

    # 2) 加载单个样本
    sample = load_single_sample().to(device)

    # 3) 预热（消除首次推理的编译/缓存开销）
    print(f"\n正在预热 ({WARMUP_ROUNDS} 次)...")
    for _ in range(WARMUP_ROUNDS):
        _ = model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("预热完毕。")

    # 4) 正式计时
    print(f"\n开始正式推理计时 ({NUM_REPEATS} 次)...")
    latencies = []

    if device.type == "cuda":
        # GPU：使用 CUDA Event 精确计时
        for i in range(NUM_REPEATS):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start_event.record()
            _ = model(sample)
            end_event.record()
            torch.cuda.synchronize()

            elapsed_ms = start_event.elapsed_time(end_event)
            latencies.append(elapsed_ms)
    else:
        # CPU：使用 time.perf_counter 高精度计时
        for i in range(NUM_REPEATS):
            t0 = time.perf_counter()
            _ = model(sample)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0  # 转为毫秒
            latencies.append(elapsed_ms)

    # 5) 统计
    latencies = np.array(latencies)
    mean_ms = np.mean(latencies)
    std_ms = np.std(latencies)
    median_ms = np.median(latencies)
    min_ms = np.min(latencies)
    max_ms = np.max(latencies)
    p95_ms = np.percentile(latencies, 95)
    p99_ms = np.percentile(latencies, 99)

    # 6) 输出报告
    print("\n" + "=" * 55)
    print("          推理时延性能报告 (Inference Latency)")
    print("=" * 55)
    print(f"  设备 (Device)           : {device}")
    print(f"  模型 (Model)            : {model.__class__.__name__}")
    print(f"  输入形状 (Input Shape)  : {tuple(sample.shape)}")
    print(f"  预热次数 (Warmup)       : {WARMUP_ROUNDS}")
    print(f"  测量次数 (Repeats)      : {NUM_REPEATS}")
    print("-" * 55)
    print(f"  平均时延 (Mean)         : {mean_ms:.4f} ms")
    print(f"  标准差 (Std)            : {std_ms:.4f} ms")
    print(f"  中位数 (Median)         : {median_ms:.4f} ms")
    print(f"  最小值 (Min)            : {min_ms:.4f} ms")
    print(f"  最大值 (Max)            : {max_ms:.4f} ms")
    print(f"  P95                     : {p95_ms:.4f} ms")
    print(f"  P99                     : {p99_ms:.4f} ms")
    print("-" * 55)
    print(f"  吞吐量 (Throughput)     : {1000.0 / mean_ms:.2f} samples/sec")
    print("=" * 55)


if __name__ == "__main__":
    measure_latency()
