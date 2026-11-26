import ast
import os
import glob
import time
import torch
import numpy as np
import configparser
from Dataset import Dataset
from ResNet_model import ResNetSTFT
from torch.utils.data import DataLoader

# ====== 1. 读取配置文件 ======
config = configparser.ConfigParser()
config.read('config.ini')

# ====== 2. 评测核心配置区 ======
NUM_REPEATS = 1000  # 重复推理次数（用于统计时延分布）
WARMUP_ROUNDS = 20  # 预热次数（消除 GPU 算子冷启动与显存分配开销）
BATCH_SIZE = 1  # 设为 1 表示单样本核心时延测试，可根据需要调整


# ==============================


def load_benchmark_sample(test_loader):
    """
    从 DataLoader 中提取第一个真实的 Batch 作为基准测试输入，
    确保输入张量的 Topology（形状、数据类型）与生产环境完全一致。
    """
    for data, _ in test_loader:
        # shape: (BATCH_SIZE, Channel, Freq, Time)
        sample = data.float()
        # 如果硬件测试要求严格的单样本，强制截取
        if sample.size(0) != BATCH_SIZE and BATCH_SIZE == 1:
            sample = sample[0:1]
        elif sample.size(0) != BATCH_SIZE:
            # 如果配置了更大的 batch_size，且第一组数据不够，则进行填充
            repeat_times = (BATCH_SIZE // sample.size(0)) + 1
            sample = sample.repeat(repeat_times, 1, 1, 1)[:BATCH_SIZE]

        print(f"成功构建基准输入张量，形状 (Input Shape): {tuple(sample.shape)}")
        return sample
    raise ValueError("测试集 DataLoader 为空，无法提取样本！")


def set_dataloader():
    """复用原有逻辑，用于抽取真实测试样本"""
    import joblib
    path_to_files = config.get('general', 'path_to_files')
    encoder = joblib.load(config.get('general', 'path_to_encoder'))
    label_allow_list = encoder.classes_

    test_file_paths, test_labels = [], []
    for parent_path in glob.glob(path_to_files):
        label = os.path.basename(parent_path)
        if label not in label_allow_list:
            continue
        test_parent_path = os.path.join(parent_path, 'test/*')
        for files_path in glob.glob(test_parent_path):
            h5_file_path = os.path.join(files_path, '*.h5')
            for file_path in glob.glob(h5_file_path):
                test_file_paths.append(file_path)
                test_labels.append(label)

    test_labels = encoder.transform(test_labels)
    test_set = Dataset(test_file_paths, test_labels)
    return DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


@torch.no_grad()
def measure_resnet_latency():
    # 1) 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"推理测试设备 (Device): {device}")

    num_classes = config.getint('general', 'num_classes')
    input_shape = ast.literal_eval(config.get('general', 'input_shape'))

    # 2) 加载模型与权重
    print("正在初始化网络并加载模型权重...")
    model = ResNetSTFT(input_shape, num_classes).to(device)
    model_weights = torch.load(config.get('general', 'path_to_model_loss'), map_location=device)
    model.load_state_dict(model_weights)
    model.eval()
    print("模型加载与状态冻结（eval）完成。")

    # 3) 获取基准真实数据样本
    test_loader = set_dataloader()
    sample = load_benchmark_sample(test_loader).to(device)

    # 4) GPU/CPU 算子冷启动预热
    print(f"\n正在进行核心预热 ({WARMUP_ROUNDS} 次)...")
    for _ in range(WARMUP_ROUNDS):
        _ = model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("预热完毕，硬件流水线已就绪。")

    # 5) 正式进入高性能时延评测循环
    print(f"\n开始正式推理计时 ({NUM_REPEATS} 次)...")
    latencies = []

    if device.type == "cuda":
        # GPU 环境：使用 CUDA Event 硬件时间戳，阻断异步机制造成的测量误差
        for _ in range(NUM_REPEATS):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()  # 确保 GPU 队列清空
            start_event.record()
            _ = model(sample)
            end_event.record()
            torch.cuda.synchronize()  # 强行同步以获取高精度硬件耗时

            elapsed_ms = start_event.elapsed_time(end_event)
            latencies.append(elapsed_ms)
    else:
        # CPU 环境：使用系统级高精度计数器 time.perf_counter
        for _ in range(NUM_REPEATS):
            t0 = time.perf_counter()
            _ = model(sample)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            latencies.append(elapsed_ms)

    # 6) 数理统计与百分位数（Percentile）解析
    latencies = np.array(latencies)
    mean_ms = np.mean(latencies)
    std_ms = np.std(latencies)
    median_ms = np.median(latencies)
    min_ms = np.min(latencies)
    max_ms = np.max(latencies)
    p95_ms = np.percentile(latencies, 95)
    p99_ms = np.percentile(latencies, 99)

    # 7) 打印官方格式标准性能报告
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
    # 吞吐量基于 Batch 级别进行换算，能更真实反映系统处理能力
    print(f"  吞吐量 (Throughput)     : {(1000.0 / mean_ms) * BATCH_SIZE:.2f} samples/sec")
    print("=" * 55)


if __name__ == "__main__":
    measure_resnet_latency()