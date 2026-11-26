import ast
import os
import torch
import glob
import joblib
import time  # 引入用于参考总耗时的 Python 时间库
from Dataset import Dataset
from ResNet_model import ResNet
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib

matplotlib.use('Agg')  # 使用Agg后端
import matplotlib.pyplot as plt
import configparser
from sklearn.model_selection import train_test_split
from collections import defaultdict

config = configparser.ConfigParser()
config.read('config.ini')

# GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
path_to_files = config.get('general', 'path_to_files')
encoder = joblib.load(config.get('general', 'path_to_encoder'))
label_allow_list = encoder.classes_
batch_size = config.getint('general', 'batch_size_for_test')
num_classes = config.getint('general', 'num_classes')
input_shape = config.get('general', 'input_shape')
input_shape = ast.literal_eval(input_shape)
conf = np.zeros((num_classes, num_classes))

label_counter = defaultdict(int)


def set_dataloader():
    test_file_paths, test_labels = [], []

    for parent_path in glob.glob(path_to_files):
        label = os.path.basename(parent_path)
        if label not in label_allow_list:
            continue

        test_parent_path = os.path.join(parent_path, 'test/*')
        for files_path in glob.glob(test_parent_path):
            h5_file_path = os.path.join(files_path, '*.h5')
            for id, file_path in enumerate(glob.glob(h5_file_path)):
                test_file_paths.append(file_path)
                test_labels.append(label)

    test_labels = encoder.transform(test_labels)
    print('test nums:', len(test_labels))

    test_set = Dataset(test_file_paths, test_labels)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return test_loader


if __name__ == '__main__':
    test_loader = set_dataloader()

    model = ResNet(input_shape, num_classes).to(device)
    model_weights = torch.load(config.get('general', 'path_to_model'))
    model.load_state_dict(model_weights)
    model.eval()

    # ========================================================
    # 🌟 新增技术点 1：GPU 算子冷启动预热（消除首次分配开销）
    # ========================================================
    print("正在进行 GPU 算子预热（对齐 ResNetSTFT 形状）...")
    with torch.no_grad():
        # 从测试集抽取一个真实 batch 的数据用于对齐输入张量拓扑
        for dummy_data, _ in test_loader:
            dummy_input = dummy_data.float().to(device)
            break
        # 预热 10 次使其彻底在核心内驻留
        for _ in range(10):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    print("预热完毕，正式进入测试集流计算与时延统计...")

    # ========================================================
    # 🌟 新增技术点 2：高精度硬件计时器构建与计数器初始化
    # ========================================================
    total_pure_inference_time_ms = 0.0
    total_processed_samples = 0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 遍历dataloader
    predict_label_list = []
    true_label_list = []

    for data, targets in test_loader:
        actual_batch_size = data.size(0)
        data, targets = data.to(device), targets.to(device)

        # --------------------------------------------------------
        # 🌟 核心计时区域：只把纯模型的网络前向图包络进来
        # --------------------------------------------------------
        with torch.no_grad():
            torch.cuda.synchronize()  # 确保上一批次流完全出栈
            start_event.record()  # 硬件时间戳：入栈开始

            logits = model(data)

            end_event.record()  # 硬件时间戳：入栈结束
            torch.cuda.synchronize()  # 强行同步，阻断异步机制以获取精准耗时

        # 提取这一批次的硬件纯计算耗时（单位：毫秒）
        batch_time_ms = start_event.elapsed_time(end_event)
        total_pure_inference_time_ms += batch_time_ms
        total_processed_samples += actual_batch_size
        # --------------------------------------------------------

        # 后续数据重构、分类矩阵映射与画图逻辑均不计入时延，以防 I/O 阻塞测速
        _, predicted = torch.max(logits.data, 1)
        predict_label = encoder.inverse_transform(predicted.cpu())
        predict_label_list.append(predict_label)

        true_label = encoder.inverse_transform(targets.cpu())
        true_label_list.append(true_label)
        conf += confusion_matrix(true_label, predict_label, labels=encoder.classes_)

    # ========================================================
    # 🌟 新增技术点 3：双指标（每批次/单样本）时延报告解析与打印
    # ========================================================
    avg_latency_per_batch_ms = total_pure_inference_time_ms / len(test_loader)
    avg_latency_per_sample_ms = total_pure_inference_time_ms / total_processed_samples

    print("\n================ ResNetSTFT 推理时延性能报告 ================")
    print(f"测试样本总数 (Total Samples)       : {total_processed_samples}")
    print(f"总推理批次数 (Total Batches)       : {len(test_loader)} (Batch Size: {batch_size})")
    print(f"网络总计纯算力耗时 (Total GPU Time): {total_pure_inference_time_ms / 1000.0:.4f} 秒")
    print(f"批次平均推理时延 (Avg Batch Latency): {avg_latency_per_batch_ms:.2f} 毫秒 (ms)")
    print(f"单个样本平均推理时延 (Avg Latency/Sample): {avg_latency_per_sample_ms:.4f} 毫秒 (ms)")
    print("==============================================================\n")

    # 原有逻辑：混淆矩阵归一化与绘制
    conf_normalized = conf / conf.sum(axis=1)[:, np.newaxis]

    save_folder = config.get('general', 'folder_to_result')
    os.makedirs(save_folder, exist_ok=True)

    plt.figure(figsize=(15, 12))
    sns.heatmap(conf_normalized, annot=True, cmap='Blues', fmt='.2f',
                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Normalized Confusion Matrix')

    filename = os.path.join(save_folder, config.get('general', 'path_test_result'))
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    predict_label_list = np.concatenate(predict_label_list, axis=0)
    true_label_list = np.concatenate(true_label_list, axis=0)

    # 原有逻辑：分类报告打印
    report = classification_report(true_label_list, predict_label_list, target_names=encoder.classes_, digits=4)
    print("Classification Report:")
    print(report)