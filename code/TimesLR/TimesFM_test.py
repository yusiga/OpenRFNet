import ast
import os
import torch
import glob
import joblib
import time  # 引入 Python 时间库用于计算总耗时参考
from Dataset import Dataset
from TimesFM_ArcFace_lora import TimesFMForClassification
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib
from pytorch_metric_learning import losses

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
conf = np.zeros((num_classes, num_classes))
label_counter = defaultdict(int)

criterion = losses.ArcFaceLoss(
    num_classes=num_classes,
    embedding_size=64,
    margin=28.6,
    scale=16
).to(device)


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
    # 注意：算单个样本平均时间时，建议保持 shuffle=False，drop_last=False 确保统计完整性
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return test_loader


if __name__ == '__main__':
    test_loader = set_dataloader()
    model = TimesFMForClassification(num_classes, device=device).to(device)
    checkpoint = torch.load(config.get('general', 'path_to_model'))
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion.load_state_dict(checkpoint['criterion_state_dict'])
    model.eval()

    # ==========================================
    # 🌟 新增：GPU 基准预热（消除首次冷启动误差）
    # ==========================================
    print("正在进行 GPU 算子预热...")
    with torch.no_grad():
        dummy_input = torch.randn(batch_size, 1, 100000).to(device)
        for _ in range(10):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    print("预热完毕，开始正式推理与时延统计...")

    # 用于累加纯网络推理时间的变量（单位：毫秒）
    total_pure_inference_time_ms = 0.0
    total_processed_samples = 0

    # 遍历dataloader
    predict_label_list = []
    true_label_list = []

    # 构造 PyTorch 专用的 GPU 计时器
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for data, targets in test_loader:
        actual_batch_size = data.size(0)
        data, targets = data.to(device), targets.to(device)

        # ==========================================
        # 🌟 新增：精准记录这一批次的纯网络推理时延
        # ==========================================
        with torch.no_grad():
            torch.cuda.synchronize()  # 确保之前的流操作已清空
            start_event.record()  # 记录开始时间点

            embeddings = model(data)
            logits = criterion.get_logits(embeddings)

            end_event.record()  # 记录结束时间点
            torch.cuda.synchronize()  # 等待当前推理在硬件上真正执行完毕

        # 计算该批次在 GPU 上的挂钟耗时（单位：毫秒）
        batch_time_ms = start_event.elapsed_time(end_event)
        total_pure_inference_time_ms += batch_time_ms
        total_processed_samples += actual_batch_size

        # 后续的标签解析与矩阵统计（不计入推理时延内，保证耗时纯净）
        _, predicted = torch.max(logits.data, 1)
        predict_label = encoder.inverse_transform(predicted.cpu())
        predict_label_list.append(predict_label)

        true_label = encoder.inverse_transform(targets.cpu())
        true_label_list.append(true_label)
        conf += confusion_matrix(true_label, predict_label, labels=encoder.classes_)

    # ==========================================
    # 🌟 新增：推理时延最终指标计算与打印
    # ==========================================
    avg_latency_per_batch_ms = total_pure_inference_time_ms / len(test_loader)
    avg_latency_per_sample_ms = total_pure_inference_time_ms / total_processed_samples

    print("\n================ 推理时延性能报告 ================")
    print(f"测试样本总数 (Total Samples): {total_processed_samples}")
    print(f"总推理批次数 (Total Batches): {len(test_loader)} (Batch Size: {batch_size})")
    print(f"纯网络总耗时 (Total GPU Time): {total_pure_inference_time_ms / 1000.0:.4f} 秒")
    print(f"每批次平均时延 (Avg Latency per Batch): {avg_latency_per_batch_ms:.2f} 毫秒 (ms)")
    print(f"单样本平均时延 (Avg Latency per Sample): {avg_latency_per_sample_ms:.4f} 毫秒 (ms)")
    print("==================================================\n")

    # 绘制混淆矩阵
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

    # 计算分类报告
    report = classification_report(true_label_list, predict_label_list, target_names=encoder.classes_, digits=4)
    print("Classification Report:")
    print(report)