import ast
import glob
import os
import h5py
import joblib
import numpy as np
import torch
import configparser
from torch.utils.data import Dataset, DataLoader
from scipy.io import savemat
from tqdm import tqdm
from ResNet_model import ResNet

# 加载配置
config = configparser.ConfigParser()
config.read('config.ini')

# GPU 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = config.getint('general', 'num_classes')
path_to_files = config.get('general', 'path_to_files')
encoder = joblib.load(config.get('general', 'path_to_encoder_all'))
label_allow_list = encoder.classes_

# ================= 1. 定义 Dataset =================
class TestIQDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        label = self.labels[index]

        # ====== 读取 HDF5 文件 ======
        with h5py.File(file_path, 'r') as file:
            magnitude_norm = file['STFT Magnitude'][:]

        magnitude_tensor = torch.from_numpy(magnitude_norm).float()

        return magnitude_tensor, label


if __name__ == "__main__":
    input_shape = config.get('general', 'input_shape')
    input_shape = ast.literal_eval(input_shape)
    model = ResNet(input_shape, num_classes).to(device)
    checkpoint = torch.load(config.get('general', 'path_to_model'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # --- 扫描所有测试文件 ---
    all_test_files = []
    all_test_labels = []

    print("Scanning test files...")
    for parent_path in glob.glob(path_to_files):
        label = os.path.basename(parent_path)
        if label not in label_allow_list:
            continue

        test_h5_pattern = os.path.join(parent_path, 'test/**/*.h5')
        files = glob.glob(test_h5_pattern, recursive=True)
        all_test_files.extend(files)
        all_test_labels.extend([label] * len(files))

    test_dataset = TestIQDataset(all_test_files, all_test_labels)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # === 优化 3：改用全局列表收集整个 Batch 的 Tensor ===
    all_logits = []
    all_feats = []
    all_labels = []

    print(f"Starting Batch Inference (Total: {len(all_test_files)} samples)...")
    with torch.no_grad():
        # 这里加上了你导入的 tqdm 进度条，体验更好
        for batch_data, batch_labels in tqdm(test_loader, desc="Inferencing"):
            # pin_memory=True 配合 non_blocking=True 可以加速 GPU 传输
            batch_data = batch_data.to(device, non_blocking=True)

            # 模型推理
            feat,logits = model(batch_data)

            # 直接把整个 Batch 的 Tensor 移到 CPU 并收集
            all_logits.append(logits.cpu())
            all_feats.append(feat.cpu())
            all_labels.extend(batch_labels)  # 字符串列表直接 extend

    # 在矩阵层面一次性拼接，极大减少 Python 循环开销
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_feats = torch.cat(all_feats, dim=0).numpy()
    all_labels = np.array(all_labels)  # 转为 numpy 方便做 mask 过滤

    # --- 保存结果 ---
    folder_to_openmax = config.get('general', 'folder_to_open')
    data_folder = os.path.join(folder_to_openmax, 'data')
    os.makedirs(data_folder, exist_ok=True)

    print("\nSaving results to .mat files...")
    for label in label_allow_list:
        # 使用 numpy 的布尔索引（Mask）一次性切出当前类别的所有数据
        class_mask = (all_labels == label)

        cls_logits = all_logits[class_mask]
        cls_feats = all_feats[class_mask]

        if len(cls_logits) > 0:
            data_file = os.path.join(data_folder, f'{label}_data.mat')
            savemat(data_file, {
                'logits': cls_logits,
                'feat': cls_feats
            })
            print(f"[{label}] Saved: {len(cls_logits)} samples")
        else:
            print(f"[{label}] No data found, skipped.")