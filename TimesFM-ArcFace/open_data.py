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
from My_TimesFM_lora import TimesFMForClassification
from pytorch_metric_learning import losses

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
            magnitude_norm = file['IQ Magnitude'][:]

        magnitude_tensor = torch.from_numpy(magnitude_norm).float()

        return magnitude_tensor, label



if __name__ == "__main__":
    criterion = losses.ArcFaceLoss(
        num_classes=num_classes,
        embedding_size=64,
        margin=28.6,  # 角度间隔
        scale=16  # 缩放因子
    ).to(device)
    model = TimesFMForClassification(num_classes, device=device).to(device)
    checkpoint = torch.load(config.get('general', 'path_to_model_stage2'))
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion.load_state_dict(checkpoint['criterion_state_dict'])
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


    results_bank = {name: {'logits': [], 'feat64': []} for name in label_allow_list}

    print(f"Starting Batch Inference (Total: {len(all_test_files)} samples)...")
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)

            # 模型推理
            embeddings = model(batch_data)
            logits = criterion.get_logits(embeddings)

            logits_np = logits.cpu().numpy()
            feats_np = embeddings.cpu().numpy()

            for i in range(len(batch_labels)):
                target_label = batch_labels[i]
                results_bank[target_label]['logits'].append(logits_np[i])
                results_bank[target_label]['feat64'].append(feats_np[i])

    # --- 保存结果 ---
    folder_to_openmax = config.get('general', 'folder_to_openmax')
    data_folder = os.path.join(folder_to_openmax, 'data')
    os.makedirs(data_folder, exist_ok=True)

    print("\nSaving results to .mat files...")
    for label in label_allow_list:
        logits_list = results_bank[label]['logits']
        feat64_list = results_bank[label]['feat64']

        if logits_list:
            data_file = os.path.join(data_folder, f'{label}_data.mat')
            savemat(data_file, {
                'logits': np.array(logits_list),
                'feat64': np.array(feat64_list)
            })
            print(f"[{label}] Saved: {len(logits_list)} samples")
        else:
            print(f"[{label}] No data found, skipped.")