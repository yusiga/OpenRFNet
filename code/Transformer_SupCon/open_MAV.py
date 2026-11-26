import ast
import os
import glob
import h5py
import joblib
from scipy.io import savemat
import numpy as np
import torch
from Sup_TransResNet_model import TransResNet
import scipy.spatial.distance as spd
import configparser
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # 引入进度条提高交互体验

config = configparser.ConfigParser()
config.read('config.ini')

encoder = joblib.load(config.get('general', 'path_to_encoder'))
label_allow_list = encoder.classes_
# GPU 属性配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = config.getint('general', 'num_classes')
input_shape = config.get('general', 'input_shape')
input_shape = ast.literal_eval(input_shape)
model = TransResNet(input_shape, num_classes).to(device)
checkpoint = torch.load(config.get('general', 'path_to_ce'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


class IQDataset(Dataset):
    def __init__(self, file_paths, labels_encoded):
        self.file_paths = file_paths
        self.labels = labels_encoded  # 优化 1：内部直接存储数字化编码

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        label = self.labels[index]

        with h5py.File(file_path, 'r') as file:
            magnitude_norm = file['STFT Magnitude'][:]

        magnitude_tensor = torch.as_tensor(magnitude_norm, dtype=torch.float32)
        return magnitude_tensor, label


def save_data(correct_data, name, label_name):
    correct_data = np.array(correct_data)

    # # 修复 Bug 1：对每个特征向量按行(axis=1)单独归一化，最契合 ArcFace 的超球面特性
    # row_norms = np.linalg.norm(correct_data, axis=1, keepdims=True) + 1e-10
    # correct_data_norm = correct_data / row_norms

    # 1. 计算 MAV (Mean Activation Vector) 并对中心向量再次归一化
    mean_data = np.mean(correct_data, axis=0)
    # mean_data_norm = mean_data / (np.linalg.norm(mean_data) + 1e-10)

    folder_to_openmax = config.get('general', 'folder_to_open')

    # 保存 MAV
    mav_folder = f'{folder_to_openmax}/{name}/MAV'
    os.makedirs(mav_folder, exist_ok=True)
    mav_file = os.path.join(mav_folder, f'{label_name}_mav.mat')
    savemat(mav_file, {'mav': mean_data})
    print(f"[{label_name}] MAV saved successfully.")

    # # 2. 批量计算特征向量到 MAV 的余弦距离
    # cosine_distances = []
    # for data_vector in correct_data:
    #     dist_cosine = spd.cosine(data_vector, mean_data)
    #     cosine_distances.append(dist_cosine)
    #
    # # 3. 创建保存目录并存储
    # distance_folder = f'{folder_to_openmax}/{name}/distances'
    # os.makedirs(distance_folder, exist_ok=True)
    #
    # distance_path = os.path.join(distance_folder, f'{label_name}_distances.mat')
    # savemat(distance_path, {
    #     'cosine': np.array(cosine_distances)
    # })
    #
    # print(f"[{label_name}] Cosine distances saved.")


if __name__ == "__main__":
    all_file_paths = []
    all_labels_str = []

    print("Scanning files and building dataset...")
    path_to_files = config.get('general', 'path_to_files')
    for parent_path in glob.glob(path_to_files):
        label = os.path.basename(parent_path)
        if label not in label_allow_list:
            continue

        h5_pattern = os.path.join(parent_path, 'train/**/*.h5')
        files = glob.glob(h5_pattern, recursive=True)
        all_file_paths.extend(files)
        all_labels_str.extend([label] * len(files))

    # 在进入 Dataset 前，将字符串标签转换为整型编码，防止多线程跨进程卡顿
    all_labels_encoded = encoder.transform(all_labels_str)

    dataset = IQDataset(all_file_paths, all_labels_encoded)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # 优化 3：改用全局列表收集所有 Batch 的推断结果
    collected_embeddings = []
    collected_true_labels = []
    collected_pred_labels = []

    print(f"Starting Batch Inference (Total: {len(all_file_paths)} samples)...")

    with torch.no_grad():
        for batch_data, batch_labels in tqdm(loader, desc="Inferencing"):
            batch_data = batch_data.to(device, non_blocking=True)

            feat,logits = model(batch_data)

            # 获取预测的类别对应索引
            predicted_idx = torch.argmax(logits, dim=1)

            # 收集 Tensor
            collected_embeddings.append(feat.cpu())
            collected_true_labels.append(batch_labels)
            collected_pred_labels.append(predicted_idx.cpu())

    # 一次性矩阵拼接
    all_embeddings = torch.cat(collected_embeddings, dim=0).numpy()
    all_true_labels = torch.cat(collected_true_labels, dim=0).numpy()
    all_pred_labels = torch.cat(collected_pred_labels, dim=0).numpy()

    # 找出模型分类正确的全局布尔掩码
    correct_mask = (all_true_labels == all_pred_labels)

    # 3. 统一计算各类的 MAV 并保存
    print("\nCalculating MAVs and saving distances...")
    for idx, label_name in enumerate(label_allow_list):
        # 掩码条件：属于当前类别，且预测正确
        class_mask = all_true_labels == idx
        # class_mask = (all_true_labels == idx) & correct_mask
        correct_feats = all_embeddings[class_mask]

        if len(correct_feats) > 0:
            save_data(correct_feats, 'feat', label_name)
        else:
            print(f"Warning: No correct predictions for class {label_name}, skipping.")