import ast
import os
import glob
import h5py
import joblib
from scipy.io import savemat
import numpy as np
import torch
from My_TimesFM_lora import TimesFMForClassification
import scipy.spatial.distance as spd
import configparser
from pytorch_metric_learning import losses

config = configparser.ConfigParser()
config.read('config.ini')
from torch.utils.data import DataLoader, Dataset

encoder = joblib.load(config.get('general', 'path_to_encoder'))
label_allow_list = encoder.classes_
# GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = config.getint('general', 'num_classes')
input_shape = config.get('general', 'input_shape')
input_shape = ast.literal_eval(input_shape)

criterion = losses.ArcFaceLoss(
    num_classes=num_classes,
    embedding_size=64,
    margin=28.6,  # 角度间隔
    scale=32  # 缩放因子
).to(device)
model = TimesFMForClassification(num_classes, device=device).to(device)
checkpoint = torch.load(config.get('general', 'path_to_model_stage2'))
model.load_state_dict(checkpoint['model_state_dict'])
criterion.load_state_dict(checkpoint['criterion_state_dict'])
model.eval()


class IQDataset(Dataset):
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


def save_data(correct_data, name, label_name):
    # 确保是 numpy 数组
    correct_data = np.array(correct_data)
    correct_data_norm = correct_data / (np.linalg.norm(correct_data) + 1e-10)
    # 1. 计算 MAV (Mean Activation Vector)
    mean_data = np.mean(correct_data_norm, axis=0)
    mean_data_norm = mean_data / (np.linalg.norm(mean_data) + 1e-10)

    folder_to_openmax = config.get('general', 'folder_to_openmax')

    # 保存 MAV (保存归一化后的版本，最契合 ArcFace)
    mav_folder = f'{folder_to_openmax}/{name}/MAV'
    os.makedirs(mav_folder, exist_ok=True)
    mav_file = os.path.join(mav_folder, f'{label_name}_mav.mat')
    savemat(mav_file, {'mav': mean_data_norm})
    print(f"{label_name} ({name}) MAV has saved")

    # 2. 同时计算两种距离
    cosine_distances = []

    for data_vector in correct_data_norm:

        # 计算余弦距离 (1 - cos_sim)
        dist_cosine = spd.cosine(data_vector, mean_data_norm)
        cosine_distances.append(dist_cosine)

    # 3. 创建保存目录并存储
    distance_folder = f'{folder_to_openmax}/{name}/distances'
    os.makedirs(distance_folder, exist_ok=True)

    # 将两种距离都放入字典保存到同一个 mat 文件
    distance_path = os.path.join(distance_folder, f'{label_name}_distances.mat')
    savemat(distance_path, {
        'cosine': cosine_distances
    })

    print(f"Saved Euclidean & Cosine distances ({name}) for {label_name}")


if __name__ == "__main__":
    all_file_paths = []
    all_labels_str = []

    print("Scanning files and building dataset...")
    path_to_files = config.get('general', 'path_to_files')
    for parent_path in glob.glob(path_to_files):
        label = os.path.basename(parent_path)
        if label not in label_allow_list:
            continue

        # 使用递归匹配所有 h5，避免多层循环
        h5_pattern = os.path.join(parent_path, 'train/**/*.h5')
        files = glob.glob(h5_pattern, recursive=True)
        all_file_paths.extend(files)
        all_labels_str.extend([label] * len(files))


    dataset = IQDataset(all_file_paths, all_labels_str)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    feat_bank = {name: {'correct': [], 'high': []} for name in label_allow_list}

    model.eval()
    print(f"Starting Batch Inference (Total: {len(all_file_paths)} samples)...")

    with torch.no_grad():
        for batch_data, batch_labels in loader:
            batch_data = batch_data.to(device)

            embeddings = model(batch_data)
            logits = criterion.get_logits(embeddings)

            probs = torch.nn.functional.softmax(logits, dim=1)
            conf, predicted_idx = torch.max(probs, 1)

            embeddings_np = embeddings.cpu().numpy()
            predicted_labels = encoder.inverse_transform(predicted_idx.cpu().numpy())

            for i in range(len(batch_labels)):
                true_label = batch_labels[i]  # 这里的 true_label 是字符串
                pred_label = predicted_labels[i]

                if pred_label == true_label:
                    feat_bank[true_label]['correct'].append(embeddings_np[i])

    # 3. 统一计算各类的 MAV 并保存
    print("\nCalculating MAVs and saving distances...")
    for label_name in label_allow_list:
        correct_feats = feat_bank[label_name]['correct']

        if correct_feats:
            save_data(correct_feats, 'feat64', label_name)
        else:
            print(f"Warning: No correct predictions for class {label_name}, skipping.")
