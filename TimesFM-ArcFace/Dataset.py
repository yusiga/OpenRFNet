import numpy as np
import torch
import h5py
from torch.utils import data
from scipy.signal import resample


class Dataset(data.Dataset):
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

        # magnitude = np.sqrt(IQ_data[0]**2 + IQ_data[1]** 2)
        #
        # eps = 1e-10
        # magnitude_db = 20 * np.log10(magnitude + eps)
        #
        # # ====== 4. Min-Max 归一化 (消除远近偏差) ======
        # # 将当前切片的 dB 值缩放到 [0, 1] 区间
        # db_min = np.min(magnitude_db)
        # db_max = np.max(magnitude_db)
        #
        # # 加上 eps 防止所有点数值一样导致除以 0
        # magnitude_norm = (magnitude_db - db_min) / (db_max - db_min)
        #
        # # ====== 5. 整理形状并转为 Tensor ======
        # # 增加通道维度 [1, L]
        magnitude_tensor = torch.from_numpy(magnitude_norm).float()

        return magnitude_tensor, torch.tensor(label).long()
