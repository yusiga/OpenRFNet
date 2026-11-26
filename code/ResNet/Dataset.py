import numpy as np
import torch
import h5py
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        label = self.labels[index]
        with h5py.File(file_path, 'r') as file:
            stft = file['STFT Magnitude'][:]
        return torch.from_numpy(stft).float(), torch.tensor(label).long()


# class Dataset(data.Dataset):
#     def __init__(self, file_paths, labels):
#         self.file_paths = file_paths
#         self.labels = labels
#
#     def __len__(self):
#         return len(self.file_paths)
#
#     def __getitem__(self, index):
#         file_path = self.file_paths[index]
#         label = self.labels[index]
#
#         # ====== 读取 HDF5 文件 ======
#         with h5py.File(file_path, 'r') as file:
#             IQ_data = file['IQ'][:]
#
#         magnitude = np.sqrt(IQ_data[0]**2 + IQ_data[1]** 2)
#         mat = magnitude.reshape(250, -1)
#         mat = mat[np.newaxis, :, :]
#         # 转换为张量
#         magnitude_tensor = torch.from_numpy(mat).float()
#         return magnitude_tensor, torch.tensor(label).long()