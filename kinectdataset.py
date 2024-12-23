import numpy as np
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.data import Dataset as GeometricDataset

class SkeletonDataset(Dataset):
    def __init__(self, csv_folder, seq_length=125):
        """
        csv_folder: 包含 CSV 文件的文件夹路径
        seq_length: 每个样本的时间序列长度
        """
        self.csv_folder = csv_folder
        self.seq_length = seq_length
        self.files = []
        self.labels = []

        for label, subfolder in enumerate(os.listdir(csv_folder)):
            subfolder_path = os.path.join(csv_folder, subfolder)
            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    if file.endswith(".csv"):
                        self.files.append(os.path.join(subfolder_path, file))
                        self.labels.append(label)  # 每个子文件夹代表一个动作标签

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]

        # 读取 CSV 文件，跳过表头
        data = pd.read_csv(file_path)
        # skeleton_data = data.iloc[1:, :].values  # 去掉帧数列，提取骨骼数据
        skeleton_data = data.values

        # 确保数据长度一致，如果小于 seq_length 补零，否则截断
        if skeleton_data.shape[0] < self.seq_length:
            skeleton_data = np.pad(skeleton_data, ((0, self.seq_length - skeleton_data.shape[0]), (0, 0)), 'constant')
        else:
            skeleton_data = skeleton_data[:self.seq_length]

        skeleton_data = torch.tensor(skeleton_data, dtype=torch.float32)

        return skeleton_data, label


