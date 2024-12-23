import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from models import LSTMTeacher, CNNLSTM,TransformerTeacher, AttentionLSTMModel,DualStreamATTLSTM # 确保您的模型定义文件路径正确

class SkeletonTestDataset(Dataset):
    def __init__(self, root_dir, label_map, seq_length=125):
        """
        Args:
            root_dir (str): 测试数据集的根目录，包含 22 个动作文件夹。
            label_map (dict): 动作类别到标签的映射。
            seq_length (int): 每个样本的时间序列长度。
        """
        self.samples = []
        self.labels = []
        self.label_map = label_map
        self.seq_length = seq_length

        for action, label in label_map.items():
            action_dir = os.path.join(root_dir, action)
            if os.path.exists(action_dir):
                for file_name in os.listdir(action_dir):
                    if file_name.endswith('.csv'):
                        file_path = os.path.join(action_dir, file_name)
                        self.samples.append(file_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        label = self.labels[idx]

        # 加载 CSV 文件并转换为张量
        data = pd.read_csv(file_path, header=None).values

        # 确保数据长度一致：小于 seq_length 补零，大于则截断
        if data.shape[0] < self.seq_length:
            data = np.pad(data, ((0, self.seq_length - data.shape[0]), (0, 0)), 'constant')
        else:
            data = data[:self.seq_length]

        data = torch.tensor(data, dtype=torch.float32)
        return data, label

# 定义动作类别与标签的映射
label_map_utd = {
    "a1": 0, "a10": 1, "a11": 2, "a12": 3, "a13": 4, "a14": 5, "a15": 6, "a16": 7, "a17": 8, "a18": 9,
    "a19": 10, "a2": 11, "a20": 12, "a21": 13, "a22": 14, "a23": 15, "a24": 16, "a25": 17, "a26": 18,
    "a27": 19, "a3": 20, "a4": 21, "a5": 22, "a6": 23, "a7": 24, "a8": 25, "a9": 26
}
label_map_czu = {
    "a1": 0, "a10": 1, "a11": 2, "a12": 3, "a13": 4, "a14": 5, "a15": 6, "a16": 7, "a17": 8, "a18": 9,
    "a19": 10, "a2": 11, "a20": 12, "a21": 13, "a22": 14, "a3": 15, "a4": 16, "a5": 17, "a6": 18,
    "a7": 19, "a8": 20, "a9": 21
}
label_map_asl = {
    "a": 0, "b": 1, "bad": 2, "c": 3, "d": 4, "deaf": 5, "e": 6, "f": 7, "fine": 8,
    "g": 9, "good": 10, "goodbye": 11, "h": 12, "hello": 13, "hungry": 14, "i": 15,
    "j": 16, "k": 17, "l": 18, "m": 19, "me": 20, "n": 21, "no": 22, "o": 23, "p": 24,
    "please": 25, "q": 26, "r": 27, "s": 28, "sorry": 29, "t": 30, "thankyou": 31,
    "u": 32, "v": 33, "w": 34, "x": 35, "y": 36, "yes": 37, "you": 38, "z": 39
}
label_map_daily = {
    "a1": 0, "a02": 1, "a03": 2, "a04": 3, "a05": 4, "a06": 5, "a07": 6, "a08": 7, "a09": 8, "a10": 9,
    "a11": 10, "a12": 11, "a13": 12, "a14": 13, "a15": 14, "a16": 15, "a17": 16, "a18": 17, "a19": 18
}

# 测试数据集路径
test_root_dir = "./dataset/dailyandsport/test"  # 替换为您的测试集根目录路径
test_dataset = SkeletonTestDataset(test_root_dir, label_map_daily)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 加载模型
model = LSTMTeacher(input_size=45,  num_classes=19)
model.load_state_dict(torch.load("checkpoints/daily/lstm_best.pth"))
model = model.cuda()
model.eval()

# 测试模型
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.cuda(), labels.cuda()

        # 前向传播
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        # 统计准确率
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
