import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from models import LSTMTeacher, CNNLSTM,TransformerTeacher, AttentionLSTMModel,DualStreamATTLSTM 

class SkeletonTestDataset(Dataset):
    def __init__(self, root_dir, label_map, seq_length=125):

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

        data = pd.read_csv(file_path, header=None).values

        if data.shape[0] < self.seq_length:
            data = np.pad(data, ((0, self.seq_length - data.shape[0]), (0, 0)), 'constant')
        else:
            data = data[:self.seq_length]

        data = torch.tensor(data, dtype=torch.float32)
        return data, label

test_root_dir = "./dataset/dailyandsport/test" 
test_dataset = SkeletonTestDataset(test_root_dir, label_map_daily)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
model = LSTMTeacher(input_size=45,  num_classes=19)
model.load_state_dict(torch.load("checkpoints/daily/lstm_best.pth"))
model = model.cuda()
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
