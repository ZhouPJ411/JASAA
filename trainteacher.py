import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
from kinectdataset import SkeletonDataset
from models import LSTMTeacher,RNNTeacher,TransformerTeacher, CNNLSTM, AttentionLSTMModel, DualStreamATTLSTM
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# 数据集
train_dataset = SkeletonDataset(csv_folder="./dataset\dailyandsport/train")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

csv_folder = "./dataset/dailyandsport/train"  # 替换为您的训练数据集路径
subfolders = os.listdir(csv_folder)
# subfolders.sort()  # 确保顺序是固定的（按字母顺序）
label_map = {label: subfolder for label, subfolder in enumerate(subfolders)}
print("动作标签映射关系：")
for label, subfolder in label_map.items():
    print(f"动作名称: {subfolder} -> 标签: {label}")
# 模型
model = TransformerTeacher(input_size=45, num_classes=19)
model = model.cuda()  # 使用 GPU（如果可用）

# 损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0007)

# 训练过程
num_epochs = 1000
first_epoch = 0
best_acc=0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 统计准确率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")



print("Training finished")
