import torch
import torch.nn as nn
import torch.fft

class LSTM(nn.Module):
    def __init__(self, input_size=75, hidden_size=128, num_layers=3, num_classes=22, dropout=0.5):
        super(LSTM, self).__init__()
        # 多层 LSTM，增加深度
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)

        # 添加一个前馈神经网络（两层全连接网络，带激活函数）
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)

        # 可选的 dropout，用于正则化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # LSTM 的输出
        out, (hn, cn) = self.lstm(x)  # out shape: (batch_size, seq_len, hidden_size)

        # 只取最后一个时间步的输出
        out = out[:, -1, :]  # shape: (batch_size, hidden_size)

        # 全连接层和激活函数
        out = self.fc1(out)
        out = self.activation(out)

        # dropout 用于减小过拟合
        out = self.dropout(out)

        # 最后的分类层
        out = self.fc2(out)
        return out

class RNN(nn.Module):
    def __init__(self, input_size=75, hidden_size=128, num_layers=3, num_classes=22, dropout=0.5):
        super(RNNTeacher, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 多层 RNN
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True, nonlinearity='tanh', dropout=dropout)

        # 添加一个前馈神经网络（两层全连接网络，带激活函数）
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)

        # 可选的 dropout，用于正则化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # shape: (num_layers, batch_size, hidden_size)

        # RNN 的输出
        out, hn = self.rnn(x, h0)  # out shape: (batch_size, seq_len, hidden_size)

        # 只取最后一个时间步的输出
        out = out[:, -1, :]  # shape: (batch_size, hidden_size)

        # 全连接层和激活函数
        out = self.fc1(out)
        out = self.activation(out)

        # dropout 用于减小过拟合
        out = self.dropout(out)

        # 最后的分类层
        out = self.fc2(out)
        return out

class Transformer(nn.Module):
    def __init__(self, input_size=75, hidden_size=128, num_layers=3, num_classes=22, dropout=0.5):
        super(TransformerTeacher, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 输入嵌入层，将输入的特征维度从 input_size 转换为 hidden_size
        self.embedding = nn.Linear(input_size, hidden_size)

        # 定义 Transformer 编码器
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 添加一个前馈神经网络（两层全连接网络，带激活函数）
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)

        # 可选的 dropout，用于正则化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 输入的特征维度转换为 hidden_size 维度
        x = self.embedding(x)  # shape: (batch_size, seq_len, hidden_size)

        # Transformer 输入需要交换维度：shape -> (seq_len, batch_size, hidden_size)
        x = x.permute(1, 0, 2)  # shape: (seq_len, batch_size, hidden_size)

        # Transformer 编码器的输出
        out = self.transformer(x)  # shape: (seq_len, batch_size, hidden_size)

        # 只取最后一个时间步的输出
        out = out[-1, :, :]  # shape: (batch_size, hidden_size)

        # 全连接层和激活函数
        out = self.fc1(out)
        out = self.activation(out)

        # dropout 用于减小过拟合
        out = self.dropout(out)

        # 最后的分类层
        out = self.fc2(out)
        return out

class CNNLSTM(nn.Module):
    def __init__(self, input_size=75, seq_len=200, cnn_channels=64, kernel_size=3,
                 hidden_size=128, num_layers=2, num_classes=22, dropout=0.5):
        super(CNNLSTM, self).__init__()

        # CNN 模块
        self.conv1 = nn.Conv2d(1, cnn_channels, kernel_size=(kernel_size, input_size), padding=(kernel_size // 2, 0))
        self.bn1 = nn.BatchNorm2d(cnn_channels)
        self.relu = nn.ReLU()

        # LSTM 模块
        self.lstm = nn.LSTM(cnn_channels, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)

        # 全连接层
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)

        # 添加一个通道维度，以适应 CNN 输入
        x = x.unsqueeze(1)  # shape: (batch_size, 1, seq_len, input_size)

        # CNN 提取时空特征
        x = self.conv1(x)  # shape: (batch_size, cnn_channels, seq_len, 1)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.squeeze(-1)  # shape: (batch_size, cnn_channels, seq_len)

        # 调整维度以适配 LSTM
        x = x.permute(0, 2, 1)  # shape: (batch_size, seq_len, cnn_channels)

        # LSTM 模块
        out, (hn, cn) = self.lstm(x)  # out shape: (batch_size, seq_len, hidden_size)

        # 取最后一个时间步的输出
        out = out[:, -1, :]  # shape: (batch_size, hidden_size)

        # 全连接层
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)  # shape: (batch_size, num_classes)

        return out

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # [N, in_features] -> [N, out_features]
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size(0)  # number of nodes
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)


class GATModule(nn.Module):
    def __init__(self, num_nodes, in_features, out_features, dropout=0.5, alpha=0.2, n_heads=1):
        super(GATModule, self).__init__()
        self.attention_layers = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout, alpha, concat=True)
            for _ in range(n_heads)
        ])
        self.out_features = out_features
        self.num_nodes = num_nodes

    def forward(self, features, adj):
        x = torch.cat([att(features, adj) for att in self.attention_layers], dim=-1)
        return x


class SparseAttention(nn.Module):
    def __init__(self, hidden_dim, local_context_size=5):
        super(SparseAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.local_context_size = local_context_size
        self.attention = nn.Linear(self.hidden_dim, 1)

    def forward(self, lstm_output):
        seq_len = lstm_output.size(1)
        center_position = seq_len // 2
        start_position = max(0, center_position - self.local_context_size // 2)
        end_position = min(seq_len, start_position + self.local_context_size)

        local_output = lstm_output[:, start_position:end_position, :]
        attention_weights = torch.relu(self.attention(local_output))

        padded_attention_weights = torch.zeros_like(lstm_output[:, :, 0])
        padded_attention_weights[:, start_position:end_position] = attention_weights.squeeze(dim=-1)

        attention_weights = torch.softmax(padded_attention_weights, dim=1)

        context_vector = attention_weights.unsqueeze(dim=-1) * lstm_output
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, attention_weights


class AttentionLSTMModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_dim_1=128, hidden_dim_2=64, dropout=0.5, num_nodes=25):
        super(AttentionLSTMModel, self).__init__()

        self.lstm_1 = nn.LSTM(input_size, hidden_dim_1, batch_first=True)
        self.sattention_1 = SparseAttention(hidden_dim_1)
        self.fc_1 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.activation_1 = nn.ReLU()

        self.gat = GATModule(num_nodes, hidden_dim_2, hidden_dim_2, dropout=dropout)

        self.lstm_2 = nn.LSTM(hidden_dim_2, hidden_dim_2, batch_first=True)
        self.sattention_2 = SparseAttention(hidden_dim_2)
        self.fc_2 = nn.Linear(hidden_dim_2, hidden_dim_2)
        self.activation_2 = nn.ReLU()

        self.fc1 = nn.Linear(hidden_dim_2, hidden_dim_2 // 2)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim_2 // 2, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x, _ = self.lstm_1(x)
        x, _ = self.sattention_1(x)
        x = self.activation_1(self.fc_1(x))

        # GAT for spatial attention
        gat_input = x[:, -1, :]  # Use the last time step
        gat_output = self.gat(gat_input, adj)

        x = gat_output.unsqueeze(1)  # Add sequence dimension back
        x, _ = self.lstm_2(x)
        x, _ = self.sattention_2(x)
        x = self.activation_2(self.fc_2(x))

        x = self.fc1(x)
        x = self.activation(x)

        x = self.dropout(x)
        x = self.fc2(x)

        return x


class DualStreamATTLSTM(nn.Module):
    def __init__(self, input_size, num_classes, hidden_dim_1=128, hidden_dim_2=64, dropout = 0.5):
        super(DualStreamATTLSTM, self).__init__()

        self.lstm_1 = nn.LSTM(input_size, hidden_dim_1, batch_first=True)
        self.sattention_1 = SparseAttention(hidden_dim_1)
        self.fc_1 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.activation_1 = nn.ReLU()

        self.lstm_2 = nn.LSTM(hidden_dim_2, hidden_dim_2, batch_first=True)
        self.sattention_2 = SparseAttention(hidden_dim_2)
        self.fc_2 = nn.Linear(hidden_dim_2, hidden_dim_2)
        self.activation_2 = nn.ReLU()


        self.fc1 = nn.Linear(hidden_dim_2*2, hidden_dim_2)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim_2, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #时域流
        out_time, _ = self.lstm_1(x)
        out_time, _ = self.sattention_1(out_time)
        out_time = self.activation_1(self.fc_1(out_time))

        out_time = out_time.unsqueeze(1)  # 因为我们已经得到了context_vector，需要增加一个seq_len维度以送入第二个LSTM
        out_time, _ = self.lstm_2(out_time)
        out_time, _ = self.sattention_2(out_time)
        out_time = self.activation_2(self.fc_2(out_time))
        # 频域流
        freq_features = torch.fft.rfft(x, dim=1)  # 快速傅里叶变换
        freq_features = freq_features.real  # 取实部（也可以选择使用复数形式）
        out_freq, _ = self.lstm_1(freq_features)
        out_freq, _ = self.sattention_1(out_freq)
        out_freq = self.activation_1(self.fc_1(out_freq))

        out_freq = out_freq.unsqueeze(1)  # 因为我们已经得到了context_vector，需要增加一个seq_len维度以送入第二个LSTM
        out_freq, _ = self.lstm_2(out_freq)
        out_freq, _ = self.sattention_2(out_freq)
        out_freq = self.activation_2(self.fc_2(out_freq))

        out = torch.cat((out_time, out_freq), dim=1)  # 拼接时域和频域特征

        out = self.fc1(out)
        out = self.activation(out)

        # dropout 用于减小过拟合
        out = self.dropout(out)

        # 最后的分类层
        out = self.fc2(out)
        return out

