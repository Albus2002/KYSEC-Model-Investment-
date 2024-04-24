import numpy as np
import math
import pandas as pd
import tracemalloc
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import weight_norm
import matplotlib.pyplot as plt


#基类对象
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError
    


#CNN + LSTM拼接
class CNNLSTMModel(BaseModel):
    def __init__(self, input_channels, feature_dim, time_step, output_size, input_size, hidden_size, num_layers, dropout_rate):
        super(CNNLSTMModel, self).__init__()
        self.feature_dim = feature_dim
        self.time_step = time_step
        self.num_layers = num_layers
        self.input_size = input_size

        self.conv1 = nn.Conv1d(in_channels = input_channels, out_channels = feature_dim, kernel_size = 3, stride = 1, padding = 1)
        self.maxpool = nn.MaxPool1d(kernel_size = 2, stride=2)

        self.output_height = (((time_step - 3 + 2 * 1) // 1 + 1) - 2 + 2 * 0) // 2 + 1 # 30
        self.output_length = (((feature_dim - 3 + 2 * 1) // 1 + 1) - 2 + 2 * 0 ) // 2 + 1 # 17
        print(self.output_height)
        print(self.output_length)
        self.output_feature_dim = input_channels
        self.lstm = nn.LSTM(input_size = self.output_height * self.feature_dim, 
                            hidden_size = hidden_size, num_layers = self.num_layers, 
                            batch_first=True, dropout = dropout_rate)
        
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.maxpool(x)
        # Flatten the output for the LSTM
        x = x.reshape(x.size(0), -1, self.feature_dim * (self.time_step // 2))
        x, (hn, cn) = self.lstm(x)
        x = self.fc1(x[:, -1, :])
        return x
# TCN模型及相关函数
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNModel(BaseModel):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = self.tcn(x.transpose(1, 2))
        return self.linear(x[:, :, -1])

# BiATCN模型

class BiATCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout, hidden_size, num_layers):
        super(BiATCN, self).__init__()
        # 定义TCN部分
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        
        # 定义双向LSTM部分
        self.lstm = nn.LSTM(input_size=num_channels[-1], hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        
        # 定义注意力机制部分
        self.attention = nn.Linear(2 * hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

        # 定义输出层
        self.fc = nn.Linear(2 * hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.tcn(x.transpose(1, 2))  # 将batch_size和seq_len维度交换
        x = x.transpose(1, 2)
        # 双向LSTM层
        x, _ = self.lstm(x)
        
        # 注意力机制
        attention_weights = self.softmax(self.attention(x).squeeze(2))
        x = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)  # 加权求和

        # 输出层
        out = self.fc(x)
        return out

# BiAGRU模型

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)

        # 初始化注意力权重
        nn.init.xavier_uniform_(self.att_weights.data)

    def forward(self, x):
        # x.shape: (batch_size, seq_len, hidden_size)
        scores = torch.bmm(x, self.att_weights.permute(1, 0).unsqueeze(0).repeat(x.size(0), 1, 1))
        # scores.shape: (batch_size, seq_len, 1)
        scores = self.softmax(scores.squeeze(2))
        # scores.shape: (batch_size, seq_len)
        weighted_output = torch.bmm(x.transpose(1, 2), scores.unsqueeze(2)).squeeze(2)
        # weighted_output.shape: (batch_size, hidden_size)
        return weighted_output, scores

class BiAGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super(BiAGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 定义双向GRU层
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        
        # 定义注意力层
        self.attention = Attention(hidden_size * 2)  # 因为是双向，所以尺寸是hidden_size的两倍
        
        # 定义输出层
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 同样是因为双向

    def forward(self, x):
        # x.shape: (batch_size, seq_len, input_size)
        gru_out, _ = self.gru(x)
        # gru_out.shape: (batch_size, seq_len, hidden_size * 2)
        attn_out, attn_scores = self.attention(gru_out)
        # attn_out.shape: (batch_size, hidden_size * 2)
        output = self.fc(attn_out)
        # output.shape: (batch_size, output_size)
        return output

#Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).unsqueeze(1)
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(input_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=nhid, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)
        self.encoder = nn.Linear(input_size, input_size)
        self.decoder = nn.Linear(input_size, output_size)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.input_size)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        # x = x.unsqueeze(1)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        #print(x.shape)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(DLinear, self).__init__()
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs['individual']
        self.channels = configs['enc_in']

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.channels,self.pred_len)
            self.Linear_Trend = nn.Linear(self.channels,self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        seasonal_init = seasonal_init.squeeze(2)
        trend_init = trend_init.squeeze(2)
        # print(seasonal_init.shape)
        # print(trend_init.shape)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
            # print(seasonal_output.shape)
            # print(trend_output.shape)

        x = seasonal_output + trend_output
        return x # to [Batch, Output length, Channel]


# 通用训练器
class Trainer:
    def __init__(self, model, loaders, criterion, optimizer, num_epochs,device):
        self.model = model
        self.loaders = loaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
    
    def train(self):
        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            # 打印当前的内存使用情况
            # snapshot = tracemalloc.take_snapshot()
            # top_stats = snapshot.statistics('lineno')

            # print("[ Top 10 ]")
            # for stat in top_stats[:10]:
            #     print(stat)
            self.model.train()
            total_batches = len(self.loaders['train'])
            for batch_idx, (inputs, targets) in enumerate(self.loaders['train']):
                inputs, targets = inputs.to(self.device), targets.to(self.device)  # 移动数据到设备上
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                outputs = outputs.squeeze(1)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                # 打印进度
                progress = ((batch_idx + 1) / total_batches) * 100
                print(f'Epoch {epoch+1}/{self.num_epochs}, Batch {batch_idx+1}/{total_batches}, Progress: {progress:.2f}%', end='\r')

            # 在每个epoch结束时打印损失
            print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item()}')
            self.evaluate(self.loaders['test'])

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)  # 移动数据到设备上
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f'Average Loss: {avg_loss}')
        return avg_loss

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


#通用预测器，注意输出的格式是tensor还是numpy
class Predictor:
    def __init__(self, model, scaler,dataset,device):
        self.model = model
        self.scaler = scaler
        self.dataset = dataset
        self.device = device

    def predict(self):
        self.model.eval()
        X_tensor = self.dataset.get_data()[1].to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        predictions = predictions.cpu().numpy()
        return predictions 
    def plot_and_save_results(self, true_values, predictions, file_name='predictions.png'):
        plt.figure(figsize=(12, 6))
        plt.plot(true_values, label='Actual Values', color='blue')
        plt.plot(predictions, label='Predictions', color='red', linestyle='--')
        plt.title('Comparison of Actual and Predicted Values')
        plt.xlabel('Sample Index')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.savefig(file_name, dpi=350)
        plt.show()